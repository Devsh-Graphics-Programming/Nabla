// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_S_INTERCHANGE_IO_H_INCLUDED_
#define _NBL_ASSET_S_INTERCHANGE_IO_H_INCLUDED_
#include "nbl/asset/interchange/SFileIOPolicy.h"
#include "nbl/system/IFile.h"
#include <algorithm>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
namespace nbl::asset
{
//! Shared read/write helpers that execute a resolved IO plan and collect simple telemetry.
class SInterchangeIO
{
    public:
        //! Tracks IO call count and byte distribution for tiny-io diagnostics.
        struct STelemetry
        {
            uint64_t callCount = 0ull; //!< Number of IO calls recorded.
            uint64_t totalBytes = 0ull; //!< Sum of processed bytes across all calls.
            uint64_t minBytes = std::numeric_limits<uint64_t>::max(); //!< Smallest processed byte count observed so far.

            inline void account(const uint64_t bytes)
            {
                ++callCount;
                totalBytes += bytes;
                if (bytes < minBytes)
                    minBytes = bytes;
            }

            inline uint64_t getMinOrZero() const { return callCount ? minBytes : 0ull; }
            inline uint64_t getAvgOrZero() const { return callCount ? (totalBytes / callCount) : 0ull; }
        };
        using SReadTelemetry = STelemetry;
        using SWriteTelemetry = STelemetry;
        //! Flags large payloads that were served through suspiciously small IO calls.
        //! Defaults are 1 MiB, 1 KiB, 64 B, and 1024 calls.
        static inline bool isTinyIOTelemetryLikely(const STelemetry& telemetry, const uint64_t payloadBytes, const uint64_t bigPayloadThresholdBytes = (1ull << 20), const uint64_t lowAvgBytesThreshold = 1024ull, const uint64_t tinyChunkBytesThreshold = 64ull, const uint64_t tinyChunkCallsThreshold = 1024ull)
        {
            if (payloadBytes <= bigPayloadThresholdBytes)
                return false;
            const uint64_t minBytes = telemetry.getMinOrZero();
            const uint64_t avgBytes = telemetry.getAvgOrZero();
            return avgBytes < lowAvgBytesThreshold || (minBytes < tinyChunkBytesThreshold && telemetry.callCount > tinyChunkCallsThreshold);
        }
        //! Same tiny-io heuristic but pulls thresholds from the resolved IO policy.
        static inline bool isTinyIOTelemetryLikely(const STelemetry& telemetry, const uint64_t payloadBytes, const SFileIOPolicy& ioPolicy) { return isTinyIOTelemetryLikely(telemetry, payloadBytes, ioPolicy.runtimeTuning.tinyIoPayloadThresholdBytes, ioPolicy.runtimeTuning.tinyIoAvgBytesThreshold, ioPolicy.runtimeTuning.tinyIoMinBytesThreshold, ioPolicy.runtimeTuning.tinyIoMinCallCount); }
        //! Issues one read request and verifies that the full byte count was returned.
        static inline bool readFileExact(system::IFile* file, void* dst, const size_t offset, const size_t bytes, SReadTelemetry* ioTelemetry = nullptr)
        {
            if (!file || (!dst && bytes != 0ull)) return false;
            if (bytes == 0ull) return true;
            system::IFile::success_t success;
            file->read(success, dst, offset, bytes);
            if (success && ioTelemetry) ioTelemetry->account(success.getBytesProcessed());
            return success && success.getBytesProcessed() == bytes;
        }

        /**
            Reads a byte range using the resolved whole-file or chunked strategy.
            When `ioTime` is non-null it also reports wall time in `TimeUnit`.
            Default `TimeUnit` is milliseconds.
        */
        template<typename TimeUnit = std::chrono::duration<double, std::milli>>
        requires std::same_as<TimeUnit, std::chrono::duration<typename TimeUnit::rep, typename TimeUnit::period>>
        static inline bool readFileWithPolicy(system::IFile* file, void* dst, const size_t offset, const size_t bytes, const SResolvedFileIOPolicy& ioPlan, SReadTelemetry* ioTelemetry = nullptr, TimeUnit* ioTime = nullptr)
        {
            using clock_t = std::chrono::high_resolution_clock;
            const auto ioStart = ioTime ? clock_t::now() : clock_t::time_point{};
            auto finalize = [&](const bool ok) -> bool { if (ioTime) *ioTime = std::chrono::duration_cast<TimeUnit>(clock_t::now() - ioStart); return ok; };
            if (!file || (!dst && bytes != 0ull))
                return finalize(false);
            if (bytes == 0ull)
                return finalize(true);
            auto* out = reinterpret_cast<uint8_t*>(dst);
            switch (ioPlan.strategy)
            {
                case SResolvedFileIOPolicy::Strategy::WholeFile:
                    return finalize(readFileExact(file, out, offset, bytes, ioTelemetry));
                case SResolvedFileIOPolicy::Strategy::Chunked:
                default:
                {
                    const size_t inFlightDepth = ioPlan.chunkedInFlightDepth;
                    auto inFlight = std::make_unique<SChunkedRequest[]>(inFlightDepth);
                    size_t submitOffset = 0ull;
                    size_t activeCount = 0ull;
                    size_t submitIndex = 0ull;
                    size_t drainIndex = 0ull;
                    const uint64_t chunkSizeBytes = ioPlan.chunkSizeBytes();
                    auto submitChunk = [&]() -> bool {
                        if (submitOffset >= bytes || activeCount >= inFlightDepth)
                            return false;
                        auto& request = inFlight[submitIndex];
                        const size_t toRead = static_cast<size_t>(std::min<uint64_t>(chunkSizeBytes, bytes - submitOffset));
                        request.success.emplace();
                        file->read(*request.success, out + submitOffset, offset + submitOffset, toRead);
                        request.bytes = toRead;
                        request.active = true;
                        submitOffset += toRead;
                        submitIndex = (submitIndex + 1ull) % inFlightDepth;
                        ++activeCount;
                        return true;
                    };
                    auto drainChunk = [&]() -> bool {
                        auto& request = inFlight[drainIndex];
                        if (!request.active)
                            return false;
                        const bool ok = drainChunkedRequest(request, ioTelemetry);
                        drainIndex = (drainIndex + 1ull) % inFlightDepth;
                        --activeCount;
                        return ok;
                    };
                    while (submitOffset < bytes || activeCount)
                    {
                        while (submitChunk()) {}
                        if (activeCount && !drainChunk())
                            return finalize(false);
                    }
                    return finalize(true);
                }
            }
        }
        //! Describes one contiguous output buffer written as part of a larger stream.
        struct SBufferRange
        {
            const void* data = nullptr; //!< Start of the contiguous byte range.
            size_t byteCount = 0ull; //!< Number of bytes to write from `data`.
        };
        //! Writes one or more buffers sequentially at `fileOffset` and advances it on success.
        static inline bool writeBuffersWithPolicyAtOffset(system::IFile* file, const SResolvedFileIOPolicy& ioPlan, const std::span<const SBufferRange> buffers, size_t& fileOffset, SWriteTelemetry* ioTelemetry = nullptr)
        {
            if (!file) return false;
            const uint64_t chunkSizeBytes = ioPlan.chunkSizeBytes();
            for (const auto& buffer : buffers)
            {
                if (!buffer.data && buffer.byteCount != 0ull) return false;
                if (buffer.byteCount == 0ull)
                    continue;
                const auto* data = reinterpret_cast<const uint8_t*>(buffer.data);
                size_t writtenTotal = 0ull;
                if (ioPlan.strategy == SResolvedFileIOPolicy::Strategy::WholeFile)
                {
                    const size_t toWrite = buffer.byteCount;
                    system::IFile::success_t success;
                    file->write(success, data, fileOffset, toWrite);
                    if (!success)
                        return false;
                    const size_t written = success.getBytesProcessed();
                    if (written == 0ull)
                        return false;
                    if (ioTelemetry)
                        ioTelemetry->account(written);
                    writtenTotal += written;
                }
                else
                {
                    const size_t inFlightDepth = ioPlan.chunkedInFlightDepth;
                    auto inFlight = std::make_unique<SChunkedRequest[]>(inFlightDepth);
                    size_t submitOffset = 0ull;
                    size_t activeCount = 0ull;
                    size_t submitIndex = 0ull;
                    size_t drainIndex = 0ull;
                    auto submitChunk = [&]() -> bool {
                        if (submitOffset >= buffer.byteCount || activeCount >= inFlightDepth)
                            return false;
                        auto& request = inFlight[submitIndex];
                        const size_t toWrite = static_cast<size_t>(std::min<uint64_t>(chunkSizeBytes, buffer.byteCount - submitOffset));
                        request.success.emplace();
                        file->write(*request.success, data + submitOffset, fileOffset + submitOffset, toWrite);
                        request.bytes = toWrite;
                        request.active = true;
                        submitOffset += toWrite;
                        submitIndex = (submitIndex + 1ull) % inFlightDepth;
                        ++activeCount;
                        return true;
                    };
                    auto drainChunk = [&]() -> bool {
                        auto& request = inFlight[drainIndex];
                        if (!request.active)
                            return false;
                        const bool ok = drainChunkedRequest(request, ioTelemetry);
                        if (ok)
                            writtenTotal += request.bytes;
                        drainIndex = (drainIndex + 1ull) % inFlightDepth;
                        --activeCount;
                        return ok;
                    };
                    while (submitOffset < buffer.byteCount || activeCount)
                    {
                        while (submitChunk()) {}
                        if (activeCount && !drainChunk())
                            return false;
                    }
                }
                fileOffset += writtenTotal;
            }
            return true;
        }
        //! Writes one or more buffers starting from file offset `0`.
        static inline bool writeBuffersWithPolicy(system::IFile* file, const SResolvedFileIOPolicy& ioPlan, const std::span<const SBufferRange> buffers, SWriteTelemetry* ioTelemetry = nullptr) { size_t fileOffset = 0ull; return writeBuffersWithPolicyAtOffset(file, ioPlan, buffers, fileOffset, ioTelemetry); }
        //! Single-buffer convenience wrapper over `writeBuffersWithPolicyAtOffset`.
        static inline bool writeFileWithPolicyAtOffset(system::IFile* file, const SResolvedFileIOPolicy& ioPlan, const void* data, size_t byteCount, size_t& fileOffset, SWriteTelemetry* ioTelemetry = nullptr) { const SBufferRange buffers[] = {{.data = data, .byteCount = byteCount}}; return writeBuffersWithPolicyAtOffset(file, ioPlan, buffers, fileOffset, ioTelemetry); }
        //! Single-buffer convenience wrapper over `writeBuffersWithPolicy`.
        static inline bool writeFileWithPolicy(system::IFile* file, const SResolvedFileIOPolicy& ioPlan, const void* data, size_t byteCount, SWriteTelemetry* ioTelemetry = nullptr) { const SBufferRange buffers[] = {{.data = data, .byteCount = byteCount}}; return writeBuffersWithPolicy(file, ioPlan, buffers, ioTelemetry); }
    private:
        struct SChunkedRequest
        {
            std::optional<system::IFile::success_t> success = std::nullopt;
            size_t bytes = 0ull;
            bool active = false;
        };
        static inline bool drainChunkedRequest(SChunkedRequest& request, STelemetry* ioTelemetry)
        {
            const size_t processed = request.success ? request.success->getBytesProcessed():0ull;
            request.success.reset();
            request.active = false;
            if (processed != request.bytes || processed == 0ull)
                return false;
            if (ioTelemetry)
                ioTelemetry->account(processed);
            return true;
        }
};
using SFileIOTelemetry = SInterchangeIO::STelemetry;
using SFileReadTelemetry = SInterchangeIO::SReadTelemetry;
using SFileWriteTelemetry = SInterchangeIO::SWriteTelemetry;
}
#endif
