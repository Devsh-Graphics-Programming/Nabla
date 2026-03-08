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
#include <span>
namespace nbl::asset
{
class SInterchangeIO // Shared read/write helpers that execute a resolved IO plan and collect simple telemetry.
{
    public:
        struct STelemetry { uint64_t callCount = 0ull, totalBytes = 0ull, minBytes = std::numeric_limits<uint64_t>::max(); inline void account(const uint64_t bytes) { ++callCount; totalBytes += bytes; if (bytes < minBytes) minBytes = bytes; } inline uint64_t getMinOrZero() const { return callCount ? minBytes : 0ull; } inline uint64_t getAvgOrZero() const { return callCount ? (totalBytes / callCount) : 0ull; } }; // Tracks IO call count and byte distribution for tiny-io diagnostics.
        using SReadTelemetry = STelemetry;
        using SWriteTelemetry = STelemetry;
        /* Default 1 MiB. Default 1 KiB. Default 64 B. Default 1024 calls. */
        static inline bool isTinyIOTelemetryLikely(const STelemetry& telemetry, const uint64_t payloadBytes, const uint64_t bigPayloadThresholdBytes = (1ull << 20), const uint64_t lowAvgBytesThreshold = 1024ull, const uint64_t tinyChunkBytesThreshold = 64ull, const uint64_t tinyChunkCallsThreshold = 1024ull) // Flags large payloads that were served through suspiciously small IO calls.
        {
            if (payloadBytes <= bigPayloadThresholdBytes)
                return false;
            const uint64_t minBytes = telemetry.getMinOrZero();
            const uint64_t avgBytes = telemetry.getAvgOrZero();
            return avgBytes < lowAvgBytesThreshold || (minBytes < tinyChunkBytesThreshold && telemetry.callCount > tinyChunkCallsThreshold);
        }
        static inline bool isTinyIOTelemetryLikely(const STelemetry& telemetry, const uint64_t payloadBytes, const SFileIOPolicy& ioPolicy) { return isTinyIOTelemetryLikely(telemetry, payloadBytes, ioPolicy.runtimeTuning.tinyIoPayloadThresholdBytes, ioPolicy.runtimeTuning.tinyIoAvgBytesThreshold, ioPolicy.runtimeTuning.tinyIoMinBytesThreshold, ioPolicy.runtimeTuning.tinyIoMinCallCount); } // Same tiny-io heuristic but pulls thresholds from the resolved IO policy.
        static inline bool readFileExact(system::IFile* file, void* dst, const size_t offset, const size_t bytes, SReadTelemetry* ioTelemetry = nullptr) // Issues one read request and verifies that the full byte count was returned.
        {
            if (!file || (!dst && bytes != 0ull)) return false;
            if (bytes == 0ull) return true;
            system::IFile::success_t success;
            file->read(success, dst, offset, bytes);
            if (success && ioTelemetry) ioTelemetry->account(success.getBytesProcessed());
            return success && success.getBytesProcessed() == bytes;
        }
        template<typename TimeUnit = std::chrono::duration<double, std::milli>>
        requires std::same_as<TimeUnit, std::chrono::duration<typename TimeUnit::rep, typename TimeUnit::period>>
        static inline bool readFileWithPolicy(system::IFile* file, void* dst, const size_t offset, const size_t bytes, const SResolvedFileIOPolicy& ioPlan, SReadTelemetry* ioTelemetry = nullptr, TimeUnit* ioTime = nullptr) // Reads a byte range using the resolved whole-file or chunked strategy. When ioTime is non-null it also reports wall time in TimeUnit. Default TimeUnit is milliseconds.
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
                    size_t bytesRead = 0ull;
                    const uint64_t chunkSizeBytes = ioPlan.chunkSizeBytes();
                    while (bytesRead < bytes)
                    {
                        const size_t toRead = static_cast<size_t>(std::min<uint64_t>(chunkSizeBytes, bytes - bytesRead));
                        system::IFile::success_t success;
                        file->read(success, out + bytesRead, offset + bytesRead, toRead);
                        if (!success)
                            return finalize(false);
                        const size_t processed = success.getBytesProcessed();
                        if (processed == 0ull)
                            return finalize(false);
                        if (ioTelemetry)
                            ioTelemetry->account(processed);
                        bytesRead += processed;
                    }
                    return finalize(true);
                }
            }
        }
        struct SBufferRange { const void* data = nullptr; size_t byteCount = 0ull; }; // Describes one contiguous output buffer written as part of a larger stream.
        static inline bool writeBuffersWithPolicyAtOffset(system::IFile* file, const SResolvedFileIOPolicy& ioPlan, const std::span<const SBufferRange> buffers, size_t& fileOffset, SWriteTelemetry* ioTelemetry = nullptr) // Writes one or more buffers sequentially at fileOffset and advances it on success.
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
                while (writtenTotal < buffer.byteCount)
                {
                    const size_t toWrite = ioPlan.strategy == SResolvedFileIOPolicy::Strategy::WholeFile ? (buffer.byteCount - writtenTotal) : static_cast<size_t>(std::min<uint64_t>(chunkSizeBytes, buffer.byteCount - writtenTotal));
                    system::IFile::success_t success;
                    file->write(success, data + writtenTotal, fileOffset + writtenTotal, toWrite);
                    if (!success)
                        return false;
                    const size_t written = success.getBytesProcessed();
                    if (written == 0ull)
                        return false;
                    if (ioTelemetry)
                        ioTelemetry->account(written);
                    writtenTotal += written;
                }
                fileOffset += writtenTotal;
            }
            return true;
        }
        static inline bool writeBuffersWithPolicy(system::IFile* file, const SResolvedFileIOPolicy& ioPlan, const std::span<const SBufferRange> buffers, SWriteTelemetry* ioTelemetry = nullptr) { size_t fileOffset = 0ull; return writeBuffersWithPolicyAtOffset(file, ioPlan, buffers, fileOffset, ioTelemetry); } // Writes one or more buffers starting from file offset 0.
        static inline bool writeFileWithPolicyAtOffset(system::IFile* file, const SResolvedFileIOPolicy& ioPlan, const void* data, size_t byteCount, size_t& fileOffset, SWriteTelemetry* ioTelemetry = nullptr) { const SBufferRange buffers[] = {{.data = data, .byteCount = byteCount}}; return writeBuffersWithPolicyAtOffset(file, ioPlan, buffers, fileOffset, ioTelemetry); } // Single-buffer convenience wrapper over writeBuffersWithPolicyAtOffset.
        static inline bool writeFileWithPolicy(system::IFile* file, const SResolvedFileIOPolicy& ioPlan, const void* data, size_t byteCount, SWriteTelemetry* ioTelemetry = nullptr) { const SBufferRange buffers[] = {{.data = data, .byteCount = byteCount}}; return writeBuffersWithPolicy(file, ioPlan, buffers, ioTelemetry); } // Single-buffer convenience wrapper over writeBuffersWithPolicy.
};
using SFileIOTelemetry = SInterchangeIO::STelemetry;
using SFileReadTelemetry = SInterchangeIO::SReadTelemetry;
using SFileWriteTelemetry = SInterchangeIO::SWriteTelemetry;
}
#endif
