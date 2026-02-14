// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_S_INTERCHANGE_IO_COMMON_H_INCLUDED_
#define _NBL_ASSET_S_INTERCHANGE_IO_COMMON_H_INCLUDED_


#include "nbl/asset/interchange/SFileIOPolicy.h"
#include "nbl/system/IFile.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>


namespace nbl::asset
{

struct SFileIOTelemetry
{
    uint64_t callCount = 0ull;
    uint64_t totalBytes = 0ull;
    uint64_t minBytes = std::numeric_limits<uint64_t>::max();

    inline void account(const uint64_t bytes)
    {
        ++callCount;
        totalBytes += bytes;
        if (bytes < minBytes)
            minBytes = bytes;
    }

    inline uint64_t getMinOrZero() const
    {
        return callCount ? minBytes : 0ull;
    }

    inline uint64_t getAvgOrZero() const
    {
        return callCount ? (totalBytes / callCount) : 0ull;
    }
};

using SFileReadTelemetry = SFileIOTelemetry;
using SFileWriteTelemetry = SFileIOTelemetry;

inline bool isTinyIOTelemetryLikely(
    const SFileIOTelemetry& telemetry,
    const uint64_t payloadBytes,
    const uint64_t bigPayloadThresholdBytes = (1ull << 20),
    const uint64_t lowAvgBytesThreshold = 1024ull,
    const uint64_t tinyChunkBytesThreshold = 64ull,
    const uint64_t tinyChunkCallsThreshold = 1024ull)
{
    if (payloadBytes <= bigPayloadThresholdBytes)
        return false;

    const uint64_t minBytes = telemetry.getMinOrZero();
    const uint64_t avgBytes = telemetry.getAvgOrZero();
    return
        avgBytes < lowAvgBytesThreshold ||
        (minBytes < tinyChunkBytesThreshold && telemetry.callCount > tinyChunkCallsThreshold);
}

inline bool isTinyIOTelemetryLikely(const SFileIOTelemetry& telemetry, const uint64_t payloadBytes, const SFileIOPolicy& ioPolicy)
{
    return isTinyIOTelemetryLikely(
        telemetry,
        payloadBytes,
        ioPolicy.runtimeTuning.tinyIoPayloadThresholdBytes,
        ioPolicy.runtimeTuning.tinyIoAvgBytesThreshold,
        ioPolicy.runtimeTuning.tinyIoMinBytesThreshold,
        ioPolicy.runtimeTuning.tinyIoMinCallCount);
}

inline bool readFileExact(system::IFile* file, void* dst, const size_t offset, const size_t bytes, SFileReadTelemetry* ioTelemetry = nullptr)
{
    if (!file || (!dst && bytes != 0ull))
        return false;
    if (bytes == 0ull)
        return true;

    system::IFile::success_t success;
    file->read(success, dst, offset, bytes);
    if (success && ioTelemetry)
        ioTelemetry->account(success.getBytesProcessed());
    return success && success.getBytesProcessed() == bytes;
}

inline bool readFileWithPolicy(system::IFile* file, uint8_t* dst, const size_t offset, const size_t bytes, const SResolvedFileIOPolicy& ioPlan, SFileReadTelemetry* ioTelemetry = nullptr)
{
    if (!file || (!dst && bytes != 0ull))
        return false;
    if (bytes == 0ull)
        return true;

    switch (ioPlan.strategy)
    {
        case SResolvedFileIOPolicy::Strategy::WholeFile:
            return readFileExact(file, dst, offset, bytes, ioTelemetry);
        case SResolvedFileIOPolicy::Strategy::Chunked:
        default:
        {
            size_t bytesRead = 0ull;
            while (bytesRead < bytes)
            {
                const size_t toRead = static_cast<size_t>(std::min<uint64_t>(ioPlan.chunkSizeBytes, bytes - bytesRead));
                system::IFile::success_t success;
                file->read(success, dst + bytesRead, offset + bytesRead, toRead);
                if (!success)
                    return false;
                const size_t processed = success.getBytesProcessed();
                if (processed == 0ull)
                    return false;
                if (ioTelemetry)
                    ioTelemetry->account(processed);
                bytesRead += processed;
            }
            return true;
        }
    }
}

inline bool readFileWithPolicyTimed(system::IFile* file, uint8_t* dst, const size_t offset, const size_t bytes, const SResolvedFileIOPolicy& ioPlan, double* ioMs = nullptr, SFileReadTelemetry* ioTelemetry = nullptr)
{
    using clock_t = std::chrono::high_resolution_clock;
    const auto ioStart = clock_t::now();
    const bool ok = readFileWithPolicy(file, dst, offset, bytes, ioPlan, ioTelemetry);
    if (ioMs)
        *ioMs = std::chrono::duration<double, std::milli>(clock_t::now() - ioStart).count();
    return ok;
}

inline bool writeFileWithPolicyAtOffset(system::IFile* file, const SResolvedFileIOPolicy& ioPlan, const uint8_t* data, size_t byteCount, size_t& fileOffset, SFileWriteTelemetry* ioTelemetry = nullptr)
{
    if (!file || (!data && byteCount != 0ull))
        return false;
    if (byteCount == 0ull)
        return true;

    size_t writtenTotal = 0ull;
    while (writtenTotal < byteCount)
    {
        const size_t toWrite =
            ioPlan.strategy == SResolvedFileIOPolicy::Strategy::WholeFile ?
                (byteCount - writtenTotal) :
                static_cast<size_t>(std::min<uint64_t>(ioPlan.chunkSizeBytes, byteCount - writtenTotal));
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
    return true;
}

inline bool writeFileWithPolicy(system::IFile* file, const SResolvedFileIOPolicy& ioPlan, const uint8_t* data, size_t byteCount, SFileWriteTelemetry* ioTelemetry = nullptr)
{
    size_t fileOffset = 0ull;
    return writeFileWithPolicyAtOffset(file, ioPlan, data, byteCount, fileOffset, ioTelemetry);
}

inline bool writeTwoBuffersWithPolicy(system::IFile* file, const SResolvedFileIOPolicy& ioPlan, const uint8_t* dataA, size_t byteCountA, const uint8_t* dataB, size_t byteCountB, SFileWriteTelemetry* ioTelemetry = nullptr)
{
    size_t fileOffset = 0ull;
    if (!writeFileWithPolicyAtOffset(file, ioPlan, dataA, byteCountA, fileOffset, ioTelemetry))
        return false;
    return writeFileWithPolicyAtOffset(file, ioPlan, dataB, byteCountB, fileOffset, ioTelemetry);
}

}

#endif
