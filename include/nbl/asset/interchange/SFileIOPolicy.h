// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_S_FILE_IO_POLICY_H_INCLUDED_
#define _NBL_ASSET_S_FILE_IO_POLICY_H_INCLUDED_


#include <algorithm>
#include <cstdint>


namespace nbl::asset
{

struct SFileIOPolicy
{
    struct SRuntimeTuning
    {
        enum class Mode : uint8_t
        {
            None,
            Heuristic,
            Hybrid
        };

        Mode mode = Mode::Heuristic;
        float maxOverheadRatio = 0.05f;
        float samplingBudgetRatio = 0.05f;
        float minExpectedGainRatio = 0.03f;
        uint32_t maxWorkers = 0u;
        uint32_t samplingMaxCandidates = 4u;
        uint32_t samplingPasses = 1u;
        uint64_t samplingMinWorkUnits = 0ull;
        uint32_t targetChunksPerWorker = 4u;
    };

    enum class Strategy : uint8_t
    {
        Auto,
        WholeFile,
        Chunked
    };

    Strategy strategy = Strategy::Auto;
    bool strict = false;
    uint64_t wholeFileThresholdBytes = 64ull * 1024ull * 1024ull;
    uint64_t chunkSizeBytes = 4ull * 1024ull * 1024ull;
    uint64_t maxStagingBytes = 256ull * 1024ull * 1024ull;
    SRuntimeTuning runtimeTuning = {};
};

struct SResolvedFileIOPolicy
{
    enum class Strategy : uint8_t
    {
        WholeFile,
        Chunked
    };

    Strategy strategy = Strategy::Chunked;
    uint64_t chunkSizeBytes = 0ull;
    bool valid = true;
    const char* reason = "ok";
};

inline SResolvedFileIOPolicy resolveFileIOPolicy(const SFileIOPolicy& _policy, const uint64_t byteCount, const bool sizeKnown = true)
{
    constexpr uint64_t MIN_CHUNK_SIZE = 64ull * 1024ull;

    const uint64_t maxStaging = std::max(_policy.maxStagingBytes, MIN_CHUNK_SIZE);
    const uint64_t requestedChunk = std::max(_policy.chunkSizeBytes, MIN_CHUNK_SIZE);
    const uint64_t chunkSize = std::min(requestedChunk, maxStaging);

    auto makeChunked = [&](const char* reason) -> SResolvedFileIOPolicy
    {
        return SResolvedFileIOPolicy{
            .strategy = SResolvedFileIOPolicy::Strategy::Chunked,
            .chunkSizeBytes = chunkSize,
            .valid = true,
            .reason = reason
        };
    };
    auto makeWhole = [&](const char* reason) -> SResolvedFileIOPolicy
    {
        return SResolvedFileIOPolicy{
            .strategy = SResolvedFileIOPolicy::Strategy::WholeFile,
            .chunkSizeBytes = chunkSize,
            .valid = true,
            .reason = reason
        };
    };

    switch (_policy.strategy)
    {
        case SFileIOPolicy::Strategy::WholeFile:
        {
            if (sizeKnown && byteCount <= maxStaging)
                return makeWhole("requested_whole_file");
            if (_policy.strict)
            {
                return SResolvedFileIOPolicy{
                    .strategy = SResolvedFileIOPolicy::Strategy::WholeFile,
                    .chunkSizeBytes = chunkSize,
                    .valid = false,
                    .reason = "whole_file_not_feasible_strict"
                };
            }
            return makeChunked(sizeKnown ? "whole_file_not_feasible_fallback_chunked" : "whole_file_unknown_size_fallback_chunked");
        }
        case SFileIOPolicy::Strategy::Chunked:
            return makeChunked("requested_chunked");
        case SFileIOPolicy::Strategy::Auto:
        default:
        {
            if (!sizeKnown)
                return makeChunked("auto_unknown_size");
            const uint64_t wholeThreshold = std::min(_policy.wholeFileThresholdBytes, maxStaging);
            if (byteCount <= wholeThreshold)
                return makeWhole("auto_small_enough_for_whole_file");
            return makeChunked("auto_too_large_for_whole_file");
        }
    }
}

inline const char* toString(const SFileIOPolicy::Strategy strategy)
{
    switch (strategy)
    {
        case SFileIOPolicy::Strategy::Auto:
            return "auto";
        case SFileIOPolicy::Strategy::WholeFile:
            return "whole";
        case SFileIOPolicy::Strategy::Chunked:
            return "chunked";
        default:
            return "unknown";
    }
}

inline const char* toString(const SResolvedFileIOPolicy::Strategy strategy)
{
    switch (strategy)
    {
        case SResolvedFileIOPolicy::Strategy::WholeFile:
            return "whole";
        case SResolvedFileIOPolicy::Strategy::Chunked:
            return "chunked";
        default:
            return "unknown";
    }
}

}

#endif
