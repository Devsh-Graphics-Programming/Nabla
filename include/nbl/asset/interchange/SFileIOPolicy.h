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
        // Runtime tuning strategy for worker/chunk selection.
        enum class Mode : uint8_t
        {
            // Disable runtime tuning. Use static heuristics only.
            None,
            // Use deterministic heuristics derived from input size and hardware.
            Heuristic,
            // Use heuristics and optionally refine with lightweight sampling.
            Hybrid
        };

        // Runtime tuning mode.
        Mode mode = Mode::Heuristic;
        // Maximum acceptable tuning overhead as a fraction of estimated full workload time.
        float maxOverheadRatio = 0.05f;
        // Maximum sampling budget as a fraction of estimated full workload time.
        float samplingBudgetRatio = 0.05f;
        // Minimum expected gain required to keep extra workers enabled.
        float minExpectedGainRatio = 0.03f;
        // Hard cap for worker count. 0 means auto.
        uint32_t maxWorkers = 0u;
        // Reserved hardware threads not used by the loader. Prevents full CPU saturation.
        uint32_t workerHeadroom = 2u;
        // Maximum number of worker-count candidates tested in hybrid mode.
        uint32_t samplingMaxCandidates = 4u;
        // Number of benchmark passes per candidate in hybrid mode.
        uint32_t samplingPasses = 1u;
        // Minimum work units required before hybrid sampling is allowed. 0 means auto.
        uint64_t samplingMinWorkUnits = 0ull;
        // Target chunk count assigned to each worker for loader stages.
        uint32_t targetChunksPerWorker = 4u;
        // Target chunk count assigned to each worker for hash stages.
        uint32_t hashTaskTargetChunksPerWorker = 1u;
        // Hash inlining threshold. Inputs up to this size prefer inline hash build.
        uint64_t hashInlineThresholdBytes = 1ull << 20;
        // Lower bound for sampled byte count in hybrid mode.
        uint64_t minSampleBytes = 4ull << 10;
        // Upper bound for sampled byte count in hybrid mode.
        uint64_t maxSampleBytes = 128ull << 10;
        // Payload size threshold for tiny-IO anomaly detection.
        uint64_t tinyIoPayloadThresholdBytes = 1ull << 20;
        // Average operation size threshold for tiny-IO anomaly detection.
        uint64_t tinyIoAvgBytesThreshold = 1024ull;
        // Minimum operation size threshold for tiny-IO anomaly detection.
        uint64_t tinyIoMinBytesThreshold = 64ull;
        // Minimum operation count required to report tiny-IO anomaly.
        uint64_t tinyIoMinCallCount = 1024ull;
    };

    // File IO strategy selection mode.
    enum class Strategy : uint8_t
    {
        // Pick whole-file or chunked dynamically based on file size and policy limits.
        Auto,
        // Force whole-file path. May fallback when not feasible unless strict=true.
        WholeFile,
        // Force chunked path.
        Chunked
    };

    // Requested IO strategy.
    Strategy strategy = Strategy::Auto;
    // If true and requested strategy is not feasible then resolution fails instead of fallback.
    bool strict = false;
    // Maximum payload size allowed for whole-file strategy in auto mode.
    uint64_t wholeFileThresholdBytes = 64ull * 1024ull * 1024ull;
    // Chunk size used by chunked strategy.
    uint64_t chunkSizeBytes = 4ull * 1024ull * 1024ull;
    // Maximum staging allocation allowed for whole-file strategy.
    uint64_t maxStagingBytes = 256ull * 1024ull * 1024ull;
    // Runtime tuning controls used by loaders and hash stages.
    SRuntimeTuning runtimeTuning = {};
};

struct SResolvedFileIOPolicy
{
    // Strategy selected after resolving SFileIOPolicy against runtime constraints.
    enum class Strategy : uint8_t
    {
        WholeFile,
        Chunked
    };

    // Effective strategy chosen by resolver.
    Strategy strategy = Strategy::Chunked;
    // Effective chunk size. Also set for whole-file for telemetry consistency.
    uint64_t chunkSizeBytes = 0ull;
    // False when strict policy cannot be satisfied.
    bool valid = true;
    // Human-readable resolver reason used in logs and diagnostics.
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
