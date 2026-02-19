// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_S_FILE_IO_POLICY_H_INCLUDED_
#define _NBL_ASSET_S_FILE_IO_POLICY_H_INCLUDED_


#include "nbl/core/util/bitflag.h"
#include "nbl/system/to_string.h"

#include <algorithm>
#include <cstdint>
#include <string>


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
        uint16_t maxWorkers = 0u;
        // Reserved hardware threads not used by the loader. Prevents full CPU saturation.
        uint8_t workerHeadroom = 2u;
        // Maximum number of worker-count candidates tested in hybrid mode.
        uint8_t samplingMaxCandidates = 4u;
        // Number of benchmark passes per candidate in hybrid mode.
        uint8_t samplingPasses = 1u;
        // Minimum work units required before hybrid sampling is allowed. 0 means auto.
        uint64_t samplingMinWorkUnits = 0ull;
        // Target chunk count assigned to each worker for loader stages.
        uint8_t targetChunksPerWorker = 4u;
        // Target chunk count assigned to each worker for hash stages.
        uint8_t hashTaskTargetChunksPerWorker = 1u;
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
        // Force whole-file strategy. May fallback when not feasible unless strict=true.
        WholeFile,
        // Force chunked strategy.
        Chunked
    };

    enum E_FLAGS : uint8_t
    {
        EF_NONE = 0u,
        EF_STRICT_BIT = 1u << 0u
    };

    static inline constexpr uint8_t MIN_CHUNK_SIZE_LOG2 = 16u;
    static inline constexpr uint8_t MAX_BYTE_SIZE_LOG2 = 63u;

    static inline constexpr uint8_t clampBytesLog2(const uint8_t value, const uint8_t minValue = 0u)
    {
        return std::clamp<uint8_t>(value, minValue, MAX_BYTE_SIZE_LOG2);
    }

    static inline constexpr uint64_t bytesFromLog2(const uint8_t value, const uint8_t minValue = 0u)
    {
        return 1ull << clampBytesLog2(value, minValue);
    }

    // Requested IO strategy.
    Strategy strategy = Strategy::Auto;
    // Resolution flags.
    core::bitflag<E_FLAGS> flags = EF_NONE;
    // Maximum payload size allowed for whole-file strategy in auto mode.
    uint8_t wholeFileThresholdLog2 = 26u; // 64 MiB
    // Chunk size used by chunked strategy encoded as log2(bytes).
    uint8_t chunkSizeLog2 = 22u; // 4 MiB
    // Maximum staging allocation for whole-file strategy encoded as log2(bytes).
    uint8_t maxStagingLog2 = 28u; // 256 MiB
    // Runtime tuning controls used by loaders and hash stages.
    SRuntimeTuning runtimeTuning = {};

    inline bool strict() const
    {
        return flags.hasAnyFlag(EF_STRICT_BIT);
    }

    inline uint64_t wholeFileThresholdBytes() const
    {
        return bytesFromLog2(wholeFileThresholdLog2, MIN_CHUNK_SIZE_LOG2);
    }

    inline uint64_t chunkSizeBytes() const
    {
        return bytesFromLog2(chunkSizeLog2, MIN_CHUNK_SIZE_LOG2);
    }

    inline uint64_t maxStagingBytes() const
    {
        return bytesFromLog2(maxStagingLog2, MIN_CHUNK_SIZE_LOG2);
    }
};

struct SResolvedFileIOPolicy
{
    // Strategy selected after resolving SFileIOPolicy against runtime constraints.
    enum class Strategy : uint8_t
    {
        Invalid = 0u,
        WholeFile,
        Chunked
    };

    SResolvedFileIOPolicy() = default;
    inline SResolvedFileIOPolicy(const SFileIOPolicy& policy, const uint64_t byteCount, const bool sizeKnown = true, const bool fileMappable = false) :
        SResolvedFileIOPolicy(resolve(policy, byteCount, sizeKnown, fileMappable))
    {
    }

    // Effective strategy chosen by resolver. Invalid means strict policy resolution failed.
    Strategy strategy = Strategy::Invalid;
    // Effective chunk size encoded as log2(bytes). Also set for whole-file for telemetry consistency.
    uint8_t chunkSizeLog2 = SFileIOPolicy::MIN_CHUNK_SIZE_LOG2;
    // Human-readable resolver reason used in logs and diagnostics.
    const char* reason = "invalid";

    inline bool isValid() const
    {
        return strategy != Strategy::Invalid;
    }

    inline uint64_t chunkSizeBytes() const
    {
        return SFileIOPolicy::bytesFromLog2(chunkSizeLog2, SFileIOPolicy::MIN_CHUNK_SIZE_LOG2);
    }

    static inline SResolvedFileIOPolicy resolve(const SFileIOPolicy& policy, const uint64_t byteCount, const bool sizeKnown = true, const bool fileMappable = false)
    {
        const uint8_t maxStagingLog2 = SFileIOPolicy::clampBytesLog2(policy.maxStagingLog2, SFileIOPolicy::MIN_CHUNK_SIZE_LOG2);
        const uint8_t chunkSizeLog2 = std::min<uint8_t>(
            SFileIOPolicy::clampBytesLog2(policy.chunkSizeLog2, SFileIOPolicy::MIN_CHUNK_SIZE_LOG2),
            maxStagingLog2);
        const uint64_t maxStaging = SFileIOPolicy::bytesFromLog2(maxStagingLog2, SFileIOPolicy::MIN_CHUNK_SIZE_LOG2);
        const uint64_t wholeThreshold = policy.wholeFileThresholdBytes();

        auto makeResolved = [&](const Strategy strategy, const char* const reason) -> SResolvedFileIOPolicy
        {
            SResolvedFileIOPolicy resolved = {};
            resolved.strategy = strategy;
            resolved.chunkSizeLog2 = chunkSizeLog2;
            resolved.reason = reason;
            return resolved;
        };

        switch (policy.strategy)
        {
            case SFileIOPolicy::Strategy::WholeFile:
            {
                if (fileMappable || (sizeKnown && byteCount <= maxStaging))
                    return makeResolved(Strategy::WholeFile, fileMappable ? "requested_whole_file_mappable" : "requested_whole_file");
                if (policy.strict())
                    return makeResolved(Strategy::Invalid, "whole_file_not_feasible_strict");
                return makeResolved(Strategy::Chunked, sizeKnown ? "whole_file_not_feasible_fallback_chunked" : "whole_file_unknown_size_fallback_chunked");
            }
            case SFileIOPolicy::Strategy::Chunked:
                return makeResolved(Strategy::Chunked, "requested_chunked");
            case SFileIOPolicy::Strategy::Auto:
            default:
            {
                if (!sizeKnown)
                    return makeResolved(fileMappable ? Strategy::WholeFile : Strategy::Chunked, fileMappable ? "auto_unknown_size_mappable_whole_file" : "auto_unknown_size");

                const uint64_t wholeLimit = fileMappable ?
                    std::max<uint64_t>(wholeThreshold, maxStaging) :
                    std::min<uint64_t>(wholeThreshold, maxStaging);
                if (byteCount <= wholeLimit)
                    return makeResolved(Strategy::WholeFile, fileMappable ? "auto_mappable_prefers_whole_file" : "auto_small_enough_for_whole_file");
                return makeResolved(Strategy::Chunked, "auto_too_large_for_whole_file");
            }
        }
    }
};

inline SResolvedFileIOPolicy resolveFileIOPolicy(const SFileIOPolicy& policy, const uint64_t byteCount, const bool sizeKnown = true, const bool fileMappable = false)
{
    return SResolvedFileIOPolicy(policy, byteCount, sizeKnown, fileMappable);
}

inline const char* toString(const SFileIOPolicy::Strategy value)
{
    switch (value)
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

inline const char* toString(const SResolvedFileIOPolicy::Strategy value)
{
    switch (value)
    {
        case SResolvedFileIOPolicy::Strategy::Invalid:
            return "invalid";
        case SResolvedFileIOPolicy::Strategy::WholeFile:
            return "whole";
        case SResolvedFileIOPolicy::Strategy::Chunked:
            return "chunked";
        default:
            return "unknown";
    }
}

}

namespace nbl::system::impl
{
template<>
struct to_string_helper<asset::SFileIOPolicy::Strategy>
{
    static inline std::string __call(const asset::SFileIOPolicy::Strategy value)
    {
        return asset::toString(value);
    }
};

template<>
struct to_string_helper<asset::SResolvedFileIOPolicy::Strategy>
{
    static inline std::string __call(const asset::SResolvedFileIOPolicy::Strategy value)
    {
        return asset::toString(value);
    }
};
}

#endif
