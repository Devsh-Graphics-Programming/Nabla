// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_S_FILE_IO_POLICY_H_INCLUDED_
#define _NBL_ASSET_S_FILE_IO_POLICY_H_INCLUDED_
#include "nbl/core/util/bitflag.h"
#include "nbl/system/to_string.h"
#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <string>
namespace nbl::asset
{
enum class EFileIOStrategy : uint8_t
{
    Invalid = 0u,
    Auto,
    WholeFile,
    Chunked
};
struct SFileIOPolicy
{
    struct SRuntimeTuning
    {
        enum class Mode : uint8_t
        {
            Sequential,
            None = Sequential,
            Heuristic,
            Hybrid
        };
        Mode mode = Mode::Heuristic;
        float maxOverheadRatio = 0.05f;
        float samplingBudgetRatio = 0.05f;
        float minExpectedGainRatio = 0.03f;
        uint16_t maxWorkers = 0u;
        uint8_t workerHeadroom = 2u;
        uint8_t samplingMaxCandidates = 4u;
        uint8_t samplingPasses = 1u;
        uint64_t samplingMinWorkUnits = 0ull;
        uint8_t targetChunksPerWorker = 4u;
        uint8_t hashTaskTargetChunksPerWorker = 1u;
        uint64_t hashInlineThresholdBytes = 1ull << 20;
        uint64_t minSampleBytes = 4ull << 10;
        uint64_t maxSampleBytes = 128ull << 10;
        uint64_t tinyIoPayloadThresholdBytes = 1ull << 20;
        uint64_t tinyIoAvgBytesThreshold = 1024ull;
        uint64_t tinyIoMinBytesThreshold = 64ull;
        uint64_t tinyIoMinCallCount = 1024ull;
    };

    using Strategy = EFileIOStrategy;

    enum E_FLAGS : uint8_t
    {
        EF_NONE = 0u,
        EF_STRICT_BIT = 1u << 0u
    };

    static inline constexpr uint64_t MIN_CHUNK_SIZE_BYTES = 64ull << 10u;
    static inline constexpr uint8_t MIN_CHUNK_SIZE_LOG2 = static_cast<uint8_t>(std::bit_width(MIN_CHUNK_SIZE_BYTES) - 1u);
    static inline constexpr uint8_t MAX_BYTE_SIZE_LOG2 = std::numeric_limits<uint64_t>::digits - 1u;
    static inline constexpr uint64_t DEFAULT_WHOLE_FILE_THRESHOLD_BYTES = 64ull << 20u;
    static inline constexpr uint64_t DEFAULT_CHUNK_SIZE_BYTES = 4ull << 20u;
    static inline constexpr uint64_t DEFAULT_MAX_STAGING_BYTES = 256ull << 20u;

    static_assert(std::has_single_bit(MIN_CHUNK_SIZE_BYTES));
    static_assert(std::has_single_bit(DEFAULT_WHOLE_FILE_THRESHOLD_BYTES));
    static_assert(std::has_single_bit(DEFAULT_CHUNK_SIZE_BYTES));
    static_assert(std::has_single_bit(DEFAULT_MAX_STAGING_BYTES));

    static inline constexpr uint8_t clampBytesLog2(const uint8_t value, const uint8_t minValue = 0u) { return std::clamp<uint8_t>(value, minValue, MAX_BYTE_SIZE_LOG2); }

    static inline constexpr uint64_t bytesFromLog2(const uint8_t value, const uint8_t minValue = 0u) { return 1ull << clampBytesLog2(value, minValue); }
    Strategy strategy = Strategy::Auto;
    core::bitflag<E_FLAGS> flags = EF_NONE;
    uint8_t wholeFileThresholdLog2 = static_cast<uint8_t>(std::bit_width(DEFAULT_WHOLE_FILE_THRESHOLD_BYTES) - 1u);
    uint8_t chunkSizeLog2 = static_cast<uint8_t>(std::bit_width(DEFAULT_CHUNK_SIZE_BYTES) - 1u);
    uint8_t maxStagingLog2 = static_cast<uint8_t>(std::bit_width(DEFAULT_MAX_STAGING_BYTES) - 1u);
    SRuntimeTuning runtimeTuning = {};

    inline constexpr bool strict() const { return flags.hasAnyFlag(EF_STRICT_BIT); }

    inline constexpr uint64_t wholeFileThresholdBytes() const { return bytesFromLog2(wholeFileThresholdLog2, MIN_CHUNK_SIZE_LOG2); }

    inline constexpr uint64_t chunkSizeBytes() const { return bytesFromLog2(chunkSizeLog2, MIN_CHUNK_SIZE_LOG2); }

    inline constexpr uint64_t maxStagingBytes() const { return bytesFromLog2(maxStagingLog2, MIN_CHUNK_SIZE_LOG2); }
};
struct SResolvedFileIOPolicy
{
    using Strategy = EFileIOStrategy;

    constexpr SResolvedFileIOPolicy() = default;
    inline constexpr SResolvedFileIOPolicy(const SFileIOPolicy& policy, const uint64_t byteCount, const bool sizeKnown = true, const bool fileMappable = false) : SResolvedFileIOPolicy(resolve(policy, byteCount, sizeKnown, fileMappable)) {}
    Strategy strategy = Strategy::Invalid;
    uint8_t chunkSizeLog2 = SFileIOPolicy::MIN_CHUNK_SIZE_LOG2;
    const char* reason = "invalid";

    inline constexpr bool isValid() const { return strategy != Strategy::Invalid; }

    inline constexpr uint64_t chunkSizeBytes() const { return SFileIOPolicy::bytesFromLog2(chunkSizeLog2, SFileIOPolicy::MIN_CHUNK_SIZE_LOG2); }

    static inline constexpr SResolvedFileIOPolicy resolve(const SFileIOPolicy& policy, const uint64_t byteCount, const bool sizeKnown = true, const bool fileMappable = false)
    {
        const uint8_t maxStagingLog2 = SFileIOPolicy::clampBytesLog2(policy.maxStagingLog2, SFileIOPolicy::MIN_CHUNK_SIZE_LOG2);
        const uint8_t chunkSizeLog2 = std::min<uint8_t>(
            SFileIOPolicy::clampBytesLog2(policy.chunkSizeLog2, SFileIOPolicy::MIN_CHUNK_SIZE_LOG2),
            maxStagingLog2);
        const uint64_t maxStaging = SFileIOPolicy::bytesFromLog2(maxStagingLog2, SFileIOPolicy::MIN_CHUNK_SIZE_LOG2);
        const uint64_t wholeThreshold = policy.wholeFileThresholdBytes();

        auto makeResolved = [&](const Strategy strategy, const char* const reason) -> SResolvedFileIOPolicy { SResolvedFileIOPolicy resolved = {}; resolved.strategy = strategy; resolved.chunkSizeLog2 = chunkSizeLog2; resolved.reason = reason; return resolved; };

        switch (policy.strategy)
        {
            case SFileIOPolicy::Strategy::Invalid:
                return makeResolved(Strategy::Invalid, "invalid_requested_strategy");
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
}
namespace nbl::system::impl
{
template<>
struct to_string_helper<asset::EFileIOStrategy>
{
    static inline std::string __call(const asset::EFileIOStrategy value)
    {
        switch (value)
        {
            case asset::EFileIOStrategy::Invalid:
                return "invalid";
            case asset::EFileIOStrategy::Auto:
                return "auto";
            case asset::EFileIOStrategy::WholeFile:
                return "whole";
            case asset::EFileIOStrategy::Chunked:
                return "chunked";
            default:
                return "unknown";
        }
    }
};
}
#endif
