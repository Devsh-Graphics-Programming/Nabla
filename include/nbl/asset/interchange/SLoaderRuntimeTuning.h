// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_S_LOADER_RUNTIME_TUNING_H_INCLUDED_
#define _NBL_ASSET_S_LOADER_RUNTIME_TUNING_H_INCLUDED_

#include "nbl/asset/interchange/SFileIOPolicy.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

namespace nbl::asset
{
struct SLoaderRuntimeTuningRequest
{
    uint64_t inputBytes = 0ull;
    uint64_t totalWorkUnits = 0ull;
    uint64_t minWorkUnitsPerWorker = 1ull;
    uint64_t minBytesPerWorker = 1ull;
    uint32_t hardwareThreads = 0u;
    uint32_t hardMaxWorkers = 0u;
    uint32_t targetChunksPerWorker = 0u;
    uint64_t minChunkWorkUnits = 1ull;
    uint64_t maxChunkWorkUnits = std::numeric_limits<uint64_t>::max();
    const uint8_t* sampleData = nullptr;
    uint64_t sampleBytes = 0ull;
    uint32_t samplePasses = 0u;
    uint32_t sampleMaxCandidates = 0u;
    uint64_t sampleMinWorkUnits = 0ull;
};

struct SLoaderRuntimeTuningResult
{
    size_t workerCount = 1ull;
    uint64_t chunkWorkUnits = 1ull;
    size_t chunkCount = 1ull;
};

struct SLoaderRuntimeTuner
{
    private:
        struct SBenchmarkSampleStats
        {
            uint64_t medianNs = 0ull;
            uint64_t minNs = 0ull;
            uint64_t maxNs = 0ull;
            uint64_t totalNs = 0ull;
        };

    public:
        template<typename Fn>
        requires std::invocable<Fn&, size_t>
        static void dispatchWorkers(const size_t workerCount, Fn&& fn)
        {
            if (workerCount <= 1ull)
            {
                fn(0ull);
                return;
            }

            std::vector<std::jthread> workers;
            workers.reserve(workerCount - 1ull);
            for (size_t workerIx = 1ull; workerIx < workerCount; ++workerIx)
                workers.emplace_back([&fn, workerIx]() { fn(workerIx); });
            fn(0ull);
        }
        static constexpr uint64_t ceilDiv(const uint64_t numerator, const uint64_t denominator) { return (numerator + denominator - 1ull) / denominator; }

        template<typename TimeUnit = std::chrono::nanoseconds>
        requires std::same_as<TimeUnit, std::chrono::duration<typename TimeUnit::rep, typename TimeUnit::period>>
        static inline TimeUnit benchmarkSample(const uint8_t* const sampleData, const uint64_t sampleBytes, const size_t workerCount, const uint32_t passes)
        {
            if (!sampleData || sampleBytes == 0ull || workerCount == 0ull)
                return TimeUnit::zero();
            const uint32_t passCount = std::max<uint32_t>(1u, passes);
            std::vector<uint64_t> partial(workerCount, 0ull);
            uint64_t elapsedNs = 0ull;
            using clock_t = std::chrono::steady_clock;
            for (uint32_t passIx = 0u; passIx < passCount; ++passIx)
            {
                const auto passStart = clock_t::now();
                SLoaderRuntimeTuner::dispatchWorkers(workerCount, [&](const size_t workerIx)
                {
                    const uint64_t begin = (sampleBytes * workerIx) / workerCount;
                    const uint64_t end = (sampleBytes * (workerIx + 1ull)) / workerCount;
                    const uint8_t* ptr = sampleData + begin;
                    uint64_t local = 0ull;
                    for (uint64_t i = 0ull, count = end - begin; i < count; ++i)
                        local += static_cast<uint64_t>(ptr[i]);
                    partial[workerIx] ^= local;
                });
                elapsedNs += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(clock_t::now() - passStart).count());
            }
            uint64_t reduced = 0ull;
            for (const uint64_t v : partial)
                reduced ^= v;
            static std::atomic<uint64_t> sink = 0ull;
            sink.fetch_xor(reduced, std::memory_order_relaxed);
            return std::chrono::duration_cast<TimeUnit>(std::chrono::nanoseconds(elapsedNs));
        }

        static inline SBenchmarkSampleStats benchmarkSampleStats(const uint8_t* const sampleData, const uint64_t sampleBytes, const size_t workerCount, const uint32_t passes, const uint32_t observations)
        {
            SBenchmarkSampleStats stats = {};
            if (!sampleData || sampleBytes == 0ull || workerCount == 0ull)
                return stats;
            const uint32_t observationCount = std::max<uint32_t>(1u, observations);
            std::vector<uint64_t> samples;
            samples.reserve(observationCount);
            benchmarkSample(sampleData, sampleBytes, workerCount, 1u);
            for (uint32_t obsIx = 0u; obsIx < observationCount; ++obsIx)
            {
                const uint64_t elapsedNs = static_cast<uint64_t>(benchmarkSample(sampleData, sampleBytes, workerCount, passes).count());
                if (elapsedNs == 0ull)
                    continue;
                stats.totalNs += elapsedNs;
                samples.push_back(elapsedNs);
            }
            if (samples.empty())
                return SBenchmarkSampleStats{};
            std::sort(samples.begin(), samples.end());
            stats.minNs = samples.front();
            stats.maxNs = samples.back();
            if ((samples.size() & 1ull) != 0ull)
                stats.medianNs = samples[samples.size() / 2ull];
            else
                stats.medianNs = (samples[samples.size() / 2ull - 1ull] + samples[samples.size() / 2ull]) / 2ull;
            return stats;
        }

        static inline void appendCandidate(std::vector<size_t>& dst, const size_t candidate) { if (candidate != 0ull && std::find(dst.begin(), dst.end(), candidate) == dst.end()) dst.push_back(candidate); }

        static inline uint64_t resolveSampleBytes(const SFileIOPolicy& ioPolicy, const uint64_t knownInputBytes)
        {
            if (knownInputBytes == 0ull)
                return 0ull;

            const uint64_t minSampleBytes = std::max<uint64_t>(1ull, ioPolicy.runtimeTuning.minSampleBytes);
            const uint64_t maxSampleBytes = std::max<uint64_t>(minSampleBytes, ioPolicy.runtimeTuning.maxSampleBytes);
            const uint64_t cappedMin = std::min<uint64_t>(minSampleBytes, knownInputBytes);
            const uint64_t cappedMax = std::min<uint64_t>(maxSampleBytes, knownInputBytes);
            const uint64_t adaptive = std::max<uint64_t>(knownInputBytes / 64ull, cappedMin);
            return std::clamp<uint64_t>(adaptive, cappedMin, cappedMax);
        }

        static inline bool shouldInlineHashBuild(const SFileIOPolicy& ioPolicy, const uint64_t inputBytes) { return inputBytes <= std::max<uint64_t>(1ull, ioPolicy.runtimeTuning.hashInlineThresholdBytes); }
        static inline size_t resolveHardwareThreads(const uint32_t requested = 0u) { const size_t hw = requested ? static_cast<size_t>(requested) : static_cast<size_t>(std::thread::hardware_concurrency()); return hw ? hw : 1ull; }

        static inline size_t resolveHardMaxWorkers(const size_t hardwareThreads, const uint32_t workerHeadroom)
        {
            const size_t hw = std::max<size_t>(1ull, hardwareThreads), minWorkers = hw >= 2ull ? 2ull : 1ull, headroom = static_cast<size_t>(workerHeadroom);
            if (headroom == 0ull)
                return hw;
            if (hw <= headroom)
                return minWorkers;
            return std::max<size_t>(minWorkers, hw - headroom);
        }

        static inline SLoaderRuntimeTuningResult tune(const SFileIOPolicy& ioPolicy, const SLoaderRuntimeTuningRequest& request)
        {
            using RTMode = SFileIOPolicy::SRuntimeTuning::Mode;
            SLoaderRuntimeTuningResult result = {};
            if (request.totalWorkUnits == 0ull)
            {
                result.chunkWorkUnits = 0ull;
                result.chunkCount = 0ull;
                return result;
            }
            const size_t hw = SLoaderRuntimeTuner::resolveHardwareThreads(request.hardwareThreads);
            size_t maxWorkers = hw;
            if (request.hardMaxWorkers > 0u)
                maxWorkers = std::min(maxWorkers, static_cast<size_t>(request.hardMaxWorkers));
            if (ioPolicy.runtimeTuning.maxWorkers > 0u)
                maxWorkers = std::min(maxWorkers, static_cast<size_t>(ioPolicy.runtimeTuning.maxWorkers));
            maxWorkers = std::max<size_t>(1ull, maxWorkers);
            const uint64_t minWorkUnitsPerWorker = std::max<uint64_t>(1ull, request.minWorkUnitsPerWorker);
            const uint64_t minBytesPerWorker = std::max<uint64_t>(1ull, request.minBytesPerWorker);
            const size_t maxByWork = static_cast<size_t>(SLoaderRuntimeTuner::ceilDiv(request.totalWorkUnits, minWorkUnitsPerWorker));
            const size_t maxByBytes = request.inputBytes ? static_cast<size_t>(SLoaderRuntimeTuner::ceilDiv(request.inputBytes, minBytesPerWorker)) : maxWorkers;
            const bool heuristicEnabled = ioPolicy.runtimeTuning.mode != RTMode::Sequential;
            const bool hybridEnabled = ioPolicy.runtimeTuning.mode == RTMode::Hybrid;
            size_t workerCount = 1ull;
            if (heuristicEnabled)
                workerCount = std::max<size_t>(1ull, std::min({ maxWorkers, maxByWork, maxByBytes }));
            const size_t targetChunksPerWorker = std::max<size_t>(1ull, static_cast<size_t>(request.targetChunksPerWorker ? request.targetChunksPerWorker : ioPolicy.runtimeTuning.targetChunksPerWorker));
            if (workerCount > 1ull && heuristicEnabled)
            {
                const double maxOverheadRatio = std::max(0.0, static_cast<double>(ioPolicy.runtimeTuning.maxOverheadRatio));
                const double minExpectedGainRatio = std::clamp(static_cast<double>(ioPolicy.runtimeTuning.minExpectedGainRatio), 0.0, 0.99);
                while (workerCount > 1ull)
                {
                    const double idealGain = 1.0 - (1.0 / static_cast<double>(workerCount));
                    const double overheadRatio = static_cast<double>(workerCount * targetChunksPerWorker) / static_cast<double>(std::max<uint64_t>(1ull, request.totalWorkUnits));
                    if (idealGain < minExpectedGainRatio || overheadRatio > maxOverheadRatio)
                    {
                        --workerCount;
                        continue;
                    }
                    break;
                }
            }
            const size_t heuristicWorkerCount = std::max<size_t>(1ull, workerCount);
            if (heuristicEnabled && hybridEnabled && request.sampleData != nullptr && request.sampleBytes > 0ull && heuristicWorkerCount > 1ull && maxWorkers > 1ull)
            {
                const uint64_t autoMinSamplingWorkUnits = std::max<uint64_t>(static_cast<uint64_t>(targetChunksPerWorker) * 8ull, static_cast<uint64_t>(maxWorkers * targetChunksPerWorker));
                const uint64_t minSamplingWorkUnits = request.sampleMinWorkUnits ? request.sampleMinWorkUnits : (ioPolicy.runtimeTuning.samplingMinWorkUnits ? ioPolicy.runtimeTuning.samplingMinWorkUnits : autoMinSamplingWorkUnits);
                if (request.totalWorkUnits >= minSamplingWorkUnits)
                {
                    const double samplingBudgetRatio = std::clamp(static_cast<double>(ioPolicy.runtimeTuning.samplingBudgetRatio), 0.0, 0.5);
                    uint64_t effectiveSampleBytes = request.sampleBytes;
                    if (request.inputBytes)
                        effectiveSampleBytes = std::min<uint64_t>(effectiveSampleBytes, request.inputBytes);
                    if (effectiveSampleBytes > 0ull && samplingBudgetRatio > 0.0)
                    {
                        if (request.inputBytes > 0ull)
                        {
                            const uint64_t sampleDivisor = std::max<uint64_t>(4ull, static_cast<uint64_t>(heuristicWorkerCount) * static_cast<uint64_t>(targetChunksPerWorker));
                            const uint64_t adaptiveSampleBytes = std::max<uint64_t>(1ull, request.inputBytes / sampleDivisor);
                            effectiveSampleBytes = std::min<uint64_t>(effectiveSampleBytes, adaptiveSampleBytes);
                        }
                        const uint32_t samplePasses = request.samplePasses ? request.samplePasses : ioPolicy.runtimeTuning.samplingPasses;
                        uint32_t maxCandidates = request.sampleMaxCandidates ? request.sampleMaxCandidates : ioPolicy.runtimeTuning.samplingMaxCandidates;
                        maxCandidates = std::max<uint32_t>(2u, maxCandidates);
                        std::vector<size_t> candidates;
                        candidates.reserve(maxCandidates);
                        appendCandidate(candidates, heuristicWorkerCount);
                        appendCandidate(candidates, heuristicWorkerCount > 1ull ? (heuristicWorkerCount - 1ull) : 1ull);
                        appendCandidate(candidates, std::min(maxWorkers, heuristicWorkerCount + 1ull));
                        if (heuristicWorkerCount > 2ull)
                            appendCandidate(candidates, heuristicWorkerCount - 2ull);
                        if (heuristicWorkerCount + 2ull <= maxWorkers)
                            appendCandidate(candidates, heuristicWorkerCount + 2ull);
                        if (candidates.size() > maxCandidates)
                            candidates.resize(maxCandidates);
                        const auto heuristicStatsProbe = benchmarkSampleStats(request.sampleData, effectiveSampleBytes, heuristicWorkerCount, samplePasses, 2u);
                        if (heuristicStatsProbe.medianNs > 0ull)
                        {
                            const double scale = request.inputBytes ? (static_cast<double>(request.inputBytes) / static_cast<double>(effectiveSampleBytes)) : 1.0;
                            const uint64_t estimatedFullNs = static_cast<uint64_t>(static_cast<double>(heuristicStatsProbe.medianNs) * std::max(1.0, scale));
                            const uint64_t samplingBudgetNs = static_cast<uint64_t>(static_cast<double>(estimatedFullNs) * samplingBudgetRatio);
                            uint64_t spentNs = heuristicStatsProbe.totalNs;
                            const size_t alternativeCandidates = (candidates.size() > 0ull) ? (candidates.size() - 1ull) : 0ull;
                            if (alternativeCandidates > 0ull && spentNs < samplingBudgetNs)
                            {
                                const uint64_t spareBudgetNs = samplingBudgetNs - spentNs;
                                const uint64_t estimatedEvalNs = std::max<uint64_t>(1ull, heuristicStatsProbe.medianNs);
                                const uint64_t estimatedEvaluations = std::max<uint64_t>(1ull, spareBudgetNs / estimatedEvalNs);
                                uint32_t observations = static_cast<uint32_t>(std::clamp<uint64_t>(
                                    estimatedEvaluations / static_cast<uint64_t>(alternativeCandidates),
                                    1ull,
                                    3ull));

                                SBenchmarkSampleStats bestStats = heuristicStatsProbe;
                                size_t bestWorker = heuristicWorkerCount;
                                for (const size_t candidate : candidates)
                                {
                                    if (candidate == heuristicWorkerCount)
                                        continue;
                                    if (spentNs >= samplingBudgetNs)
                                        break;
                                    const auto candidateStats = benchmarkSampleStats(
                                        request.sampleData, effectiveSampleBytes, candidate, samplePasses, observations);
                                    if (candidateStats.medianNs == 0ull)
                                        continue;
                                    spentNs += candidateStats.totalNs;
                                    if (candidateStats.medianNs < bestStats.medianNs)
                                    {
                                        bestStats = candidateStats;
                                        bestWorker = candidate;
                                    }
                                }
                                if (bestWorker != heuristicWorkerCount)
                                {
                                    const double gain = static_cast<double>(heuristicStatsProbe.medianNs - bestStats.medianNs) / static_cast<double>(heuristicStatsProbe.medianNs);
                                    const uint64_t heuristicSpan = heuristicStatsProbe.maxNs - heuristicStatsProbe.minNs;
                                    const uint64_t bestSpan = bestStats.maxNs - bestStats.minNs;
                                    const double heuristicNoise = static_cast<double>(heuristicSpan) / static_cast<double>(std::max<uint64_t>(1ull, heuristicStatsProbe.medianNs));
                                    const double bestNoise = static_cast<double>(bestSpan) / static_cast<double>(std::max<uint64_t>(1ull, bestStats.medianNs));
                                    const double requiredGain = std::max(std::clamp(static_cast<double>(ioPolicy.runtimeTuning.minExpectedGainRatio), 0.0, 0.99), std::clamp(std::max(heuristicNoise, bestNoise) * 1.25, 0.0, 0.99));
                                    if (gain >= requiredGain)
                                        workerCount = bestWorker;
                                }
                            }
                        }
                    }
                }
            }
            result.workerCount = std::max<size_t>(1ull, workerCount);
            const uint64_t minChunkWorkUnits = std::max<uint64_t>(1ull, request.minChunkWorkUnits);
            uint64_t maxChunkWorkUnits = std::max<uint64_t>(minChunkWorkUnits, request.maxChunkWorkUnits);
            const uint64_t desiredChunkCount = static_cast<uint64_t>(std::max<size_t>(1ull, result.workerCount * targetChunksPerWorker));
            uint64_t chunkWorkUnits = SLoaderRuntimeTuner::ceilDiv(request.totalWorkUnits, desiredChunkCount);
            chunkWorkUnits = std::clamp<uint64_t>(chunkWorkUnits, minChunkWorkUnits, maxChunkWorkUnits);

            result.chunkWorkUnits = chunkWorkUnits;
            result.chunkCount = static_cast<size_t>(SLoaderRuntimeTuner::ceilDiv(request.totalWorkUnits, chunkWorkUnits));
            return result;
        }
};

}
#endif
