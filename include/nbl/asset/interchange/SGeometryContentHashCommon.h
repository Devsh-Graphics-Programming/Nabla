// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_S_GEOMETRY_CONTENT_HASH_COMMON_H_INCLUDED_
#define _NBL_ASSET_S_GEOMETRY_CONTENT_HASH_COMMON_H_INCLUDED_


#include "nbl/asset/ICPUPolygonGeometry.h"
#include "nbl/asset/interchange/SLoaderRuntimeTuning.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>


namespace nbl::asset
{

inline void collectGeometryBuffers(
    ICPUPolygonGeometry* geometry,
    core::vector<core::smart_refctd_ptr<ICPUBuffer>>& buffers)
{
    buffers.clear();
    if (!geometry)
        return;

    auto appendViewBuffer = [&buffers](const IGeometry<ICPUBuffer>::SDataView& view) -> void
    {
        if (!view || !view.src.buffer)
            return;
        for (const auto& existing : buffers)
        {
            if (existing.get() == view.src.buffer.get())
                return;
        }
        buffers.push_back(core::smart_refctd_ptr<ICPUBuffer>(view.src.buffer));
    };

    appendViewBuffer(geometry->getPositionView());
    appendViewBuffer(geometry->getIndexView());
    appendViewBuffer(geometry->getNormalView());
    for (const auto& view : *geometry->getAuxAttributeViews())
        appendViewBuffer(view);
    for (const auto& view : *geometry->getJointWeightViews())
    {
        appendViewBuffer(view.indices);
        appendViewBuffer(view.weights);
    }
    if (auto jointOBB = geometry->getJointOBBView(); jointOBB)
        appendViewBuffer(*jointOBB);
}

inline void recomputeGeometryContentHashesParallel(ICPUPolygonGeometry* geometry, const SFileIOPolicy& ioPolicy)
{
    if (!geometry)
        return;

    core::vector<core::smart_refctd_ptr<ICPUBuffer>> buffers;
    collectGeometryBuffers(geometry, buffers);
    if (buffers.empty())
        return;

    core::vector<size_t> pending;
    pending.reserve(buffers.size());
    uint64_t totalBytes = 0ull;
    for (size_t i = 0ull; i < buffers.size(); ++i)
    {
        auto& buffer = buffers[i];
        if (!buffer || buffer->getContentHash() != IPreHashed::INVALID_HASH)
            continue;
        totalBytes += static_cast<uint64_t>(buffer->getSize());
        pending.push_back(i);
    }
    if (pending.empty())
        return;

    const size_t hw = resolveLoaderHardwareThreads();
    const uint8_t* hashSampleData = nullptr;
    uint64_t hashSampleBytes = 0ull;
    for (const auto pendingIx : pending)
    {
        auto& buffer = buffers[pendingIx];
        const auto* ptr = reinterpret_cast<const uint8_t*>(buffer->getPointer());
        if (!ptr)
            continue;
        hashSampleData = ptr;
        hashSampleBytes = std::min<uint64_t>(static_cast<uint64_t>(buffer->getSize()), 128ull << 10);
        if (hashSampleBytes > 0ull)
            break;
    }

    SLoaderRuntimeTuningRequest tuningRequest = {};
    tuningRequest.inputBytes = totalBytes;
    tuningRequest.totalWorkUnits = pending.size();
    tuningRequest.minBytesPerWorker = std::max<uint64_t>(1ull, loaderRuntimeCeilDiv(totalBytes, static_cast<uint64_t>(pending.size())));
    tuningRequest.hardwareThreads = static_cast<uint32_t>(hw);
    tuningRequest.hardMaxWorkers = static_cast<uint32_t>(std::min(pending.size(), hw));
    tuningRequest.targetChunksPerWorker = 1u;
    tuningRequest.sampleData = hashSampleData;
    tuningRequest.sampleBytes = hashSampleBytes;
    const auto tuning = tuneLoaderRuntime(ioPolicy, tuningRequest);
    const size_t workerCount = std::min(tuning.workerCount, pending.size());

    if (workerCount > 1ull)
    {
        loaderRuntimeDispatchWorkers(workerCount, [&](const size_t workerIx)
        {
            const size_t beginIx = (pending.size() * workerIx) / workerCount;
            const size_t endIx = (pending.size() * (workerIx + 1ull)) / workerCount;
            for (size_t i = beginIx; i < endIx; ++i)
            {
                auto& buffer = buffers[pending[i]];
                buffer->setContentHash(buffer->computeContentHash());
            }
        });
        return;
    }

    for (const auto pendingIx : pending)
    {
        auto& buffer = buffers[pendingIx];
        buffer->setContentHash(buffer->computeContentHash());
    }
}

}

#endif
