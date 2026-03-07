// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_S_GEOMETRY_CONTENT_HASH_H_INCLUDED_
#define _NBL_ASSET_S_GEOMETRY_CONTENT_HASH_H_INCLUDED_


#include "nbl/asset/IPreHashed.h"
#include "nbl/asset/utils/CPolygonGeometryManipulator.h"
#include "nbl/core/hash/blake.h"


namespace nbl::asset
{

class SPolygonGeometryContentHash
{
    public:
        using mode_t = CPolygonGeometryManipulator::EContentHashMode;

        static inline void collectBuffers(
            const ICPUPolygonGeometry* geometry,
            core::vector<core::smart_refctd_ptr<ICPUBuffer>>& buffers)
        {
            CPolygonGeometryManipulator::collectUniqueBuffers(geometry, buffers);
        }

        static inline void reset(ICPUPolygonGeometry* geometry)
        {
            core::vector<core::smart_refctd_ptr<ICPUBuffer>> buffers;
            collectBuffers(geometry, buffers);
            for (auto& buffer : buffers)
                if (buffer)
                    buffer->setContentHash(IPreHashed::INVALID_HASH);
        }

        // Composes a geometry hash from indexing metadata and the current content hashes of referenced buffers.
        // It does not compute missing buffer content hashes. Any buffer without a content hash contributes INVALID_HASH.
        static inline core::blake3_hash_t composeHashFromBufferContentHashes(const ICPUPolygonGeometry* geometry)
        {
            if (!geometry)
                return IPreHashed::INVALID_HASH;

            core::blake3_hasher hashBuilder = {};
            if (const auto* indexing = geometry->getIndexingCallback(); indexing)
            {
                hashBuilder << indexing->degree();
                hashBuilder << indexing->rate();
                hashBuilder << indexing->knownTopology();
            }

            core::vector<core::smart_refctd_ptr<ICPUBuffer>> buffers;
            collectBuffers(geometry, buffers);
            for (const auto& buffer : buffers)
                hashBuilder << (buffer ? buffer->getContentHash() : IPreHashed::INVALID_HASH);
            return static_cast<core::blake3_hash_t>(hashBuilder);
        }

        static inline core::blake3_hash_t computeMissing(ICPUPolygonGeometry* geometry, const SFileIOPolicy& ioPolicy)
        {
            CPolygonGeometryManipulator::computeMissingContentHashesParallel(geometry, ioPolicy);
            return composeHashFromBufferContentHashes(geometry);
        }

        static inline core::blake3_hash_t recompute(ICPUPolygonGeometry* geometry, const SFileIOPolicy& ioPolicy)
        {
            CPolygonGeometryManipulator::recomputeContentHashesParallel(geometry, ioPolicy);
            return composeHashFromBufferContentHashes(geometry);
        }
};

}

#endif
