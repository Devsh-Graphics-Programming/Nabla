// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_S_GEOMETRY_CONTENT_HASH_COMMON_H_INCLUDED_
#define _NBL_ASSET_S_GEOMETRY_CONTENT_HASH_COMMON_H_INCLUDED_


#include "nbl/asset/utils/CPolygonGeometryManipulator.h"


namespace nbl::asset
{

class SPolygonGeometryContentHash
{
    public:
        using EMode = CPolygonGeometryManipulator::EContentHashMode;

        static inline void collectBuffers(
            ICPUPolygonGeometry* geometry,
            core::vector<core::smart_refctd_ptr<ICPUBuffer>>& buffers)
        {
            CPolygonGeometryManipulator::collectUniqueBuffers(geometry, buffers);
        }

        static inline void computeParallel(ICPUPolygonGeometry* geometry, const SFileIOPolicy& ioPolicy, const EMode mode = EMode::MissingOnly)
        {
            CPolygonGeometryManipulator::computeContentHashesParallel(geometry, ioPolicy, mode);
        }

        static inline void computeMissingParallel(ICPUPolygonGeometry* geometry, const SFileIOPolicy& ioPolicy)
        {
            CPolygonGeometryManipulator::computeMissingContentHashesParallel(geometry, ioPolicy);
        }

        static inline void recomputeParallel(ICPUPolygonGeometry* geometry, const SFileIOPolicy& ioPolicy)
        {
            CPolygonGeometryManipulator::recomputeContentHashesParallel(geometry, ioPolicy);
        }
};

}

#endif
