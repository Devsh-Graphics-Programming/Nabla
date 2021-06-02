// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_CPU_MESH_PACKER_H_INCLUDED__
#define __NBL_ASSET_I_CPU_MESH_PACKER_H_INCLUDED__

#include "nbl/asset/utils/IMeshManipulator.h"

namespace nbl
{
namespace asset
{

class ICPUMeshPacker
{
protected:
    struct Triangle
    {
        uint32_t oldIndices[3];
    };

    struct TriangleBatches
    {
        TriangleBatches(uint32_t triCnt)
        {
            triangles = core::vector<Triangle>(triCnt);
        }

        core::vector<Triangle> triangles;
        core::vector<Triangle*> ranges;
    };

    struct IdxBufferParams
    {
        SBufferBinding<ICPUBuffer> idxBuffer;
        uint32_t idxCnt = 0u;
        E_INDEX_TYPE indexType = EIT_UNKNOWN;
    };

    TriangleBatches constructTriangleBatches(const MeshBufferType* meshBuffer, IdxBufferParams idxBufferParams) const
    {
        uint32_t triCnt;
        const bool success = IMeshManipulator::getPolyCount(triCnt, meshBuffer);
        assert(success);

        const uint32_t batchCnt = calcBatchCountBound(triCnt);
        assert(batchCnt != 0u);

        struct MortonTriangle
        {
            MortonTriangle() = default;

            MortonTriangle(uint16_t fixedPointPos[3], float area)
            {
                auto tmp = reinterpret_cast<uint16_t*>(key);
                std::copy_n(fixedPointPos, 3u, tmp);
                tmp[3] = core::Float16Compressor::compress(area);
            }

            void complete(float maxArea)
            {
                auto tmp = reinterpret_cast<const uint16_t*>(key);
                const float area = core::Float16Compressor::decompress(tmp[3]);
                const float scale = 0.5f; // square root
                uint16_t logRelArea = uint16_t(65535.5f + core::clamp(scale * std::log2f(area / maxArea), -65535.5f, 0.f));
                key = core::morton4d_encode(tmp[0], tmp[1], tmp[2], logRelArea);
            }

            uint64_t key;
        };


        //TODO: use SoA instead (with core::radix_sort):
        //core::vector<Triangle> triangles;
        //core::vector<MortonTriangle> triangleMortonCodes;
        //where `triangles` is member of `TriangleBatch` struct
        struct TriangleMortonCodePair
        {
            Triangle triangle;
            MortonTriangle mortonCode;

            inline bool operator<(const TriangleMortonCodePair& other)
            {
                return this->mortonCode.key < other.mortonCode.key;
            }
        };

        TriangleBatches triangleBatches(triCnt);
        core::vector<TriangleMortonCodePair> triangles(triCnt); //#1

        //triangle reordering
        {
            //this is needed for mesh buffers with no index buffer (triangle strips and triagnle fans)
            //TODO: fix
            //bool wasTmpIdxBufferSet = false;
            //if (meshBuffer->getIndexBufferBinding().buffer == nullptr)
            //{
            //    //temporary use generated index buffer
            //    wasTmpIdxBufferSet = true;
            //    meshBuffer->setIndexBufferBinding(idxBufferParams.idxBuffer);
            //}

            const core::aabbox3df aabb = IMeshManipulator::calculateBoundingBox(meshBuffer);

            uint32_t ix = 0u;
            float maxTriangleArea = 0.0f;
            for (auto it = triangles.begin(); it != triangles.end(); it++)
            {
                auto triangleIndices = IMeshManipulator::getTriangleIndices(meshBuffer, ix++);
                //have to copy there
                std::copy(triangleIndices.begin(), triangleIndices.end(), it->triangle.oldIndices);

                core::vectorSIMDf trianglePos[3];
                trianglePos[0] = meshBuffer->getPosition(it->triangle.oldIndices[0]);
                trianglePos[1] = meshBuffer->getPosition(it->triangle.oldIndices[1]);
                trianglePos[2] = meshBuffer->getPosition(it->triangle.oldIndices[2]);

                const core::vectorSIMDf centroid = ((trianglePos[0] + trianglePos[1] + trianglePos[2]) / 3.0f) - core::vectorSIMDf(aabb.MinEdge.X, aabb.MinEdge.Y, aabb.MinEdge.Z);
                uint16_t fixedPointPos[3];
                fixedPointPos[0] = uint16_t(centroid.x * 65535.5f / aabb.getExtent().X);
                fixedPointPos[1] = uint16_t(centroid.y * 65535.5f / aabb.getExtent().Y);
                fixedPointPos[2] = uint16_t(centroid.z * 65535.5f / aabb.getExtent().Z);

                float area = core::cross(trianglePos[1] - trianglePos[0], trianglePos[2] - trianglePos[0]).x;
                it->mortonCode = MortonTriangle(fixedPointPos, area);

                if (area > maxTriangleArea)
                    maxTriangleArea = area;
            }

            /*if (wasTmpIdxBufferSet)
                meshBuffer->setIndexBufferBinding(nullptr);*/

                //complete morton code
            for (auto it = triangles.begin(); it != triangles.end(); it++)
                it->mortonCode.complete(maxTriangleArea);

            std::sort(triangles.begin(), triangles.end());
        }

        //copying, after radix_sort this will be removed
        //TODO durning radix_sort integration:
        //since there will be distinct arrays for triangles and their morton code use `triangleBatches.triangles` instead of #1
        for (uint32_t i = 0u; i < triCnt; i++)
            triangleBatches.triangles[i] = triangles[i].triangle;

        //set ranges
        Triangle* triangleArrayBegin = triangleBatches.triangles.data();
        Triangle* triangleArrayEnd = triangleArrayBegin + triangleBatches.triangles.size();
        const uint32_t triangleCnt = triangleBatches.triangles.size();

        //aabb batch division
        {
            triangleBatches.ranges.push_back(triangleArrayBegin);
            for (auto nextTriangle = triangleArrayBegin; nextTriangle < triangleArrayEnd; )
            {
                const Triangle* batchBegin = *(triangleBatches.ranges.end() - 1u);
                const Triangle* batchEnd = batchBegin + m_minTriangleCountPerMDIData;

                //find min and max edge
                core::vector3df_SIMD min(std::numeric_limits<float>::max());
                core::vector3df_SIMD max(-std::numeric_limits<float>::max());

                auto extendAABB = [&min, &max, &meshBuffer](auto triangleIt) -> void
                {
                    for (uint32_t i = 0u; i < 3u; i++)
                    {
                        auto vxPos = meshBuffer->getPosition(triangleIt->oldIndices[i]);
                        min = core::min(vxPos, min);
                        max = core::max(vxPos, max);
                    }
                };

                for (uint32_t i = 0u; i < m_minTriangleCountPerMDIData && nextTriangle != triangleArrayEnd; i++)
                    extendAABB(nextTriangle++);

                auto halfAreaAABB = [&min, &max]() -> float
                {
                    auto extent = max - min;
                    return extent.x * extent.y + extent.x * extent.z + extent.y * extent.z;
                };

                constexpr float kGrowthLimit = 1.025f;
                float batchArea = halfAreaAABB();
                for (uint16_t i = m_minTriangleCountPerMDIData; nextTriangle != triangleArrayEnd && i < m_maxTriangleCountPerMDIData; i++)
                {
                    // TODO: save the AABB of the MDI batch before it gets "try extended" (will be needed in the future for culling)
                    extendAABB(nextTriangle);
                    float newBatchArea = halfAreaAABB();
                    if (newBatchArea > kGrowthLimit* batchArea)
                        break;
                    nextTriangle++;
                    batchArea = newBatchArea;
                }

                triangleBatches.ranges.push_back(nextTriangle);
            }
        }

        return triangleBatches;
    }
};

}
}

#endif