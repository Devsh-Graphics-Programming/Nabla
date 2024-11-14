// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_MESH_PACKER_H_INCLUDED__
#define __NBL_ASSET_I_MESH_PACKER_H_INCLUDED__

#include "nbl/asset/utils/IMeshManipulator.h"
#include "nbl/core/math/morton.h"

namespace nbl
{
namespace asset
{

class IMeshPackerBase : public virtual core::IReferenceCounted
{
    public:
        constexpr static uint32_t MAX_TRIANGLES_IN_BATCH_CNT = 21845u;
        
        struct ReservedAllocationMeshBuffersBase
        {
            uint32_t mdiAllocationOffset;
            uint32_t mdiAllocationReservedCnt;
            uint32_t indexAllocationOffset;
            uint32_t indexAllocationReservedCnt;

            inline bool isValid()
            {
                return this->mdiAllocationOffset!=core::GeneralpurposeAddressAllocator<uint32_t>::invalid_address;
            }
        };
        struct PackedMeshBufferData
        {
            uint32_t mdiParameterOffset; // add to `CCPUMeshPacker::getMultiDrawIndirectBuffer()->getPointer() to get `DrawElementsIndirectCommand_t` address
            uint32_t mdiParameterCount;

            inline bool isValid()
            {
                return this->mdiParameterOffset != core::GeneralpurposeAddressAllocator<uint32_t>::invalid_address;
            }
        };

        inline uint16_t getMinTriangleCountPerMDI() const { return m_minTriangleCountPerMDIData; }
        inline uint16_t getMaxTriangleCountPerMDI() const { return m_maxTriangleCountPerMDIData; }

    protected:
        using alctrTraits = core::address_allocator_traits<core::GeneralpurposeAddressAllocator<uint32_t>>;

        IMeshPackerBase(uint16_t minTriangleCountPerMDIData, uint16_t maxTriangleCountPerMDIData)
            :m_maxTriangleCountPerMDIData(maxTriangleCountPerMDIData),
             m_minTriangleCountPerMDIData(minTriangleCountPerMDIData)
        {
            assert(minTriangleCountPerMDIData <= MAX_TRIANGLES_IN_BATCH_CNT);
            assert(maxTriangleCountPerMDIData <= MAX_TRIANGLES_IN_BATCH_CNT);
            assert(minTriangleCountPerMDIData <= maxTriangleCountPerMDIData);
            assert(minTriangleCountPerMDIData > 0u);
            assert(maxTriangleCountPerMDIData > 0u);
        };

        virtual ~IMeshPackerBase()
        {
            _NBL_ALIGNED_FREE(const_cast<void*>(alctrTraits::getReservedSpacePtr(m_MDIDataAlctr)));
            _NBL_ALIGNED_FREE(const_cast<void*>(alctrTraits::getReservedSpacePtr(m_idxBuffAlctr)));
            _NBL_ALIGNED_FREE(const_cast<void*>(alctrTraits::getReservedSpacePtr(m_vtxBuffAlctr)));
        }

        struct AllocationParamsCommon
        {
            // Maximum number of 16 bit indicies that may be allocated
            size_t indexBuffSupportedCnt = 67108864ull;                    /*   128MB*/

            /* Maximum byte size for vertex data allocation
               For `CCPUMeshPackerV1` this will be maximum byte size of buffer containing only attributes with EVIR_PER_VERTEX input rate.
               For `CCPUMeshPackerV2` this will be maximum byte size of buffer containing attributes with both EVIR_PER_VERTEX and EVIR_PER_INSTANCE input rate.
            */
            size_t vertexBuffSupportedByteSize = 134217728ull;                 /*   128MB*/

            // Maximum number of MDI structs that may be allocated
            size_t MDIDataBuffSupportedCnt = 16777216ull;                  /*   16MB assuming MDIStructType is DrawElementsIndirectCommand_t*/

            // Minimum count of 16 bit indicies allocated per allocation
            size_t indexBufferMinAllocCnt = 256ull;

            // Minimum bytes of vertex data allocated per allocation
            size_t vertexBufferMinAllocByteSize = 32ull;

            // Minimum count of MDI structs allocated per allocation
            size_t MDIDataBuffMinAllocCnt = 32ull;
        };

        void initializeCommonAllocators(const AllocationParamsCommon& allocParams)
        {
            if (allocParams.indexBuffSupportedCnt)
            {
                
                void* resSpcTmp = _NBL_ALIGNED_MALLOC(core::GeneralpurposeAddressAllocator<uint32_t>::reserved_size(alignof(uint16_t), allocParams.indexBuffSupportedCnt, allocParams.indexBufferMinAllocCnt), _NBL_SIMD_ALIGNMENT);
                assert(resSpcTmp != nullptr);
                m_idxBuffAlctr = core::GeneralpurposeAddressAllocator<uint32_t>(resSpcTmp, 0u, 0u, alignof(uint16_t), allocParams.indexBuffSupportedCnt, allocParams.indexBufferMinAllocCnt);
            }

            if (allocParams.vertexBuffSupportedByteSize)
            {
                void* resSpcTmp = _NBL_ALIGNED_MALLOC(core::GeneralpurposeAddressAllocator<uint32_t>::reserved_size(32u, allocParams.vertexBuffSupportedByteSize, allocParams.vertexBufferMinAllocByteSize), _NBL_SIMD_ALIGNMENT);
                assert(resSpcTmp != nullptr);
                m_vtxBuffAlctr = core::GeneralpurposeAddressAllocator<uint32_t>(resSpcTmp, 0u, 0u, 32u, allocParams.vertexBuffSupportedByteSize, allocParams.vertexBufferMinAllocByteSize);
            }

            if (allocParams.MDIDataBuffSupportedCnt)
            {
                void* resSpcTmp = _NBL_ALIGNED_MALLOC(core::GeneralpurposeAddressAllocator<uint32_t>::reserved_size(alignof(std::max_align_t), allocParams.MDIDataBuffSupportedCnt, allocParams.MDIDataBuffMinAllocCnt), _NBL_SIMD_ALIGNMENT);
                assert(resSpcTmp != nullptr);
                m_MDIDataAlctr = core::GeneralpurposeAddressAllocator<uint32_t>(resSpcTmp, 0u, 0u, alignof(std::max_align_t), allocParams.MDIDataBuffSupportedCnt, allocParams.MDIDataBuffMinAllocCnt);
            }
        }

        void initializeCommonAllocators(
            const core::GeneralpurposeAddressAllocator<uint32_t>& mdiAlctr,
            const core::GeneralpurposeAddressAllocator<uint32_t>& idxAlctr,
            const core::GeneralpurposeAddressAllocator<uint32_t>& vtxAlctr
            )
        {
            uint32_t alctrBuffSz = alctrTraits::get_total_size(mdiAlctr);
            void* resSpcTmp = _NBL_ALIGNED_MALLOC(alctrTraits::reserved_size(alctrBuffSz, mdiAlctr), _NBL_SIMD_ALIGNMENT);
            m_MDIDataAlctr = core::GeneralpurposeAddressAllocator<uint32_t>(alctrBuffSz, mdiAlctr, resSpcTmp);

            alctrBuffSz = alctrTraits::get_total_size(idxAlctr);
            resSpcTmp = _NBL_ALIGNED_MALLOC(alctrTraits::reserved_size(alctrBuffSz, idxAlctr), _NBL_SIMD_ALIGNMENT);
            m_idxBuffAlctr = core::GeneralpurposeAddressAllocator<uint32_t>(alctrBuffSz, idxAlctr, resSpcTmp);

            alctrBuffSz = alctrTraits::get_total_size(vtxAlctr);
            resSpcTmp = _NBL_ALIGNED_MALLOC(alctrTraits::reserved_size(alctrBuffSz, vtxAlctr), _NBL_SIMD_ALIGNMENT);
            m_vtxBuffAlctr = core::GeneralpurposeAddressAllocator<uint32_t>(alctrBuffSz, vtxAlctr, resSpcTmp);
        }

        void free(const ReservedAllocationMeshBuffersBase& rambb)
        {
            if (rambb.indexAllocationOffset != INVALID_ADDRESS)
                m_idxBuffAlctr.free_addr(rambb.indexAllocationOffset,rambb.indexAllocationReservedCnt);

            if (rambb.mdiAllocationOffset != INVALID_ADDRESS)
                m_MDIDataAlctr.free_addr(rambb.mdiAllocationOffset,rambb.mdiAllocationReservedCnt);
        }
        
        //
        _NBL_STATIC_INLINE_CONSTEXPR uint32_t INVALID_ADDRESS = core::GeneralpurposeAddressAllocator<uint32_t>::invalid_address;

        core::GeneralpurposeAddressAllocator<uint32_t> m_vtxBuffAlctr;
        core::GeneralpurposeAddressAllocator<uint32_t> m_idxBuffAlctr;
        core::GeneralpurposeAddressAllocator<uint32_t> m_MDIDataAlctr;

        const uint16_t m_minTriangleCountPerMDIData;
        const uint16_t m_maxTriangleCountPerMDIData;

};

#if 0 // REWRITE
template <typename MeshBufferType, typename MDIStructType = DrawElementsIndirectCommand_t>
class IMeshPacker : public IMeshPackerBase
{
    static_assert(std::is_base_of<DrawElementsIndirectCommand_t, MDIStructType>::value);

public:
    /*
    @param minTriangleCountPerMDIData must be <= 21845
    @param maxTriangleCountPerMDIData must be <= 21845
    */
    IMeshPacker(uint16_t minTriangleCountPerMDIData, uint16_t maxTriangleCountPerMDIData)
        :IMeshPackerBase(minTriangleCountPerMDIData, maxTriangleCountPerMDIData)
    {
    }

    //! shrinks byte size of all output buffers, so they are large enough to fit currently allocated contents. Call this function before `instantiateDataStorage`
    virtual void shrinkOutputBuffersSize()
    {
        uint32_t mdiDataBuffNewSize = m_MDIDataAlctr.safe_shrink_size(0u, alctrTraits::max_alignment(m_MDIDataAlctr));
        uint32_t idxBuffNewSize = m_idxBuffAlctr.safe_shrink_size(0u, alctrTraits::max_alignment(m_idxBuffAlctr));
        uint32_t vtxBuffNewSize = m_vtxBuffAlctr.safe_shrink_size(0u, alctrTraits::max_alignment(m_vtxBuffAlctr));

        const void* oldReserved = alctrTraits::getReservedSpacePtr(m_MDIDataAlctr);
        m_MDIDataAlctr = core::GeneralpurposeAddressAllocator(mdiDataBuffNewSize, std::move(m_MDIDataAlctr), _NBL_ALIGNED_MALLOC(alctrTraits::reserved_size(mdiDataBuffNewSize, m_MDIDataAlctr), _NBL_SIMD_ALIGNMENT));
        _NBL_ALIGNED_FREE(const_cast<void*>(oldReserved));

        oldReserved = alctrTraits::getReservedSpacePtr(m_idxBuffAlctr);
        m_idxBuffAlctr = core::GeneralpurposeAddressAllocator(idxBuffNewSize, std::move(m_idxBuffAlctr), _NBL_ALIGNED_MALLOC(alctrTraits::reserved_size(idxBuffNewSize, m_idxBuffAlctr), _NBL_SIMD_ALIGNMENT));
        _NBL_ALIGNED_FREE(const_cast<void*>(oldReserved));

        oldReserved = alctrTraits::getReservedSpacePtr(m_vtxBuffAlctr);
        m_vtxBuffAlctr = core::GeneralpurposeAddressAllocator(vtxBuffNewSize, std::move(m_vtxBuffAlctr), _NBL_ALIGNED_MALLOC(alctrTraits::reserved_size(vtxBuffNewSize, m_vtxBuffAlctr), _NBL_SIMD_ALIGNMENT));
        _NBL_ALIGNED_FREE(const_cast<void*>(oldReserved));
    }

    //! Returns maximum number of mdi structs needed to draw range of mesh buffers described by range mbBegin .. mbEnd, actual number of MDI structs needed may differ
    template <typename MeshBufferIterator>
    uint32_t calcMDIStructMaxCount(const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd)
    {
        uint32_t acc = 0u;
        for (auto mbIt = mbBegin; mbIt != mbEnd; mbIt++)
        {
            auto mb = *mbIt;
            const size_t idxCnt = calcIdxCntAfterConversionToTriangleList(mb);
            const uint32_t triCnt = idxCnt / 3;
            assert(idxCnt % 3 == 0);

            acc += calcBatchCountBound(triCnt);
        }

        return acc;
    }

protected:
    virtual ~IMeshPacker() {}

    static inline size_t calcVertexSize(const SVertexInputParams& vtxInputParams, const E_VERTEX_INPUT_RATE inputRate)
    {
        size_t size = 0ull;
        for (size_t i = 0; i < SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT; ++i)
        {
            if (vtxInputParams.enabledAttribFlags & (1u << i))
                if(vtxInputParams.bindings[i].inputRate == inputRate)
                    size += asset::getTexelOrBlockBytesize(static_cast<E_FORMAT>(vtxInputParams.attributes[i].format));
        }

        return size;
    }

    static inline uint32_t calcVertexCountBoundWithBatchDuplication(const MeshBufferType* meshBuffer)
    {
        uint32_t triCnt;
        if (IMeshManipulator::getPolyCount(triCnt,meshBuffer))
            return triCnt * 3u;
        return 0u;
    }

    inline uint32_t calcBatchCountBound(uint32_t triCnt) const
    {
        if (triCnt!=0u)
            return (triCnt-1u)/m_minTriangleCountPerMDIData+1u;
        return 0u;
    }

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
        SBufferBinding<ICPUBuffer> idxBuffer = { 0u, nullptr };
        E_INDEX_TYPE idxType = EIT_UNKNOWN;
    };

    //TODO: functions: constructTriangleBatches, convertIdxBufferToTriangles, deinterleaveAndCopyAttribute and deinterleaveAndCopyPerInstanceAttribute
    //will not work with IGPUMeshBuffer as MeshBufferType, move it to new `ICPUMeshPacker`

    TriangleBatches constructTriangleBatches(const MeshBufferType* meshBuffer, IdxBufferParams idxBufferParams, core::aabbox3df*& aabbs) const
    {
        uint32_t triCnt;
        const bool success = IMeshManipulator::getPolyCount(triCnt,meshBuffer);
        assert(success);
         
        const uint32_t batchCnt = calcBatchCountBound(triCnt);
        assert(batchCnt != 0u);

        struct MortonTriangle
        {
            MortonTriangle() = default;

            MortonTriangle(uint16_t fixedPointPos[3], float area)
            {
                auto tmp = reinterpret_cast<uint16_t*>(key);
                std::copy_n(fixedPointPos,3u,tmp);
                tmp[3] = core::Float16Compressor::compress(area);
            }

            void complete(float maxArea)
            {
                auto tmp = reinterpret_cast<const uint16_t*>(key);
                const float area = core::Float16Compressor::decompress(tmp[3]);
                const float scale = 0.5f; // square root
                uint16_t logRelArea = uint16_t(65535.5f+core::clamp(scale*std::log2f(area/maxArea),-65535.5f,0.f));
                key = core::morton4d_encode(tmp[0],tmp[1],tmp[2],logRelArea);
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

        core::smart_refctd_ptr<ICPUMeshBuffer> mbTmp = core::smart_refctd_ptr_static_cast<ICPUMeshBuffer>(meshBuffer->clone());
        mbTmp->setIndexBufferBinding(std::move(idxBufferParams.idxBuffer));
        mbTmp->setIndexType(idxBufferParams.idxType);
        mbTmp->getPipeline()->getPrimitiveAssemblyParams().primitiveType = EPT_TRIANGLE_LIST;

        //triangle reordering
        {
            const core::aabbox3df aabb = IMeshManipulator::calculateBoundingBox(mbTmp.get());

            uint32_t ix = 0u;
            float maxTriangleArea = 0.0f;
            for (auto it = triangles.begin(); it != triangles.end(); it++)
            {
                auto triangleIndices = IMeshManipulator::getTriangleIndices(mbTmp.get(), ix++);
                //have to copy there
                std::copy(triangleIndices.begin(), triangleIndices.end(), it->triangle.oldIndices);

                core::vectorSIMDf trianglePos[3];
                trianglePos[0] = mbTmp->getPosition(it->triangle.oldIndices[0]);
                trianglePos[1] = mbTmp->getPosition(it->triangle.oldIndices[1]);
                trianglePos[2] = mbTmp->getPosition(it->triangle.oldIndices[2]);

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
                    if(aabbs)
                        *aabbs = core::aabbox3df(core::vector3df(min.x, min.y, min.z), core::vector3df(max.x, max.y, max.z));

                    extendAABB(nextTriangle);
                    float newBatchArea = halfAreaAABB();
                    if (newBatchArea > kGrowthLimit* batchArea)
                        break;
                    nextTriangle++;
                    batchArea = newBatchArea;
                }

                if (aabbs)
                {
                    if (nextTriangle == triangleArrayEnd || m_minTriangleCountPerMDIData == m_maxTriangleCountPerMDIData)
                        *aabbs = core::aabbox3df(core::vector3df(min.x, min.y, min.z), core::vector3df(max.x, max.y, max.z));
                    aabbs++;
                }

                triangleBatches.ranges.push_back(nextTriangle);
            }
                
        }

        return triangleBatches;
    }

    static core::unordered_map<uint32_t, uint16_t> constructNewIndicesFromTriangleBatchAndUpdateUnifiedIndexBuffer(TriangleBatches& batches, uint32_t batchIdx, uint16_t*& indexBuffPtr)
    {
        core::unordered_map<uint32_t, uint16_t> usedVertices;
        core::vector<Triangle> newIdxTris = batches.triangles;

        auto batchBegin = batches.ranges[batchIdx];
        auto batchEnd = batches.ranges[batchIdx + 1];

        const uint32_t triangleInBatchCnt = std::distance(batchBegin, batchEnd);
        const uint32_t idxInBatchCnt = 3u * triangleInBatchCnt;

        uint32_t newIdx = 0u;
        for (uint32_t i = 0u; i < triangleInBatchCnt; i++)
        {
            const Triangle* const triangle = batchBegin + i;
            for (int32_t j = 0; j < 3; j++)
            {
                const uint32_t oldIndex = triangle->oldIndices[j];
                auto result = usedVertices.insert(std::make_pair(oldIndex, newIdx));

                newIdxTris[i].oldIndices[j] = result.second ? newIdx++ : result.first->second;
            }
        }

        //TODO: cache optimization
        //copy indices into unified index buffer
        for (size_t i = 0; i < triangleInBatchCnt; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                *indexBuffPtr = newIdxTris[i].oldIndices[j];
                indexBuffPtr++;
            }
        }

        return usedVertices;
    }

    static void deinterleaveAndCopyAttribute(MeshBufferType* meshBuffer, uint16_t attrLocation, const core::unordered_map<uint32_t, uint16_t>& usedVertices, uint8_t* dstAttrPtr)
    {
        const uint8_t* const srcAttrPtr = meshBuffer->getAttribPointer(attrLocation);
        SVertexInputParams& mbVtxInputParams = meshBuffer->getPipeline()->getVertexInputParams();
        SVertexInputAttribParams MBAttrib = mbVtxInputParams.attributes[attrLocation];
        SVertexInputBindingParams attribBinding = mbVtxInputParams.bindings[MBAttrib.binding];
        const size_t attrSize = asset::getTexelOrBlockBytesize(static_cast<E_FORMAT>(MBAttrib.format));
        const size_t stride = (attribBinding.stride) == 0 ? attrSize : attribBinding.stride;

        for (auto index : usedVertices)
        {
            const uint8_t* attrSrc = srcAttrPtr + (index.first * stride);
            uint8_t* attrDest = dstAttrPtr + (index.second * attrSize);
            memcpy(attrDest, attrSrc, attrSize);
        }
    }

    static void deinterleaveAndCopyPerInstanceAttribute(MeshBufferType* meshBuffer, uint16_t attrLocation, uint8_t* dstAttrPtr)
    {
        const uint8_t* const srcAttrPtr = meshBuffer->getAttribPointer(attrLocation);
        SVertexInputParams& mbVtxInputParams = meshBuffer->getPipeline()->getVertexInputParams();
        SVertexInputAttribParams MBAttrib = mbVtxInputParams.attributes[attrLocation];
        SVertexInputBindingParams attribBinding = mbVtxInputParams.bindings[MBAttrib.binding];
        const size_t attrSize = asset::getTexelOrBlockBytesize(static_cast<E_FORMAT>(MBAttrib.format));
        const size_t stride = (attribBinding.stride) == 0 ? attrSize : attribBinding.stride;

        const uint32_t insCnt = meshBuffer->getInstanceCount();
        for (uint32_t i = 0u; i < insCnt; i++)
        {
            const uint8_t* attrSrc = srcAttrPtr + (i * stride);
            uint8_t* attrDest = dstAttrPtr + (i * attrSize);
            memcpy(attrDest, attrSrc, attrSize);
        }
    }

    inline uint32_t calcIdxCntAfterConversionToTriangleList(const MeshBufferType* meshBuffer)
    {
        const auto& params = meshBuffer->getPipeline()->getPrimitiveAssemblyParams();

        switch (params.primitiveType)
        {
            case EPT_TRIANGLE_LIST: 
            case EPT_TRIANGLE_STRIP:
            case EPT_TRIANGLE_FAN:
                break;
            case EPT_POINT_LIST:
            case EPT_LINE_LIST:
            case EPT_LINE_STRIP:
            case EPT_LINE_LIST_WITH_ADJACENCY:
            case EPT_LINE_STRIP_WITH_ADJACENCY:
            case EPT_TRIANGLE_LIST_WITH_ADJACENCY:
            case EPT_TRIANGLE_STRIP_WITH_ADJACENCY:
            case EPT_PATCH_LIST:
            default:
                assert(false);
                break;
        }

        uint32_t triCnt;
        const bool success = IMeshManipulator::getPolyCount(triCnt, meshBuffer);
        assert(success);

        return triCnt * 3;
    }
    inline uint32_t calcIdxCntAfterConversionToTriangleList(const core::smart_refctd_ptr<MeshBufferType>& meshBuffer)
    {
        return calcIdxCntAfterConversionToTriangleList(meshBuffer.get());
    }
    inline uint32_t calcIdxCntAfterConversionToTriangleList(const core::smart_refctd_ptr<const MeshBufferType>& meshBuffer)
    {
        return calcIdxCntAfterConversionToTriangleList(meshBuffer.get());
    }

    std::pair<uint32_t, core::smart_refctd_ptr<ICPUBuffer>> convertIdxBufferToTriangles(MeshBufferType* meshBuffer)
    {
        const auto mbIdxBuffer = meshBuffer->getIndexBufferBinding().buffer;
        E_INDEX_TYPE idxType = meshBuffer->getIndexType();
        const uint32_t idxCount = meshBuffer->getIndexCount();
        if (idxCount == 0)
            return { 0u, nullptr };

        const bool iota = idxType == EIT_UNKNOWN || !mbIdxBuffer;
        core::smart_refctd_ptr<ICPUBuffer> idxBufferToProcess;
        if (iota)
        {
            idxBufferToProcess = core::make_smart_refctd_ptr<ICPUBuffer>(sizeof(uint32_t) * idxCount);
            auto ptr = reinterpret_cast<uint32_t*>(idxBufferToProcess->getPointer());
            std::iota(ptr, ptr + idxCount, 0u);
            idxType = EIT_32BIT;
        }
        else
        {
            idxBufferToProcess = mbIdxBuffer;
        }
        
        std::pair<uint32_t, core::smart_refctd_ptr<ICPUBuffer>> output;
        output.first = meshBuffer->getIndexCount();

        const auto& params = meshBuffer->getPipeline()->getPrimitiveAssemblyParams();
        switch (params.primitiveType)
        {
            case EPT_TRIANGLE_STRIP:
                output.second = IMeshManipulator::idxBufferFromTriangleStripsToTriangles(idxBufferToProcess->getPointer(), output.first, idxType, idxType);
                return output;

            case EPT_TRIANGLE_FAN:
                output.second = IMeshManipulator::idxBufferFromTrianglesFanToTriangles(idxBufferToProcess->getPointer(), output.first, idxType, idxType);
                return output;

                //TODO: packer should return when there is mesh buffer with one of following:
            case EPT_TRIANGLE_LIST:
            case EPT_POINT_LIST:
            case EPT_LINE_LIST:
            case EPT_LINE_STRIP:
            case EPT_LINE_LIST_WITH_ADJACENCY:
            case EPT_LINE_STRIP_WITH_ADJACENCY:
            case EPT_TRIANGLE_LIST_WITH_ADJACENCY:
            case EPT_TRIANGLE_STRIP_WITH_ADJACENCY:
            case EPT_PATCH_LIST:
            default:
                assert(false);
                return { 0u, nullptr };
        }
    }

    IdxBufferParams createNewIdxBufferParamsForNonTriangleListTopologies(MeshBufferType* meshBuffer)
    {
        IdxBufferParams output;

        const auto& mbPrimitiveType = meshBuffer->getPipeline()->getPrimitiveAssemblyParams().primitiveType;
        if (mbPrimitiveType == EPT_TRIANGLE_LIST)
        {
            const auto& mbIdxBuff = meshBuffer->getIndexBufferBinding();
            output.idxBuffer.offset = mbIdxBuff.offset;
            output.idxBuffer.buffer = core::smart_refctd_ptr(mbIdxBuff.buffer);
            output.idxType = meshBuffer->getIndexType();
        }
        else
        {
            auto newIdxBuffer = convertIdxBufferToTriangles(meshBuffer);
            output.idxBuffer.offset = 0u;
            output.idxBuffer.buffer = newIdxBuffer.second;
            output.idxType = EIT_32BIT;
        }

        return output;
    }

protected:
    template <typename BufferType>
    struct PackerDataStoreCommon
    {
        static_assert(std::is_base_of<core::IBuffer, BufferType>::value);

        core::smart_refctd_ptr<BufferType> MDIDataBuffer;

        inline bool isValid()
        {
            return this->MDIDataBuffer->getPointer() != nullptr;
        }
    };

};
#endif
}
}

#endif