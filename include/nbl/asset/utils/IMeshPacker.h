// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_MESH_PACKER_H_INCLUDED__
#define __NBL_ASSET_I_MESH_PACKER_H_INCLUDED__

#include "nbl/asset/utils/IMeshManipulator.h"

namespace nbl
{
namespace asset
{

class IMeshPackerBase
{
public:
    struct PackedMeshBufferData
    {
        uint32_t mdiParameterOffset; // add to `CCPUMeshPacker::getMultiDrawIndirectBuffer()->getPointer() to get `DrawElementsIndirectCommand_t` address
        uint32_t mdiParameterCount;

        inline bool isValid()
        {
            return this->mdiParameterOffset != core::GeneralpurposeAddressAllocator<uint32_t>::invalid_address;
        }
    };

protected:
    IMeshPackerBase(uint16_t minTriangleCountPerMDIData, uint16_t maxTriangleCountPerMDIData)
        :m_maxTriangleCountPerMDIData(maxTriangleCountPerMDIData),
         m_minTriangleCountPerMDIData(minTriangleCountPerMDIData),
         m_MDIDataAlctrResSpc(nullptr),
         m_idxBuffAlctrResSpc(nullptr),
         m_vtxBuffAlctrResSpc(nullptr)
    {
        assert(minTriangleCountPerMDIData <= 21845);
        assert(maxTriangleCountPerMDIData <= 21845);
    };

    virtual ~IMeshPackerBase()
    {
        _NBL_ALIGNED_FREE(m_MDIDataAlctrResSpc);
        _NBL_ALIGNED_FREE(m_idxBuffAlctrResSpc);
        _NBL_ALIGNED_FREE(m_vtxBuffAlctrResSpc);
    }

    struct AllocationParamsCommon
    {
        // TODO: review all names and documnetation!
        
        // Maximum number of 16 bit indicies that may be allocated
        size_t indexBuffSupportedCnt = 67108864ull;                    /*   128MB*/

        /* Maximum byte size for vertex data allocation
           For `CCPUMeshPackerV1` this will be maximum byte size of buffer containing only attributes with EVIR_PER_VERTEX input rate.
           For `CCPUMeshPackerV2` this will be maximum byte size of buffer containing attributes with both EVIR_PER_VERTEX and EVIR_PER_INSTANCE input rate.
        */
        size_t vertexBuffSupportedSize = 134217728ull;                 /*   128MB*/

        // Maximum number of MDI structs that may be allocated
        size_t MDIDataBuffSupportedCnt = 16777216ull;                  /*   16MB assuming MDIStructType is DrawElementsIndirectCommand_t*/

        // Minimum count of 16 bit indicies allocated per allocation
        size_t indexBufferMinAllocCnt = 256ull;

        // Minimum bytes of vertex data allocated per allocation
        size_t vertexBufferMinAllocSize = 32ull;

        // Minimum count of MDI structs allocated per allocation
        size_t MDIDataBuffMinAllocCnt = 32ull;
    };

    void initializeCommonAllocators(const AllocationParamsCommon& allocParams)
    {
        if (allocParams.indexBuffSupportedCnt)
        {
            m_idxBuffAlctrResSpc = _NBL_ALIGNED_MALLOC(core::GeneralpurposeAddressAllocator<uint32_t>::reserved_size(alignof(uint16_t), allocParams.indexBuffSupportedCnt, allocParams.indexBufferMinAllocCnt), _NBL_SIMD_ALIGNMENT);
            _NBL_DEBUG_BREAK_IF(m_idxBuffAlctrResSpc == nullptr);
            assert(m_idxBuffAlctrResSpc != nullptr);
            m_idxBuffAlctr = core::GeneralpurposeAddressAllocator<uint32_t>(m_idxBuffAlctrResSpc, 0u, 0u, alignof(uint16_t), allocParams.indexBuffSupportedCnt, allocParams.indexBufferMinAllocCnt);
        }

        if (allocParams.vertexBuffSupportedSize)
        {
            m_vtxBuffAlctrResSpc = _NBL_ALIGNED_MALLOC(core::GeneralpurposeAddressAllocator<uint32_t>::reserved_size(32u, allocParams.vertexBuffSupportedSize, allocParams.vertexBufferMinAllocSize), _NBL_SIMD_ALIGNMENT);
            _NBL_DEBUG_BREAK_IF(m_vtxBuffAlctrResSpc == nullptr);
            assert(m_vtxBuffAlctrResSpc != nullptr);
            m_vtxBuffAlctr = core::GeneralpurposeAddressAllocator<uint32_t>(m_vtxBuffAlctrResSpc, 0u, 0u, 32u, allocParams.vertexBuffSupportedSize, allocParams.vertexBufferMinAllocSize);
        }

        if (allocParams.MDIDataBuffSupportedCnt)
        {
            m_MDIDataAlctrResSpc = _NBL_ALIGNED_MALLOC(core::GeneralpurposeAddressAllocator<uint32_t>::reserved_size(alignof(std::max_align_t), allocParams.MDIDataBuffSupportedCnt, allocParams.MDIDataBuffMinAllocCnt), _NBL_SIMD_ALIGNMENT);
            _NBL_DEBUG_BREAK_IF(m_MDIDataAlctrResSpc == nullptr);
            assert(m_MDIDataAlctrResSpc != nullptr);
            m_MDIDataAlctr = core::GeneralpurposeAddressAllocator<uint32_t>(m_MDIDataAlctrResSpc, 0u, 0u, alignof(std::max_align_t), allocParams.MDIDataBuffSupportedCnt, allocParams.MDIDataBuffMinAllocCnt);
        }
    }

protected:
    void* m_MDIDataAlctrResSpc;
    void* m_idxBuffAlctrResSpc;
    void* m_vtxBuffAlctrResSpc;
    core::GeneralpurposeAddressAllocator<uint32_t> m_vtxBuffAlctr;
    core::GeneralpurposeAddressAllocator<uint32_t> m_idxBuffAlctr;
    core::GeneralpurposeAddressAllocator<uint32_t> m_MDIDataAlctr;

    const uint16_t m_minTriangleCountPerMDIData;
    const uint16_t m_maxTriangleCountPerMDIData;

};

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
            return triCnt*3u;
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

    struct TriangleBatch
    {
        core::vector<Triangle> triangles;
    };

    struct IdxBufferParams
    {
        SBufferBinding<ICPUBuffer> idxBuffer;
        uint32_t idxCnt = 0u; //TODO: if you will be sure that it will not be used anywhere, delete this and modify `convertIdxBufferToTriangles` to return only idx buffer
        E_INDEX_TYPE indexType = EIT_UNKNOWN;
    };

    //TODO: functions: constructTriangleBatches, convertIdxBufferToTriangles, deinterleaveAndCopyAttribute and deinterleaveAndCopyPerInstanceAttribute will not work with IGPUMeshBuffer as MeshBufferType, move it to new `ICPUMeshPacker`

    core::vector<TriangleBatch> constructTriangleBatches(const MeshBufferType* meshBuffer, IdxBufferParams idxBufferParams) const
    {
        uint32_t triCnt;
        const bool success = IMeshManipulator::getPolyCount(triCnt,meshBuffer);
        assert(success);

        const uint32_t batchCount = calcBatchCountBound(triCnt);
        /*
        TODO:

        //struct TriangleMortonCodePair
        //{
        //	Triangle triangle;
        //	//uint64_t mortonCode; TODO after benchmarks
        //};
        
        core::vector<TriangleMortonCodePair> triangles(triCnt);
        uint32_t ix=0u;
        for (auto it=triangles.begin(); it!=triangles.end(); it++)
        {
            *it = IMeshManipulator::getTriangleIndices(meshbuffer,ix++);
        }

        std::sort(triangles.begin(),triangles.end(),[]()->bool{return lhs.mortonCode<rhs.mortonCode;}); // maybe use our new core::radix_sort?

        // do batch splitting
        core::vector<const TriangleMortonCodePair*> batches;
        batches.reserve(calcBatchCountBound(triCnt)+1u); // actual batch count will be different
        {
            // use batches.push_back();
        }
        batches.push_back(triangles.data()+triangles.size());

        return {std::move(triangles),std::move(batches)};
        */
        core::vector<TriangleBatch> output(batchCount); // nested vectors are evil

        for (uint32_t i = 0u; i < batchCount; i++)
        {
            if (i == (batchCount - 1))
            {
                const auto lastBatchLen = triCnt % uint32_t(m_maxTriangleCountPerMDIData);
                if (lastBatchLen)
                {
                    output[i].triangles = core::vector<Triangle>(lastBatchLen);
                    continue;
                }
            }

            output[i].triangles = core::vector<Triangle>(m_maxTriangleCountPerMDIData);
        }

        //TODO: triangle reordering

        auto idxBufferPtr32Bit = static_cast<const uint32_t*>(idxBufferParams.idxBuffer.buffer->getPointer()) + (idxBufferParams.idxBuffer.offset / sizeof(uint32_t)); //will be changed after benchmarks
        auto idxBufferPtr16Bit = static_cast<const uint16_t*>(idxBufferParams.idxBuffer.buffer->getPointer()) + (idxBufferParams.idxBuffer.offset / sizeof(uint16_t));
        for (TriangleBatch& batch : output)
        {
            for (Triangle& tri : batch.triangles)
            {
                if (idxBufferParams.indexType == EIT_32BIT)
                {
                    tri.oldIndices[0] = *idxBufferPtr32Bit;
                    tri.oldIndices[1] = *(++idxBufferPtr32Bit);
                    tri.oldIndices[2] = *(++idxBufferPtr32Bit);
                    idxBufferPtr32Bit++;
                }
                else if (idxBufferParams.indexType == EIT_16BIT)
                {

                    tri.oldIndices[0] = *idxBufferPtr16Bit;
                    tri.oldIndices[1] = *(++idxBufferPtr16Bit);
                    tri.oldIndices[2] = *(++idxBufferPtr16Bit);
                    idxBufferPtr16Bit++;
                }
            }
        }

        return output;
    }

    static core::unordered_map<uint32_t, uint16_t> constructNewIndicesFromTriangleBatch(TriangleBatch& batch, uint16_t*& indexBuffPtr)
    {
        core::unordered_map<uint32_t, uint16_t> usedVertices;
        core::vector<Triangle> newIdxTris = batch.triangles;

        uint32_t newIdx = 0u;
        for (uint32_t i = 0u; i < batch.triangles.size(); i++)
        {
            const Triangle& triangle = batch.triangles[i];
            for (int32_t j = 0; j < 3; j++)
            {
                const uint32_t oldIndex = triangle.oldIndices[j];
                auto result = usedVertices.insert(std::make_pair(oldIndex, newIdx));

                newIdxTris[i].oldIndices[j] = result.second ? newIdx++ : result.first->second;
            }
        }

        //TODO: cache optimization

        //copy indices into unified index buffer
        for (size_t i = 0; i < batch.triangles.size(); i++)
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
        const uint8_t const* srcAttrPtr = meshBuffer->getAttribPointer(attrLocation);
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

    //is it too much of DRY violation?
    static void deinterleaveAndCopyPerInstanceAttribute(MeshBufferType* meshBuffer, uint16_t attrLocation, uint8_t* dstAttrPtr)
    {
        const uint8_t const* srcAttrPtr = meshBuffer->getAttribPointer(attrLocation);
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

    uint32_t calcIdxCntAfterConversionToTriangleList(MeshBufferType* meshBuffer)
    {
        const auto& params = meshBuffer->getPipeline()->getPrimitiveAssemblyParams();
        uint32_t idxCnt = meshBuffer->getIndexCount();

        switch (params.primitiveType)
        {
            case EPT_TRIANGLE_LIST: 
                return idxCnt;
            case EPT_TRIANGLE_STRIP:
            case EPT_TRIANGLE_FAN:
                return (idxCnt - 2) * 3;
                //TODO: packer should return when there is mesh buffer with one of following:
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
                return 0u;
        }
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

    IdxBufferParams retriveOrCreateNewIdxBufferParams(MeshBufferType* meshBuffer)
    {
        IdxBufferParams output;

        const auto mbPrimitiveType = meshBuffer->getPipeline()->getPrimitiveAssemblyParams().primitiveType;
        if (mbPrimitiveType == EPT_TRIANGLE_LIST)
        {
            output.idxCnt = meshBuffer->getIndexCount();
            output.idxBuffer = meshBuffer->getIndexBufferBinding();
            output.indexType = meshBuffer->getIndexType();
        }
        else
        {
            auto newIdxBuffer = convertIdxBufferToTriangles(meshBuffer);
            output.idxCnt = newIdxBuffer.first;
            output.idxBuffer.offset = 0u;
            output.idxBuffer.buffer = newIdxBuffer.second;
            output.indexType = EIT_32BIT;
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

    _NBL_STATIC_INLINE_CONSTEXPR uint32_t INVALID_ADDRESS = core::GeneralpurposeAddressAllocator<uint32_t>::invalid_address;

};

}
}

#endif