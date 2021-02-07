// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_MESH_PACKER_H_INCLUDED__
#define __NBL_ASSET_I_MESH_PACKER_H_INCLUDED__

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
        size_t indexBuffSupportedCnt = 1073741824ull;                  /*   2GB*/
        size_t vertexBuffSupportedSize = 1ull << 31ull;                /*   2GB*/
        size_t MDIDataBuffSupportedCnt = 16777216ull;                  /*   16MB assuming MDIStructType is DrawElementsIndirectCommand_t*/
        size_t indexBufferMinAllocSize = 256ull;
        size_t vertexBufferMinAllocSize = 32ull;
        size_t MDIDataBuffMinAllocSize = 32ull;
    };

    void initializeCommonAllocators(const AllocationParamsCommon& allocParams)
    {
        if (allocParams.indexBuffSupportedCnt)
        {
            m_idxBuffAlctrResSpc = _NBL_ALIGNED_MALLOC(core::GeneralpurposeAddressAllocator<uint32_t>::reserved_size(alignof(uint16_t), allocParams.indexBuffSupportedCnt, allocParams.indexBufferMinAllocSize), _NBL_SIMD_ALIGNMENT);
            _NBL_DEBUG_BREAK_IF(m_idxBuffAlctrResSpc == nullptr);
            assert(m_idxBuffAlctrResSpc != nullptr);
            m_idxBuffAlctr = core::GeneralpurposeAddressAllocator<uint32_t>(m_idxBuffAlctrResSpc, 0u, 0u, alignof(uint16_t), allocParams.indexBuffSupportedCnt, allocParams.indexBufferMinAllocSize);
        }

        if (allocParams.vertexBuffSupportedSize)
        {
            m_vtxBuffAlctrResSpc = _NBL_ALIGNED_MALLOC(core::GeneralpurposeAddressAllocator<uint32_t>::reserved_size(32u, allocParams.vertexBuffSupportedSize, allocParams.vertexBufferMinAllocSize), _NBL_SIMD_ALIGNMENT);
            //for now mesh packer will not allow mesh buffers without any per vertex attributes
            _NBL_DEBUG_BREAK_IF(m_vtxBuffAlctrResSpc == nullptr);
            assert(m_vtxBuffAlctrResSpc != nullptr);
            m_vtxBuffAlctr = core::GeneralpurposeAddressAllocator<uint32_t>(m_vtxBuffAlctrResSpc, 0u, 0u, 32u, allocParams.vertexBuffSupportedSize, allocParams.vertexBufferMinAllocSize);
        }

        if (allocParams.MDIDataBuffSupportedCnt)
        {
            m_MDIDataAlctrResSpc = _NBL_ALIGNED_MALLOC(core::GeneralpurposeAddressAllocator<uint32_t>::reserved_size(alignof(std::max_align_t), allocParams.MDIDataBuffSupportedCnt, allocParams.MDIDataBuffMinAllocSize), _NBL_SIMD_ALIGNMENT);
            _NBL_DEBUG_BREAK_IF(m_MDIDataAlctrResSpc == nullptr);
            assert(m_MDIDataAlctrResSpc != nullptr);
            m_MDIDataAlctr = core::GeneralpurposeAddressAllocator<uint32_t>(m_MDIDataAlctrResSpc, 0u, 0u, alignof(std::max_align_t), allocParams.MDIDataBuffSupportedCnt, allocParams.MDIDataBuffMinAllocSize);
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

    inline size_t calcVertexSize(const SVertexInputParams& vtxInputParams, const E_VERTEX_INPUT_RATE inputRate) const
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

    inline constexpr uint32_t calcBatchCount(uint32_t triCnt) { return (triCnt + m_maxTriangleCountPerMDIData - 1) / m_maxTriangleCountPerMDIData; }

    struct Triangle
    {
        uint32_t oldIndices[3];
    };

    struct TriangleBatch
    {
        core::vector<Triangle> triangles;
    };

    core::vector<TriangleBatch> constructTriangleBatches(MeshBufferType* meshBuffer)
    {
        const size_t idxCnt = meshBuffer->getIndexCount();
        const uint32_t triCnt = idxCnt / 3;
        assert(idxCnt % 3 == 0);

        const uint32_t batchCount = calcBatchCount(triCnt);

        core::vector<TriangleBatch> output(batchCount);

        for (uint32_t i = 0u; i < batchCount; i++)
        {
            if (i == (batchCount - 1))
            {
                if (triCnt % m_maxTriangleCountPerMDIData)
                {
                    output[i].triangles = core::vector<Triangle>(triCnt % m_maxTriangleCountPerMDIData);
                    continue;
                }
            }

            output[i].triangles = core::vector<Triangle>(m_maxTriangleCountPerMDIData);
        }

        //struct TriangleMortonCodePair
        //{
        //	Triangle triangle;
        //	//uint64_t mortonCode; TODO after benchmarks
        //};

        //TODO: triangle reordering

        const auto& srcIdxBuffer = meshBuffer->getIndexBufferBinding();
        auto idxBufferPtr32Bit = static_cast<const uint32_t*>(srcIdxBuffer.buffer->getPointer()) + (srcIdxBuffer.offset / sizeof(uint32_t)); //will be changed after benchmarks
        auto idxBufferPtr16Bit = static_cast<const uint16_t*>(srcIdxBuffer.buffer->getPointer()) + (srcIdxBuffer.offset / sizeof(uint16_t));
        for (TriangleBatch& batch : output)
        {
            for (Triangle& tri : batch.triangles)
            {
                if (meshBuffer->getIndexType() == EIT_32BIT)
                {
                    tri.oldIndices[0] = *idxBufferPtr32Bit;
                    tri.oldIndices[1] = *(++idxBufferPtr32Bit);
                    tri.oldIndices[2] = *(++idxBufferPtr32Bit);
                    idxBufferPtr32Bit++;
                }
                else if (meshBuffer->getIndexType() == EIT_16BIT)
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

    core::unordered_map<uint32_t, uint16_t> constructNewIndicesFromTriangleBatch(TriangleBatch& batch, uint16_t*& indexBuffPtr)
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

    void deinterleaveAndCopyAttribute(MeshBufferType* meshBuffer, uint16_t attrLocation, const core::unordered_map<uint32_t, uint16_t>& usedVertices, uint8_t* dstAttrPtr)
    {
        uint8_t* srcAttrPtr = meshBuffer->getAttribPointer(attrLocation);
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