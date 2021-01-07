// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_MESH_PACKER_H_INCLUDED__
#define __NBL_ASSET_I_MESH_PACKER_H_INCLUDED__

namespace nbl
{
namespace asset
{

class MeshPackerBase
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

    template <typename MeshIterator>
    struct MeshPackerConfigParams
    {
        SVertexInputParams vertexInputParams;
        core::SRange<void, MeshIterator> belongingMeshes; // pointers to sections of `sortedMeshBuffersOut`
    };

protected:
    MeshPackerBase(uint16_t minTriangleCountPerMDIData, uint16_t maxTriangleCountPerMDIData)
        :m_maxTriangleCountPerMDIData(maxTriangleCountPerMDIData),
         m_minTriangleCountPerMDIData(minTriangleCountPerMDIData),
         m_MDIDataAlctrResSpc(nullptr),
         m_idxBuffAlctrResSpc(nullptr),
         m_vtxBuffAlctrResSpc(nullptr)
    {
        assert(minTriangleCountPerMDIData <= 21845);
        assert(maxTriangleCountPerMDIData <= 21845);
    };

    virtual ~MeshPackerBase()
    {
        _NBL_ALIGNED_FREE(m_MDIDataAlctrResSpc);
        _NBL_ALIGNED_FREE(m_idxBuffAlctrResSpc);
        _NBL_ALIGNED_FREE(m_vtxBuffAlctrResSpc);
    }

    static bool cmpVtxInputParams(const SVertexInputParams& lhs, const SVertexInputParams& rhs)
    {
        if (lhs.enabledAttribFlags != rhs.enabledAttribFlags)
            return false;

        for (uint16_t attrBit = 0x0001, location = 0; location < SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; attrBit <<= 1, location++)
        {
            if (!(attrBit & lhs.enabledAttribFlags))
                continue;

            if (lhs.attributes[location].format != rhs.attributes[location].format ||
                lhs.bindings[lhs.attributes[location].binding].inputRate != rhs.bindings[rhs.attributes[location].binding].inputRate)
                return false;
        }

        return true;
    }

    struct AllocationParamsCommon
    {
        size_t indexBuffSupportedCnt;
        size_t vertexBuffSupportedSize;
        size_t MDIDataBuffSupportedCnt;
        size_t indexBufferMinAllocSize;
        size_t vertexBufferMinAllocSize;
        size_t MDIDataBuffMinAllocSize;
        uint32_t MDIDataBuffMaxAlign;
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
            m_vtxBuffAlctrResSpc = _NBL_ALIGNED_MALLOC(core::GeneralpurposeAddressAllocator<uint32_t>::reserved_size(alignof(std::max_align_t), allocParams.vertexBuffSupportedSize, allocParams.vertexBufferMinAllocSize), _NBL_SIMD_ALIGNMENT);
            //for now mesh packer will not allow mesh buffers without any per vertex attributes
            _NBL_DEBUG_BREAK_IF(m_vtxBuffAlctrResSpc == nullptr);
            assert(m_vtxBuffAlctrResSpc != nullptr);
            m_vtxBuffAlctr = core::GeneralpurposeAddressAllocator<uint32_t>(m_vtxBuffAlctrResSpc, 0u, 0u, alignof(std::max_align_t), allocParams.vertexBuffSupportedSize, allocParams.vertexBufferMinAllocSize);
        }

        if (allocParams.MDIDataBuffSupportedCnt)
        {
            m_MDIDataAlctrResSpc = _NBL_ALIGNED_MALLOC(core::GeneralpurposeAddressAllocator<uint32_t>::reserved_size(allocParams.MDIDataBuffMaxAlign, allocParams.MDIDataBuffSupportedCnt, allocParams.MDIDataBuffMinAllocSize), _NBL_SIMD_ALIGNMENT);
            _NBL_DEBUG_BREAK_IF(m_MDIDataAlctrResSpc == nullptr);
            assert(m_MDIDataAlctrResSpc != nullptr);
            m_MDIDataAlctr = core::GeneralpurposeAddressAllocator<uint32_t>(m_MDIDataAlctrResSpc, 0u, 0u, allocParams.MDIDataBuffMaxAlign, allocParams.MDIDataBuffSupportedCnt, allocParams.MDIDataBuffMinAllocSize);
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

//TODO: allow mesh buffers with only per instance attributes

template <typename MeshBufferType, typename MDIStructType = DrawElementsIndirectCommand_t>
class IMeshPacker : public MeshPackerBase
{
    static_assert(std::is_base_of<DrawElementsIndirectCommand_t, MDIStructType>::value);

public:
    /*
    @param minTriangleCountPerMDIData must be <= 21845
    @param maxTriangleCountPerMDIData must be <= 21845
    */
    IMeshPacker(uint16_t minTriangleCountPerMDIData, uint16_t maxTriangleCountPerMDIData)
        :MeshPackerBase(minTriangleCountPerMDIData, maxTriangleCountPerMDIData)
    {
    }

    //TODO: update comment
    // returns number of distinct mesh packers needed to pack the meshes and a sorted list of meshes by the meshpacker ID they should be packed into, as well as the parameters for the packers
    // `packerParamsOut` should be big enough to fit `std::distance(begin,end)` entries, the return value will tell you how many were actually written
    template<typename MeshIterator>
    static uint32_t getPackerCreationParamsFromMeshBufferRange(const MeshIterator begin, const MeshIterator end, MeshIterator sortedMeshBuffersOut,
        MeshPackerBase::MeshPackerConfigParams<MeshIterator>* packerParamsOut)
    {
        assert(begin <= end);
        if (begin == end)
            return 0;

        uint32_t packersNeeded = 1u;

        MeshPackerBase::MeshPackerConfigParams<MeshIterator> firstInpuParams
        {
            (*begin)->getPipeline()->getVertexInputParams(),
            SRange<void, MeshIterator>(sortedMeshBuffersOut, sortedMeshBuffersOut)
        };
        memcpy(packerParamsOut, &firstInpuParams, sizeof(SVertexInputParams));

        //fill array
        auto test1 = std::distance(begin, end);
        auto* packerParamsOutEnd = packerParamsOut + 1u;
        for (MeshIterator it = begin + 1; it != end; it++)
        {
            auto& currMeshVtxInputParams = (*it)->getPipeline()->getVertexInputParams();

            bool alreadyInserted = false;
            for (auto* packerParamsIt = packerParamsOut; packerParamsIt != packerParamsOutEnd; packerParamsIt++)
            {
                alreadyInserted = cmpVtxInputParams(packerParamsIt->vertexInputParams, currMeshVtxInputParams);

                if (alreadyInserted)
                    break;
            }

            if (!alreadyInserted)
            {
                packersNeeded++;

                MeshPackerBase::MeshPackerConfigParams<MeshIterator> configParams
                {
                    currMeshVtxInputParams,
                    SRange<void, MeshIterator>(sortedMeshBuffersOut, sortedMeshBuffersOut)
                };
                memcpy(packerParamsOutEnd, &configParams, sizeof(SVertexInputParams));
                packerParamsOutEnd++;
            }
        }

        auto getIndexOfArrayElement = [&](const SVertexInputParams& vtxInputParams) -> int32_t
        {
            int32_t offset = 0u;
            for (auto* it = packerParamsOut; it != packerParamsOutEnd; it++, offset++)
            {
                if (cmpVtxInputParams(vtxInputParams, it->vertexInputParams))
                    return offset;

                if (it == packerParamsOut - 1)
                    return -1;
            }
        };

        //sort meshes by SVertexInputParams
        const MeshIterator sortedMeshBuffersOutEnd = sortedMeshBuffersOut + std::distance(begin, end);

        std::copy(begin, end, sortedMeshBuffersOut);
        std::sort(sortedMeshBuffersOut, sortedMeshBuffersOutEnd,
            [&](const MeshBufferType* lhs, const MeshBufferType* rhs)
            {
                return getIndexOfArrayElement(lhs->getPipeline()->getVertexInputParams()) < getIndexOfArrayElement(rhs->getPipeline()->getVertexInputParams());
            }
        );

        //set ranges
        MeshIterator sortedMeshBuffersIt = sortedMeshBuffersOut;
        for (auto* inputParamsIt = packerParamsOut; inputParamsIt != packerParamsOutEnd; inputParamsIt++)
        {
            MeshIterator firstMBForThisRange = sortedMeshBuffersIt;
            MeshIterator lastMBForThisRange = sortedMeshBuffersIt;
            for (MeshIterator it = firstMBForThisRange; it != sortedMeshBuffersOutEnd; it++)
            {
                if (!cmpVtxInputParams(inputParamsIt->vertexInputParams, (*it)->getPipeline()->getVertexInputParams()))
                {
                    lastMBForThisRange = it;
                    break;
                }
            }

            if (inputParamsIt == packerParamsOutEnd - 1)
                lastMBForThisRange = sortedMeshBuffersOutEnd;

            inputParamsIt->belongingMeshes = SRange<void, MeshIterator>(firstMBForThisRange, lastMBForThisRange);
            sortedMeshBuffersIt = lastMBForThisRange;
        }

        return packersNeeded;
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

public:
    struct Triangle
    {
        uint32_t oldIndices[3];
    };

    struct TriangleBatch
    {
        core::vector<Triangle> triangles;
    };

    virtual core::vector<TriangleBatch> constructTriangleBatches(MeshBufferType& meshBuffer) = 0;

protected:
    _NBL_STATIC_INLINE_CONSTEXPR uint32_t INVALID_ADDRESS = core::GeneralpurposeAddressAllocator<uint32_t>::invalid_address;

};

}
}

#endif