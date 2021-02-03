// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_MESH_PACKER_V2_H_INCLUDED__
#define __NBL_ASSET_I_MESH_PACKER_V2_H_INCLUDED__

#include <nbl/asset/utils/IMeshPacker.h>

namespace nbl
{
namespace asset
{

template <typename BufferType, typename MeshBufferType, typename MDIStructType = DrawElementsIndirectCommand_t>
class IMeshPackerV2 : public IMeshPacker<MeshBufferType, MDIStructType>
{
    static_assert(std::is_base_of<IBuffer, BufferType>::value);

	using base_t = IMeshPacker<MeshBufferType, MDIStructType>;
    using AllocationParams = IMeshPackerBase::AllocationParamsCommon;

public:
    struct AttribAllocParams
    {
        size_t offset = INVALID_ADDRESS;
        size_t size = 0ull;
    };

    struct ReservedAllocationMeshBuffers
    {
        uint32_t mdiAllocationOffset;
        uint32_t mdiAllocationReservedSize;
        uint32_t indexAllocationOffset;
        uint32_t indexAllocationReservedSize;
        AttribAllocParams attribAllocParams[SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT];

        inline bool isValid()
        {
            return this->mdiAllocationOffset != core::GeneralpurposeAddressAllocator<uint32_t>::invalid_address;
        }
    };

    struct PackerDataStore : base_t::template PackerDataStoreCommon<BufferType>
    {
        core::smart_refctd_ptr<BufferType> vertexBuffer;
        core::smart_refctd_ptr<BufferType> indexBuffer;
    };

protected:
	IMeshPackerV2(const AllocationParams& allocParams, uint16_t minTriangleCountPerMDIData, uint16_t maxTriangleCountPerMDIData)
		:base_t(minTriangleCountPerMDIData, maxTriangleCountPerMDIData),
         m_allocParams(allocParams)
	{
        initializeCommonAllocators(allocParams);
    };

public:
	template <typename MeshBufferIterator>
	bool alloc(ReservedAllocationMeshBuffers* rambOut, const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd);

    inline PackerDataStore getPackerDataStore() { return m_packerDataStore; };

protected:
    PackerDataStore m_packerDataStore;
    const AllocationParams m_allocParams;

};

template <typename MeshBufferType, typename BufferType, typename MDIStructType>
template <typename MeshBufferIterator>
bool IMeshPackerV2<MeshBufferType, BufferType, MDIStructType>::alloc(ReservedAllocationMeshBuffers* rambOut, const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd)
{
    size_t i = 0ull;
    for (auto it = mbBegin; it != mbEnd; it++)
    {
        ReservedAllocationMeshBuffers& ramb = *(rambOut + i);
        const size_t idxCnt = (*it)->getIndexCount();
        const size_t maxVtxCnt = (idxCnt + 1u) / 2u;

        //allocate indices
        ramb.indexAllocationOffset = m_idxBuffAlctr.alloc_addr(idxCnt, 1u);
        if (ramb.indexAllocationOffset == INVALID_ADDRESS)
            return false;
        ramb.indexAllocationReservedSize = idxCnt * 2;

        //allocate vertices
        const auto& mbVtxInputParams = (*it)->getPipeline()->getVertexInputParams();
        for (uint16_t attrBit = 0x0001, location = 0; location < SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; attrBit <<= 1, location++)
        {
            if (!(attrBit & mbVtxInputParams.enabledAttribFlags))
                continue;

            const uint32_t attribSize = asset::getTexelOrBlockBytesize(static_cast<E_FORMAT>(mbVtxInputParams.attributes[location].format));

            ramb.attribAllocParams[location].offset = m_vtxBuffAlctr.alloc_addr(maxVtxCnt * attribSize, attribSize);

            if (ramb.attribAllocParams[location].offset == INVALID_ADDRESS)
                return false;

            ramb.attribAllocParams[location].size = maxVtxCnt * attribSize;
        }

        //allocate MDI structs
        const uint32_t minIdxCntPerPatch = m_minTriangleCountPerMDIData * 3;
        size_t possibleMDIStructsNeededCnt = (idxCnt + minIdxCntPerPatch - 1) / minIdxCntPerPatch;

        ramb.mdiAllocationOffset = m_MDIDataAlctr.alloc_addr(possibleMDIStructsNeededCnt, 1u);
        if (ramb.mdiAllocationOffset == INVALID_ADDRESS)
            return false;
        ramb.mdiAllocationReservedSize = possibleMDIStructsNeededCnt;

        i++;
    }

    return true;
}

}
}

#endif