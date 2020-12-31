// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_CPU_MESH_PACKER_V2_H_INCLUDED__
#define __NBL_ASSET_C_CPU_MESH_PACKER_V2_H_INCLUDED__

#include <nbl/asset/ICPUMesh.h>
#include <nbl/asset/IMeshPacker.h>

namespace nbl
{
namespace asset
{

// eventually I will rename it
template <typename MDIStructType = DrawElementsIndirectCommand_t>
class CCPUMeshPackerV2 final : public IMeshPacker<ICPUMeshBuffer, MDIStructType>
{
    using base_t = IMeshPacker<ICPUMeshBuffer, MDIStructType>;
    using Triangle = typename base_t::Triangle;
    using TriangleBatch = typename base_t::TriangleBatch;

public:
    struct AttribAllocParams
    {
        size_t offset = core::GeneralpurposeAddressAllocator<uint32_t>::invalid_address;
        size_t size = 0u;
    };

	struct AllocData
    {
        uint32_t mdiAllocationOffset;
        uint32_t mdiAllocationReservedSize;
        uint32_t indexAllocationOffset;
        uint32_t indexAllocationReservedSize;
        AttribAllocParams attribAllocParams[16u];

        inline bool isValid()
        {
            return this->mdiAllocationOffset != core::GeneralpurposeAddressAllocator<uint32_t>::invalid_address;
        }
    };

    struct PackedMeshBufferV2
    {
        core::smart_refctd_ptr<video::IGPUBuffer> MDIDataBuffer;
        core::smart_refctd_ptr<video::IGPUBuffer> vertexBuffer;
        core::smart_refctd_ptr<video::IGPUBuffer> indexBuffer;

        inline bool isValid()
        {
            return this->MDIDataBuffer->getPointer() != nullptr;
        }
    };

public:
	CCPUMeshPackerV2(const SVertexInputParams& preDefinedLayout, const MeshPackerBase::AllocationParams& allocParams, uint16_t minTriangleCountPerMDIData = 256u, uint16_t maxTriangleCountPerMDIData = 1024u)
		:IMeshPacker<ICPUMeshBuffer, MDIStructType>(preDefinedLayout, allocParams, minTriangleCountPerMDIData, maxTriangleCountPerMDIData)
	{

	}

	template <typename Iterator>
	bool alloc(AllocData* allocDataOut, const Iterator begin, const Iterator end);

    void instantiateDataStorage();

    template <typename Iterator>
    bool commit(MeshPackerBase::PackedMeshBufferData* pmbdOut, AllocData* allocDataIn, const Iterator begin, const Iterator end);

    PackedMeshBufferV2 createGPUPackedMeshBuffer(video::IVideoDriver* driver)
    {
        PackedMeshBufferV2 output;
        output.MDIDataBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(MDIBuffer->getSize(), MDIBuffer->getPointer());
        output.vertexBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(vtxBuffer->getSize(), vtxBuffer->getPointer());
        output.indexBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(idxBuffer->getSize(), idxBuffer->getPointer());

        return output;
    }

    virtual core::vector<TriangleBatch> constructTriangleBatches(ICPUMeshBuffer& meshBuffer);

private:
    core::smart_refctd_ptr<ICPUBuffer> vtxBuffer;
    core::smart_refctd_ptr<ICPUBuffer> idxBuffer;
    core::smart_refctd_ptr<ICPUBuffer> MDIBuffer;

};

template <typename MDIStructType>
template <typename Iterator>
bool CCPUMeshPackerV2<MDIStructType>::alloc(AllocData* allocDataOut, const Iterator begin, const Iterator end)
{
    size_t i = 0ull;
    for (auto it = begin; it != end; it++)
    {
        AllocData& allocData = *(allocDataOut + i);
        const size_t vtxCnt = (*it)->calcVertexCount();
        const size_t idxCnt = (*it)->getIndexCount();

        //allocate indices
        allocData.indexAllocationOffset = m_idxBuffAlctr.alloc_addr(idxCnt, 2u);
        if (allocData.indexAllocationOffset == INVALID_ADDRESS)
            return false;
        allocData.indexAllocationReservedSize = idxCnt * 2;

        //allocate vertices
        const auto& mbVtxInputParams = (*it)->getPipeline()->getVertexInputParams();
        for (uint16_t attrBit = 0x0001, location = 0; location < SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; attrBit <<= 1, location++)
        {
            if (!(attrBit && mbVtxInputParams.enabledAttribFlags))
                continue;

            const uint32_t attribSize = asset::getTexelOrBlockBytesize(static_cast<E_FORMAT>(mbVtxInputParams.attributes[location].format));

            //TODO: convert non PoT formats 
            if (!core::isPoT(attribSize))
                return false;

            allocData.attribAllocParams[location].offset = m_vtxBuffAlctr.alloc_addr(vtxCnt * attribSize, attribSize);

            if (allocData.attribAllocParams[location].offset == INVALID_ADDRESS)
                return false;

            allocData.attribAllocParams[location].size = vtxCnt * attribSize;
        }

        //allocate MDI structs
        const uint32_t minIdxCntPerPatch = m_minTriangleCountPerMDIData * 3;
        size_t possibleMDIStructsNeededCnt = (idxCnt + minIdxCntPerPatch - 1) / minIdxCntPerPatch;

        allocData.mdiAllocationOffset = m_MDIDataAlctr.alloc_addr(possibleMDIStructsNeededCnt, 1u);
        if (allocData.mdiAllocationOffset == INVALID_ADDRESS)
            return false;
        allocData.mdiAllocationReservedSize = possibleMDIStructsNeededCnt;

        i++;
    }

	return true;
}

template <typename MDIStructType>
void CCPUMeshPackerV2<MDIStructType>::instantiateDataStorage()
{
    MDIBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(m_allocParams.MDIDataBuffSupportedCnt * sizeof(MDIStructType));
    idxBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(m_allocParams.indexBuffSupportedCnt * sizeof(uint16_t));
    //TODO: fix after new `AllocationParams` is done
    vtxBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(m_allocParams.vertexBuffSupportedCnt);
}

template <typename MDIStructType>
template <typename Iterator>
bool CCPUMeshPackerV2<MDIStructType>::commit(MeshPackerBase::PackedMeshBufferData* pmbdOut, AllocData* allocDataIn, const Iterator begin, const Iterator end)
{
    return false;
}

template<typename MDIStructType>
auto CCPUMeshPackerV2<MDIStructType>::constructTriangleBatches(ICPUMeshBuffer& meshBuffer) -> core::vector<typename base_t::TriangleBatch>
{
    return core::vector<IMeshPacker::TriangleBatch>();
}

}
}

#endif

//TODO
/*refactor:
	- IMeshPacker::ReservedAllocationMeshBuffers applies only to the CCPUMeshPacker, move it
    - this packer doesn't use `m_perInsVtxBuffAlctr`, move it from `IMeshPacker` to `CCPUMeshPacker`
    - constructor of this packer doesn't need `SVertexInputParams` as an input parameter, move code associated with `SVertexInputParams` from constructor of the `IMeshPacker` to the constructor of the `CCPUMeshPacker`
    - fix `AllocationParams` and `PackedMeshBuffer`
*/

//AllocData as SoA?