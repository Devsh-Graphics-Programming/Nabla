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
    using AllocationParams = IMeshPackerBase::AllocationParamsCommon;

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

    template<typename BufferType>
    struct PackerDataStore
    {
        core::smart_refctd_ptr<BufferType> MDIDataBuffer;
        core::smart_refctd_ptr<BufferType> vertexBuffer;
        core::smart_refctd_ptr<BufferType> indexBuffer;

        inline bool isValid()
        {
            return this->MDIDataBuffer->getPointer() != nullptr;
        }
    };

public:
    CCPUMeshPackerV2(const AllocationParams& allocParams, uint16_t minTriangleCountPerMDIData = 256u, uint16_t maxTriangleCountPerMDIData = 1024u);

	template <typename Iterator>
	bool alloc(ReservedAllocationMeshBuffers* rambOut, const Iterator begin, const Iterator end);

    void instantiateDataStorage();

    template <typename Iterator>
    bool commit(IMeshPackerBase::PackedMeshBufferData* pmbdOut, ReservedAllocationMeshBuffers* rambIn, const Iterator begin, const Iterator end);

    PackerDataStore<video::IGPUBuffer> createGPUPackerDataStore(video::IVideoDriver* driver);
    inline PackerDataStore<ICPUBuffer> getCPUPackerDataStore() { return m_packerDataStore; };

private:

    PackerDataStore<ICPUBuffer> m_packerDataStore;

    const AllocationParams m_allocParams;

};

template <typename MDIStructType>
CCPUMeshPackerV2<MDIStructType>::CCPUMeshPackerV2(const AllocationParams& allocParams, uint16_t minTriangleCountPerMDIData, uint16_t maxTriangleCountPerMDIData)
    :IMeshPacker<ICPUMeshBuffer, MDIStructType>(minTriangleCountPerMDIData, maxTriangleCountPerMDIData),
    m_allocParams(allocParams)
{
    initializeCommonAllocators(allocParams);
}

template <typename MDIStructType>
template <typename Iterator>
bool CCPUMeshPackerV2<MDIStructType>::alloc(ReservedAllocationMeshBuffers* rambOut, const Iterator begin, const Iterator end)
{
    size_t i = 0ull;
    for (auto it = begin; it != end; it++)
    {
        ReservedAllocationMeshBuffers& ramb = *(rambOut + i);
        const size_t vtxCnt = (*it)->calcVertexCount();
        const size_t idxCnt = (*it)->getIndexCount();

        //allocate indices
        ramb.indexAllocationOffset = m_idxBuffAlctr.alloc_addr(idxCnt, 2u);
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

            ramb.attribAllocParams[location].offset = m_vtxBuffAlctr.alloc_addr(vtxCnt * attribSize, attribSize);

            if (ramb.attribAllocParams[location].offset == INVALID_ADDRESS)
                return false;

            ramb.attribAllocParams[location].size = vtxCnt * attribSize;
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

template <typename MDIStructType>
void CCPUMeshPackerV2<MDIStructType>::instantiateDataStorage()
{
    m_packerDataStore.MDIDataBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(m_allocParams.MDIDataBuffSupportedCnt * sizeof(MDIStructType));
    m_packerDataStore.indexBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(m_allocParams.indexBuffSupportedCnt * sizeof(uint16_t));
    m_packerDataStore.vertexBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(m_allocParams.vertexBuffSupportedSize);
}

template <typename MDIStructType>
template <typename Iterator>
bool CCPUMeshPackerV2<MDIStructType>::commit(IMeshPackerBase::PackedMeshBufferData* pmbdOut, ReservedAllocationMeshBuffers* rambIn, const Iterator begin, const Iterator end)
{
    size_t i = 0ull;
    for (auto it = begin; it != end; it++)
    {
        ReservedAllocationMeshBuffers& ramb = *(rambIn + i);
        IMeshPackerBase::PackedMeshBufferData& pmbd = *(pmbdOut + i);

        MDIStructType* mdiBuffPtr = static_cast<MDIStructType*>(m_packerDataStore.MDIDataBuffer->getPointer()) + ramb.mdiAllocationOffset;
        uint16_t* indexBuffPtr = static_cast<uint16_t*>(m_packerDataStore.indexBuffer->getPointer()) + ramb.indexAllocationOffset;

        const auto& mbVtxInputParams = (*it)->getPipeline()->getVertexInputParams();

        core::vector<TriangleBatch> triangleBatches = constructTriangleBatches(*it);

        size_t batchFirstIdx = ramb.indexAllocationOffset;

        for (TriangleBatch& batch : triangleBatches)
        {
            core::unordered_map<uint32_t, uint16_t> usedVertices = constructNewIndicesFromTriangleBatch(batch, indexBuffPtr);

            //copy deinterleaved vertices into unified vertex buffer
            for (uint16_t attrBit = 0x0001, location = 0; location < SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; attrBit <<= 1, location++)
            {
                if (!(mbVtxInputParams.enabledAttribFlags & attrBit))
                    continue;

                if (ramb.attribAllocParams[location].offset == INVALID_ADDRESS)
                    return false;

                uint8_t* dstAttrPtr = static_cast<uint8_t*>(m_packerDataStore.vertexBuffer->getPointer()) + ramb.attribAllocParams[location].offset;
                deinterleaveAndCopyAttribute(*it, location, usedVertices, dstAttrPtr, true);
            }

            //construct mdi data
            MDIStructType MDIData;
            MDIData.count = batch.triangles.size() * 3u;
            MDIData.instanceCount = (*it)->getInstanceCount();
            MDIData.firstIndex = batchFirstIdx;
            MDIData.baseVertex = 0u;
            MDIData.baseInstance = 0u;

            *mdiBuffPtr = MDIData;
            mdiBuffPtr++;

            batchFirstIdx += 3u * batch.triangles.size();
        }

        pmbd = { ramb.mdiAllocationOffset, static_cast<uint32_t>(triangleBatches.size()) };

        i++;
    }

    return true;
}

template <typename MDIStructType>
typename CCPUMeshPackerV2<MDIStructType>::PackerDataStore<video::IGPUBuffer> CCPUMeshPackerV2<MDIStructType>::createGPUPackerDataStore(video::IVideoDriver* driver)
{
    PackerDataStore<video::IGPUBuffer> m_output;
    m_output.MDIDataBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(m_packerDataStore.MDIDataBuffer->getSize(), m_packerDataStore.MDIDataBuffer->getPointer());
    m_output.vertexBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(m_packerDataStore.vertexBuffer->getSize(), m_packerDataStore.vertexBuffer->getPointer());
    m_output.indexBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(m_packerDataStore.indexBuffer->getSize(), m_packerDataStore.indexBuffer->getPointer());

    return m_output;
}

}
}

#endif