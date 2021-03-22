// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_CPU_MESH_PACKER_V2_H_INCLUDED__
#define __NBL_ASSET_C_CPU_MESH_PACKER_V2_H_INCLUDED__

#include <nbl/asset/ICPUMesh.h>
#include <nbl/asset/IMeshPackerV2.h>

namespace nbl
{
namespace asset
{

template <typename MDIStructType = DrawElementsIndirectCommand_t>
class CCPUMeshPackerV2 final : public IMeshPackerV2<ICPUBuffer, ICPUMeshBuffer, MDIStructType>
{
    using base_t = IMeshPackerV2<ICPUBuffer, ICPUMeshBuffer, MDIStructType>;
    using Triangle = typename base_t::Triangle;
    using TriangleBatch = typename base_t::TriangleBatch;
    using IdxBufferParams = typename base_t::base_t::IdxBufferParams;

public:
    using AllocationParams = IMeshPackerBase::AllocationParamsCommon;
    using PackerDataStore = typename base_t::PackerDataStore;
    using ReservedAllocationMeshBuffers = typename base_t::ReservedAllocationMeshBuffers;
    using AttribAllocParams = typename base_t::AttribAllocParams;
    using CombinedDataOffsetTable = typename base_t::CombinedDataOffsetTable;

public:
    CCPUMeshPackerV2(const AllocationParams& allocParams, uint16_t minTriangleCountPerMDIData = 256u, uint16_t maxTriangleCountPerMDIData = 1024u)
        :IMeshPackerV2<ICPUBuffer, ICPUMeshBuffer, MDIStructType>(allocParams, minTriangleCountPerMDIData, maxTriangleCountPerMDIData)
    {}

    void instantiateDataStorage();

    template <typename MeshBufferIterator>
    bool commit(IMeshPackerBase::PackedMeshBufferData* pmbdOut, CombinedDataOffsetTable* cdotOut, ReservedAllocationMeshBuffers* rambIn, const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd);

    inline PackerDataStore getPackerDataStore() { return m_packerDataStore; };

};

template <typename MDIStructType>
void CCPUMeshPackerV2<MDIStructType>::instantiateDataStorage()
{
    m_packerDataStore.MDIDataBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(m_allocParams.MDIDataBuffSupportedCnt * sizeof(MDIStructType));
    m_packerDataStore.indexBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(m_allocParams.indexBuffSupportedCnt * sizeof(uint16_t));
    m_packerDataStore.vertexBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(m_allocParams.vertexBuffSupportedSize);
}

template <typename MDIStructType>
template <typename MeshBufferIterator>
bool CCPUMeshPackerV2<MDIStructType>::commit(IMeshPackerBase::PackedMeshBufferData* pmbdOut, CombinedDataOffsetTable* cdotOut, ReservedAllocationMeshBuffers* rambIn, const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd)
{
    size_t i = 0ull;
    for (auto it = mbBegin; it != mbEnd; it++)
    {
        ReservedAllocationMeshBuffers& ramb = *(rambIn + i);
        IMeshPackerBase::PackedMeshBufferData& pmbd = *(pmbdOut + i);

        MDIStructType* mdiBuffPtr = static_cast<MDIStructType*>(m_packerDataStore.MDIDataBuffer->getPointer()) + ramb.mdiAllocationOffset;
        uint16_t* indexBuffPtr = static_cast<uint16_t*>(m_packerDataStore.indexBuffer->getPointer()) + ramb.indexAllocationOffset;

        const auto& mbVtxInputParams = (*it)->getPipeline()->getVertexInputParams();
        const uint32_t insCnt = (*it)->getInstanceCount();

        IdxBufferParams idxBufferParams = retriveOrCreateNewIdxBufferParams(*it);

        core::vector<TriangleBatch> triangleBatches = constructTriangleBatches(*it, idxBufferParams);

        size_t batchFirstIdx = ramb.indexAllocationOffset;
        size_t verticesAddedCnt = 0u;
        size_t instancesAddedCnt = 0u;

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

                const E_FORMAT attribFormat = static_cast<E_FORMAT>(mbVtxInputParams.attributes[location].format);
                //should I cashe it?
                const uint32_t attribSize = asset::getTexelOrBlockBytesize(attribFormat);
                const uint32_t binding = mbVtxInputParams.attributes[location].binding;
                const E_VERTEX_INPUT_RATE inputRate = mbVtxInputParams.bindings[binding].inputRate;

                uint8_t* dstAttrPtr = static_cast<uint8_t*>(m_packerDataStore.vertexBuffer->getPointer()) + ramb.attribAllocParams[location].offset;
                
                if (inputRate == EVIR_PER_VERTEX)
                {
                    const uint32_t currBatchOffsetForPerVtxAttribs = verticesAddedCnt * attribSize;
                    dstAttrPtr += currBatchOffsetForPerVtxAttribs;
                    deinterleaveAndCopyAttribute(*it, location, usedVertices, dstAttrPtr);
                }
                if (inputRate == EVIR_PER_INSTANCE)
                {
                    const uint32_t currBatchOffsetForPerInstanceAttribs = instancesAddedCnt * attribSize;
                    dstAttrPtr += currBatchOffsetForPerInstanceAttribs;
                    deinterleaveAndCopyPerInstanceAttribute(*it, location, dstAttrPtr);
                }

                auto vtxFormatInfo = virtualAttribConfig.map.find(attribFormat);

                if (vtxFormatInfo == virtualAttribConfig.map.end())
                    return false;

                cdotOut->attribInfo[location].arrayElement = vtxFormatInfo->second.second;

                if (inputRate == EVIR_PER_VERTEX)
                    cdotOut->attribInfo[location].offset = ramb.attribAllocParams[location].offset / attribSize + verticesAddedCnt;
                if (inputRate == EVIR_PER_INSTANCE)
                    cdotOut->attribInfo[location].offset = ramb.attribAllocParams[location].offset / attribSize + instancesAddedCnt;

            }

            verticesAddedCnt += usedVertices.size();
            instancesAddedCnt += insCnt;
            cdotOut++;

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

}
}

#endif