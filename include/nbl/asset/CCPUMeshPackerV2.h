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
    using TriangleBatches = typename base_t::TriangleBatches;
    using IdxBufferParams = typename base_t::base_t::IdxBufferParams;

    template<typename> friend class CGPUMeshPackerV2; //TODO: this will allow CGPUMeshPackerV2 with every template parameter to be a friend of this class, fix it

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

    //! shrinks byte size of all output buffers, so they are large enough to fit currently allocated contents. Call this function before `instantiateDataStorage`
    void shrinkOutputBuffersSize()
    {
        m_allocParams.MDIDataBuffSupportedCnt = m_MDIDataAlctr.safe_shrink_size(0u, 1u);
        m_allocParams.indexBuffSupportedCnt = m_idxBuffAlctr.safe_shrink_size(0u, 1u);
        m_allocParams.vertexBuffSupportedByteSize = m_vtxBuffAlctr.safe_shrink_size(0u, 1u);
    }

    template <typename MeshBufferIterator>
    bool commit(IMeshPackerBase::PackedMeshBufferData* pmbdOut, CombinedDataOffsetTable* cdotOut, ReservedAllocationMeshBuffers* rambIn, const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd);

    inline PackerDataStore getPackerDataStore() { return m_packerDataStore; };

    uint32_t getDSlayoutBindings(ICPUDescriptorSetLayout::SBinding* outBindings, uint32_t fsamplersBinding = 0u, uint32_t isamplersBinding = 1u, uint32_t usamplersBinding = 2u) const
    {
        return getDSlayoutBindings_internal<ICPUDescriptorSetLayout>(outBindings, fsamplersBinding, isamplersBinding, usamplersBinding);
    }

    // cannot be called before 'instantiateDataStorage'
    std::pair<uint32_t, uint32_t> getDescriptorSetWrites(ICPUDescriptorSet::SWriteDescriptorSet* outWrites, ICPUDescriptorSet::SDescriptorInfo* outInfo, ICPUDescriptorSet* dstSet, uint32_t fBuffersBinding = 0u, uint32_t iBuffersBinding = 1u, uint32_t uBuffersBinding = 2u) const
    {
        auto createBufferView = [&](E_FORMAT format)
        {
            return core::make_smart_refctd_ptr<ICPUBufferView>(core::smart_refctd_ptr(m_packerDataStore.vertexBuffer), format);
        };

        return getDescriptorSetWrites_internal<ICPUDescriptorSet, ICPUBufferView>(outWrites, outInfo, dstSet, createBufferView, fBuffersBinding, iBuffersBinding, uBuffersBinding);
    }

};

template <typename MDIStructType>
void CCPUMeshPackerV2<MDIStructType>::instantiateDataStorage()
{
    m_packerDataStore.MDIDataBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(m_allocParams.MDIDataBuffSupportedCnt * sizeof(MDIStructType));
    m_packerDataStore.indexBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(m_allocParams.indexBuffSupportedCnt * sizeof(uint16_t));
    m_packerDataStore.vertexBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(m_allocParams.vertexBuffSupportedByteSize);
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

        TriangleBatches triangleBatches = constructTriangleBatches(*it, idxBufferParams);

        size_t batchFirstIdx = ramb.indexAllocationOffset;
        size_t verticesAddedCnt = 0u;
        size_t instancesAddedCnt = 0u;

        const uint32_t batchCnt = triangleBatches.ranges.size() - 1u;
        for (uint32_t i = 0u; i < batchCnt; i++)
        {
            auto batchBegin = triangleBatches.ranges[i];
            auto batchEnd = triangleBatches.ranges[i + 1];
            const uint32_t triangleInBatchCnt = std::distance(batchBegin, batchEnd);
            const uint32_t idxInBatchCnt = 3 * triangleInBatchCnt;

            core::unordered_map<uint32_t, uint16_t> usedVertices = constructNewIndicesFromTriangleBatchAndUpdateUnifiedIndexBuffer(triangleBatches, i, indexBuffPtr);

            //copy deinterleaved vertices into unified vertex buffer
            for (uint16_t attrBit = 0x0001, location = 0; location < SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; attrBit <<= 1, location++)
            {
                if (!(mbVtxInputParams.enabledAttribFlags & attrBit))
                    continue;

                if (ramb.attribAllocParams[location].offset == INVALID_ADDRESS)
                    return false;

                const E_FORMAT attribFormat = static_cast<E_FORMAT>(mbVtxInputParams.attributes[location].format);
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

                auto vtxFormatInfo = m_virtualAttribConfig.map.find(attribFormat);

                if (vtxFormatInfo == m_virtualAttribConfig.map.end())
                    return false;

                cdotOut->attribInfo[location].arrayElement = vtxFormatInfo->second.second;

                if (inputRate == EVIR_PER_VERTEX)
                    cdotOut->attribInfo[location].offset = ramb.attribAllocParams[location].offset / attribSize + verticesAddedCnt;
                if (inputRate == EVIR_PER_INSTANCE)
                    cdotOut->attribInfo[location].offset = ramb.attribAllocParams[location].offset / attribSize + instancesAddedCnt;

            }

            verticesAddedCnt += usedVertices.size();
            cdotOut++;

            //construct mdi data
            MDIStructType MDIData;
            MDIData.count = idxInBatchCnt;
            MDIData.instanceCount = (*it)->getInstanceCount();
            MDIData.firstIndex = batchFirstIdx;
            MDIData.baseVertex = 0u;
            MDIData.baseInstance = 0u;

            *mdiBuffPtr = MDIData;
            mdiBuffPtr++;

            batchFirstIdx += idxInBatchCnt;
        }

        instancesAddedCnt += insCnt;

        pmbd = { ramb.mdiAllocationOffset, static_cast<uint32_t>(batchCnt) };

        i++;
    }

    return true;
}

}
}

#endif