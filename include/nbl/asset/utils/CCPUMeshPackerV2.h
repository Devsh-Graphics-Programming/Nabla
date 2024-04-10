// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_CPU_MESH_PACKER_V2_H_INCLUDED__
#define __NBL_ASSET_C_CPU_MESH_PACKER_V2_H_INCLUDED__

#include <nbl/asset/ICPUMesh.h>
#include <nbl/asset/utils/IMeshPackerV2.h>

namespace nbl
{
namespace asset
{
#if 0 // REWRITE
template <typename MDIStructType = DrawElementsIndirectCommand_t>
class CCPUMeshPackerV2 final : public IMeshPackerV2<ICPUBuffer,ICPUDescriptorSet,ICPUMeshBuffer,MDIStructType>
{
        using base_t = IMeshPackerV2<ICPUBuffer,ICPUDescriptorSet,ICPUMeshBuffer,MDIStructType>;
        using Triangle = typename base_t::Triangle;
        using TriangleBatches = typename base_t::TriangleBatches;
        using IdxBufferParams = typename base_t::base_t::IdxBufferParams;

    public:
        using AllocationParams = IMeshPackerBase::AllocationParamsCommon;
        using PackerDataStore = typename base_t::PackerDataStore;
        using ReservedAllocationMeshBuffers = typename base_t::ReservedAllocationMeshBuffers;
        using AttribAllocParams = typename base_t::AttribAllocParams;
        using CombinedDataOffsetTable = typename base_t::CombinedDataOffsetTable;

    public:
        CCPUMeshPackerV2(const AllocationParams& allocParams, const IMeshPackerV2Base::SupportedFormatsContainer& formats, uint16_t minTriangleCountPerMDIData = 256u, uint16_t maxTriangleCountPerMDIData = 1024u)
            : base_t(allocParams, formats, minTriangleCountPerMDIData, maxTriangleCountPerMDIData)
        {}

        void instantiateDataStorage();

        /**
        \return number of mdi structs created for mesh buffer range described by mbBegin .. mbEnd, 0 if commit failed or mbBegin == mbEnd
        */
        template <typename MeshBufferIterator>
        uint32_t commit(IMeshPackerBase::PackedMeshBufferData* pmbdOut, CombinedDataOffsetTable* cdotOut, core::aabbox3df* aabbs, ReservedAllocationMeshBuffers* rambIn, const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd);

        inline std::pair<uint32_t,uint32_t> getDescriptorSetWritesForUTB(
            ICPUDescriptorSet::SWriteDescriptorSet* outWrites, ICPUDescriptorSet::SDescriptorInfo* outInfo, ICPUDescriptorSet* dstSet,
            const typename base_t::DSLayoutParamsUTB& params = {}
        ) const
        {
            auto createBufferView = [&](core::smart_refctd_ptr<ICPUBuffer>&& buff, E_FORMAT format) -> core::smart_refctd_ptr<IDescriptor>
            {
                return core::make_smart_refctd_ptr<ICPUBufferView>(std::move(buff),format);
            };
            return base_t::getDescriptorSetWritesForUTB(outWrites,outInfo,dstSet,createBufferView,params);
        }
};

template <typename MDIStructType>
void CCPUMeshPackerV2<MDIStructType>::instantiateDataStorage()
{
    const uint32_t MDIDataBuffByteSize = base_t::m_MDIDataAlctr.get_total_size() * sizeof(MDIStructType);
    const uint32_t idxBuffByteSize = base_t::m_idxBuffAlctr.get_total_size() * sizeof(uint16_t);
    const uint32_t vtxBuffByteSize = base_t::m_vtxBuffAlctr.get_total_size();

    base_t::m_packerDataStore.MDIDataBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(MDIDataBuffByteSize);
    base_t::m_packerDataStore.indexBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(idxBuffByteSize);
    base_t::m_packerDataStore.vertexBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(vtxBuffByteSize);
}

/*
    @param pmbdOut size of this array has to be >= std::distance(mbBegin, mbEnd)
    @param cdotOut size of this array has to be >= IMeshPackerV2::calcMDIStructMaxCount(mbBegin, mbEnd)
*/
template <typename MDIStructType>
template <typename MeshBufferIterator>
uint32_t CCPUMeshPackerV2<MDIStructType>::commit(IMeshPackerBase::PackedMeshBufferData* pmbdOut, CombinedDataOffsetTable* cdotOut, core::aabbox3df* aabbs, ReservedAllocationMeshBuffers* rambIn, const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd)
{
    MDIStructType* mdiBuffPtr = static_cast<MDIStructType*>(base_t::m_packerDataStore.MDIDataBuffer->getPointer()) + rambIn->mdiAllocationOffset;

    size_t i = 0ull;
    uint32_t batchCntTotal = 0u;
    for (auto it = mbBegin; it != mbEnd; it++)
    {
        const ReservedAllocationMeshBuffers& ramb = *(rambIn + i);
        IMeshPackerBase::PackedMeshBufferData& pmbd = *(pmbdOut + i);

        //this is fucked up..
        //mdiAllocationOffset should be one for all mesh buffers in range defined by mbBegin .. mbEnd, otherwise things get fucked when there are random sizes of batches
        //TODO: so modify ReservedAllocationMeshBuffers and free function
        //MDIStructType* mdiBuffPtr = static_cast<MDIStructType*>(m_packerDataStore.MDIDataBuffer->getPointer()) + ramb.mdiAllocationOffset;
        uint16_t* indexBuffPtr = static_cast<uint16_t*>(base_t::m_packerDataStore.indexBuffer->getPointer()) + ramb.indexAllocationOffset;

        const auto& mbVtxInputParams = (*it)->getPipeline()->getVertexInputParams();
        const uint32_t insCnt = (*it)->getInstanceCount();

        IdxBufferParams idxBufferParams = base_t::createNewIdxBufferParamsForNonTriangleListTopologies(*it);

        TriangleBatches triangleBatches = base_t::constructTriangleBatches(*it, idxBufferParams, aabbs);

        size_t batchFirstIdx = ramb.indexAllocationOffset;
        size_t verticesAddedCnt = 0u;

        //TODO: check if mpv1 does redundand copies
        std::array<bool, 16> perInsAttribFromThisLocationWasCopied;
        std::fill(perInsAttribFromThisLocationWasCopied.begin(), perInsAttribFromThisLocationWasCopied.end(), false);

        const uint32_t batchCnt = triangleBatches.ranges.size() - 1u;
        for (uint32_t i = 0u; i < batchCnt; i++)
        {
            auto batchBegin = triangleBatches.ranges[i];
            auto batchEnd = triangleBatches.ranges[i+1];
            const uint32_t triangleInBatchCnt = std::distance(batchBegin,batchEnd);
            constexpr uint32_t kIndicesPerTriangle = 3u;
            const uint32_t idxInBatchCnt = triangleInBatchCnt*kIndicesPerTriangle;

            core::unordered_map<uint32_t, uint16_t> usedVertices = base_t::constructNewIndicesFromTriangleBatchAndUpdateUnifiedIndexBuffer(triangleBatches, i, indexBuffPtr);

            //copy deinterleaved vertices into unified vertex buffer
            for (uint16_t attrBit = 0x0001, location = 0; location < SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; attrBit <<= 1, location++)
            {
                if (!(mbVtxInputParams.enabledAttribFlags & attrBit))
                    continue;

                if (ramb.attribAllocParams[location].offset == base_t::INVALID_ADDRESS)
                    return 0u;

                const E_FORMAT attribFormat = static_cast<E_FORMAT>(mbVtxInputParams.attributes[location].format);
                const uint32_t attribSize = asset::getTexelOrBlockBytesize(attribFormat);
                const uint32_t binding = mbVtxInputParams.attributes[location].binding;
                const E_VERTEX_INPUT_RATE inputRate = mbVtxInputParams.bindings[binding].inputRate;

                uint8_t* dstAttrPtr = static_cast<uint8_t*>(base_t::m_packerDataStore.vertexBuffer->getPointer()) + ramb.attribAllocParams[location].offset;
                
                if (inputRate == EVIR_PER_VERTEX)
                {
                    const uint32_t currBatchOffsetForPerVtxAttribs = verticesAddedCnt * attribSize;
                    dstAttrPtr += currBatchOffsetForPerVtxAttribs;
                    base_t::deinterleaveAndCopyAttribute(*it, location, usedVertices, dstAttrPtr);
                }
                if (inputRate == EVIR_PER_INSTANCE)
                {
                    if (perInsAttribFromThisLocationWasCopied[location] == false)
                    {
                        base_t::deinterleaveAndCopyPerInstanceAttribute(*it, location, dstAttrPtr);
                        perInsAttribFromThisLocationWasCopied[location] = true;
                    }
                }

                auto& utb = base_t::m_virtualAttribConfig.utbs[base_t::VirtualAttribConfig::getUTBArrayTypeFromFormat(attribFormat)];
                auto vtxFormatInfo = utb.find(attribFormat);
                if (vtxFormatInfo==utb.end())
                    return 0u;

                uint16_t vaArrayElement = vtxFormatInfo->second;
                uint32_t vaOffset;

                if (inputRate == EVIR_PER_VERTEX)
                    vaOffset = ramb.attribAllocParams[location].offset / attribSize + verticesAddedCnt;
                if (inputRate == EVIR_PER_INSTANCE)
                    vaOffset = ramb.attribAllocParams[location].offset / attribSize;

                cdotOut->attribInfo[location] = base_t::VirtualAttribute(vaArrayElement,vaOffset);

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

        pmbd = { rambIn->mdiAllocationOffset+batchCntTotal, static_cast<uint32_t>(batchCnt) };
        batchCntTotal += batchCnt;

        i++;
    }

    return batchCntTotal;
}
#endif
}
}

#endif