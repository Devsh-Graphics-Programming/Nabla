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
        CCPUMeshPackerV2(const AllocationParams& allocParams, uint16_t minTriangleCountPerMDIData = 256u, uint16_t maxTriangleCountPerMDIData = 1024u)
            : base_t(allocParams,minTriangleCountPerMDIData,maxTriangleCountPerMDIData)
        {}

        void instantiateDataStorage();

        //! shrinks byte size of all output buffers, so they are large enough to fit currently allocated contents. Call this function before `instantiateDataStorage`
        void shrinkOutputBuffersSize()
        {
            using traits = core::address_allocator_traits<core::GeneralpurposeAddressAllocator<uint32_t>>;
            uint32_t mdiDataBuffNewSize = m_MDIDataAlctr.safe_shrink_size(0u,traits::max_alignment(m_MDIDataAlctr));
            uint32_t idxBuffNewSize = m_idxBuffAlctr.safe_shrink_size(0u,traits::max_alignment(m_idxBuffAlctr));
            uint32_t vtxBuffNewSize = m_vtxBuffAlctr.safe_shrink_size(0u,traits::max_alignment(m_vtxBuffAlctr));

            // TODO: remove members
            m_allocParams.MDIDataBuffSupportedCnt = mdiDataBuffNewSize;
            m_allocParams.indexBuffSupportedCnt = idxBuffNewSize;
            m_allocParams.vertexBuffSupportedByteSize = vtxBuffNewSize;

            const void* oldReserved = traits::getReservedSpacePtr(m_MDIDataAlctr);
            m_MDIDataAlctr = core::GeneralpurposeAddressAllocator(mdiDataBuffNewSize,std::move(m_MDIDataAlctr),_NBL_ALIGNED_MALLOC(traits::reserved_size(mdiDataBuffNewSize,m_MDIDataAlctr),_NBL_SIMD_ALIGNMENT));
            _NBL_ALIGNED_FREE(const_cast<void*>(oldReserved));

            oldReserved = traits::getReservedSpacePtr(m_idxBuffAlctr);
            m_idxBuffAlctr = core::GeneralpurposeAddressAllocator(idxBuffNewSize,std::move(m_idxBuffAlctr),_NBL_ALIGNED_MALLOC(traits::reserved_size(idxBuffNewSize,m_idxBuffAlctr),_NBL_SIMD_ALIGNMENT));
            _NBL_ALIGNED_FREE(const_cast<void*>(oldReserved));

            oldReserved = traits::getReservedSpacePtr(m_vtxBuffAlctr);
            m_vtxBuffAlctr = core::GeneralpurposeAddressAllocator(vtxBuffNewSize,std::move(m_vtxBuffAlctr),_NBL_ALIGNED_MALLOC(traits::reserved_size(vtxBuffNewSize,m_vtxBuffAlctr),_NBL_SIMD_ALIGNMENT));
            _NBL_ALIGNED_FREE(const_cast<void*>(oldReserved));
        }

        /**
        \return number of mdi structs created for mesh buffer range described by mbBegin .. mbEnd, 0 if commit failed or mbBegin == mbEnd
        */
        template <typename MeshBufferIterator>
        uint32_t commit(IMeshPackerBase::PackedMeshBufferData* pmbdOut, CombinedDataOffsetTable* cdotOut, ReservedAllocationMeshBuffers* rambIn, const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd);

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

        const core::GeneralpurposeAddressAllocator<uint32_t>& getMDIAllocator() const { return m_MDIDataAlctr; }
        const core::GeneralpurposeAddressAllocator<uint32_t>& getIndexAllocator() const { return m_idxBuffAlctr; }
        const core::GeneralpurposeAddressAllocator<uint32_t>& getVertexAllocator() const { return m_vtxBuffAlctr; }

};

template <typename MDIStructType>
void CCPUMeshPackerV2<MDIStructType>::instantiateDataStorage()
{
    m_packerDataStore.MDIDataBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(m_allocParams.MDIDataBuffSupportedCnt * sizeof(MDIStructType));
    m_packerDataStore.indexBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(m_allocParams.indexBuffSupportedCnt * sizeof(uint16_t));
    m_packerDataStore.vertexBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(m_allocParams.vertexBuffSupportedByteSize);
}

/*
    @param pmbdOut size of this array has to be >= std::distance(mbBegin, mbEnd)
    @param cdotOut size of this array has to be >= IMeshPackerV2::calcMDIStructMaxCount(mbBegin, mbEnd)
*/
template <typename MDIStructType>
template <typename MeshBufferIterator>
uint32_t CCPUMeshPackerV2<MDIStructType>::commit(IMeshPackerBase::PackedMeshBufferData* pmbdOut, CombinedDataOffsetTable* cdotOut, ReservedAllocationMeshBuffers* rambIn, const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd)
{
    MDIStructType* mdiBuffPtr = static_cast<MDIStructType*>(m_packerDataStore.MDIDataBuffer->getPointer()) + rambIn->mdiAllocationOffset;

    size_t i = 0ull;
    uint32_t batchCntTotal = 0u;
    for (auto it = mbBegin; it != mbEnd; it++)
    {
        ReservedAllocationMeshBuffers& ramb = *(rambIn + i);
        IMeshPackerBase::PackedMeshBufferData& pmbd = *(pmbdOut + i);

        //this is fucked up..
        //mdiAllocationOffset should be one for all mesh buffers in range defined by mbBegin .. mbEnd, otherwise things get fucked when there are random sizes of batches
        //TODO: so modify ReservedAllocationMeshBuffers and free function
        //MDIStructType* mdiBuffPtr = static_cast<MDIStructType*>(m_packerDataStore.MDIDataBuffer->getPointer()) + ramb.mdiAllocationOffset;
        uint16_t* indexBuffPtr = static_cast<uint16_t*>(m_packerDataStore.indexBuffer->getPointer()) + ramb.indexAllocationOffset;

        const auto& mbVtxInputParams = (*it)->getPipeline()->getVertexInputParams();
        const uint32_t insCnt = (*it)->getInstanceCount();

        IdxBufferParams idxBufferParams = retriveOrCreateNewIdxBufferParams(*it);

        TriangleBatches triangleBatches = constructTriangleBatches(*it, idxBufferParams);

        size_t batchFirstIdx = ramb.indexAllocationOffset;
        size_t verticesAddedCnt = 0u;

        //TODO: check if mpv1 does redundand copies
        std::array<bool, 16> perInsAttribFromThisLocationWasCopied;
        std::fill(perInsAttribFromThisLocationWasCopied.begin(), perInsAttribFromThisLocationWasCopied.end(), false);

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
                    return 0u;

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
                    if (perInsAttribFromThisLocationWasCopied[location] == false)
                    {
                        deinterleaveAndCopyPerInstanceAttribute(*it, location, dstAttrPtr);
                        perInsAttribFromThisLocationWasCopied[location] = true;
                    }
                }

                auto& utb = m_virtualAttribConfig.utbs[VirtualAttribConfig::getUTBArrayTypeFromFormat(attribFormat)];
                auto vtxFormatInfo = utb.find(attribFormat);
                if (vtxFormatInfo==utb.end())
                    return 0u;

                uint16_t vaArrayElement = vtxFormatInfo->second;
                uint32_t vaOffset;

                if (inputRate == EVIR_PER_VERTEX)
                    vaOffset = ramb.attribAllocParams[location].offset / attribSize + verticesAddedCnt;
                if (inputRate == EVIR_PER_INSTANCE)
                    vaOffset = ramb.attribAllocParams[location].offset / attribSize;

                cdotOut->attribInfo[location] = VirtualAttribute(vaArrayElement,vaOffset);

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

}
}

#endif