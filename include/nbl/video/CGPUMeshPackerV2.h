// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_GPU_MESH_PACKER_V2_H_INCLUDED__
#define __NBL_ASSET_C_GPU_MESH_PACKER_V2_H_INCLUDED__

#include <nbl/video/IGPUMesh.h>
#include <nbl/video/IGPUDescriptorSetLayout.h>
#include <nbl/asset/utils/IMeshPackerV2.h>
#include <nbl/asset/utils/CCPUMeshPackerV2.h>

using namespace nbl::video;

namespace nbl
{
namespace video
{
template<typename MDIStructType = asset::DrawElementsIndirectCommand_t>
class CGPUMeshPackerV2 final : public asset::IMeshPackerV2<IGPUBuffer, IGPUDescriptorSet, IGPUMeshBuffer, MDIStructType>
{
    using base_t = asset::IMeshPackerV2<IGPUBuffer, IGPUDescriptorSet, IGPUMeshBuffer, MDIStructType>;
    using Triangle = typename base_t::Triangle;
    using TriangleBatches = typename base_t::TriangleBatches;

public:
    using AllocationParams = typename base_t::AllocationParamsCommon;
    using PackerDataStore = typename base_t::PackerDataStore;
    using ReservedAllocationMeshBuffers = typename base_t::ReservedAllocationMeshBuffers;
    using AttribAllocParams = typename base_t::AttribAllocParams;

public:
    CGPUMeshPackerV2(IVideoDriver* driver, const AllocationParams& allocParams, const asset::IMeshPackerV2Base::SupportedFormatsContainer& formats, uint16_t minTriangleCountPerMDIData = 256u, uint16_t maxTriangleCountPerMDIData = 1024u)
        : base_t(allocParams, formats, minTriangleCountPerMDIData, maxTriangleCountPerMDIData), m_driver(driver)
    {}

    // TODO: protect against empty cpuMP (no allocations and then shrinked)
    CGPUMeshPackerV2(IVideoDriver* driver, const asset::CCPUMeshPackerV2<MDIStructType>* cpuMP)
        : base_t(cpuMP), m_driver(driver)
    {
        // TODO: protect against unitiliazed storage of cpuMP
        const auto& cpuMDIBuff = cpuMP->getPackerDataStore().MDIDataBuffer;
        const auto& cpuIdxBuff = cpuMP->getPackerDataStore().indexBuffer;
        const auto& cpuVtxBuff = cpuMP->getPackerDataStore().vertexBuffer;

        // TODO: call this->instantiateDataStorage() here and then copy CPU data to the initialized storage
        m_packerDataStore.MDIDataBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(cpuMDIBuff->getSize(), cpuMDIBuff->getPointer());
        m_packerDataStore.indexBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(cpuIdxBuff->getSize(), cpuIdxBuff->getPointer());
        m_packerDataStore.vertexBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(cpuVtxBuff->getSize(), cpuVtxBuff->getPointer());
    }

    void instantiateDataStorage();

    template<typename MeshBufferIterator>
    bool commit(typename base_t::PackedMeshBufferData* pmbdOut, ReservedAllocationMeshBuffers* rambIn, core::aabbox3df* aabbs, const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd);

    inline std::pair<uint32_t, uint32_t> getDescriptorSetWritesForUTB(
        IGPUDescriptorSet::SWriteDescriptorSet* outWrites, IGPUDescriptorSet::SDescriptorInfo* outInfo, IGPUDescriptorSet* dstSet,
        const typename base_t::DSLayoutParamsUTB& params = {}) const
    {
        auto createBufferView = [&](core::smart_refctd_ptr<IGPUBuffer>&& buff, E_FORMAT format) -> core::smart_refctd_ptr<IDescriptor> {
            return m_driver->createGPUBufferView(buff.get(), format);
        };
        return base_t::getDescriptorSetWritesForUTB(outWrites, outInfo, dstSet, createBufferView, params);
    }

private:
    IVideoDriver* m_driver;
};

template<typename MDIStructType>
void CGPUMeshPackerV2<MDIStructType>::instantiateDataStorage()
{
    const uint32_t MDIDataBuffByteSize = m_MDIDataAlctr.get_total_size() * sizeof(MDIStructType);
    const uint32_t idxBuffByteSize = m_idxBuffAlctr.get_total_size() * sizeof(uint16_t);
    const uint32_t vtxBuffByteSize = m_vtxBuffAlctr.get_total_size();

    m_packerDataStore.MDIDataBuffer = m_driver->createDeviceLocalGPUBufferOnDedMem(MDIDataBuffByteSize);
    m_packerDataStore.indexBuffer = m_driver->createDeviceLocalGPUBufferOnDedMem(idxBuffByteSize);
    m_packerDataStore.vertexBuffer = m_driver->createDeviceLocalGPUBufferOnDedMem(vtxBuffByteSize);
}

template<typename MDIStructType>
template<typename MeshBufferIterator>
bool CGPUMeshPackerV2<MDIStructType>::commit(typename base_t::PackedMeshBufferData* pmbdOut, ReservedAllocationMeshBuffers* rambIn, core::aabbox3df* aabbs, const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd)
{
    assert(0);
    return false;
}

}
}

#endif