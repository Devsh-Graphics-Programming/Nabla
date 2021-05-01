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
namespace asset
{

template <typename MDIStructType = DrawElementsIndirectCommand_t>
class CGPUMeshPackerV2 final : public IMeshPackerV2<video::IGPUBuffer,video::IGPUMeshBuffer,MDIStructType>
{
        using base_t = IMeshPackerV2<video::IGPUBuffer,video::IGPUMeshBuffer,MDIStructType>;
        using Triangle = typename base_t::Triangle;
        using TriangleBatches = typename base_t::TriangleBatches;

    public:
        using AllocationParams = IMeshPackerBase::AllocationParamsCommon;
        using PackerDataStore = typename base_t::PackerDataStore;
        using ReservedAllocationMeshBuffers = typename base_t::ReservedAllocationMeshBuffers;
        using AttribAllocParams = typename base_t::AttribAllocParams;

    public:
        CGPUMeshPackerV2(video::IVideoDriver* driver, const AllocationParams& allocParams, uint16_t minTriangleCountPerMDIData = 256u, uint16_t maxTriangleCountPerMDIData = 1024u)
            :IMeshPackerV2<video::IGPUBuffer, video::IGPUMeshBuffer, MDIStructType>(allocParams, minTriangleCountPerMDIData, maxTriangleCountPerMDIData),
             m_driver(driver)
        {}

        // TODO: protect against empty cpuMP (no allocations and then shrinked)
        CGPUMeshPackerV2(video::IVideoDriver* driver, const asset::CCPUMeshPackerV2<MDIStructType>* cpuMP)
            :IMeshPackerV2<video::IGPUBuffer,video::IGPUMeshBuffer,MDIStructType>(cpuMP->m_allocParams,cpuMP->m_minTriangleCountPerMDIData,cpuMP->m_maxTriangleCountPerMDIData),
             m_driver(driver)
        {
            m_virtualAttribConfig = cpuMP->m_virtualAttribConfig;

            auto& cpuMDIBuff = cpuMP->m_packerDataStore.MDIDataBuffer;
            auto& cpuIdxBuff = cpuMP->m_packerDataStore.indexBuffer;
            auto& cpuVtxBuff = cpuMP->m_packerDataStore.vertexBuffer;

            // TODO: why are the allocators not copied!?

            // TODO: call instantiateDataStorage() here and then copy CPU data to the initialized storage
            m_packerDataStore.MDIDataBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(cpuMDIBuff->getSize(), cpuMDIBuff->getPointer());
            m_packerDataStore.indexBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(cpuMDIBuff->getSize(), cpuMDIBuff->getPointer());
            m_packerDataStore.vertexBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(cpuMDIBuff->getSize(), cpuMDIBuff->getPointer());
        }

        void instantiateDataStorage();

        template <typename MeshBufferIterator>
        bool commit(IMeshPackerBase::PackedMeshBufferData* pmbdOut, ReservedAllocationMeshBuffers* rambIn, const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd);

        uint32_t getDSlayoutBindingsForUTB(IGPUDescriptorSetLayout::SBinding* outBindings, uint32_t fsamplersBinding = 0u, uint32_t isamplersBinding = 1u, uint32_t usamplersBinding = 2u) const
        {
            return getDSlayoutBindingsForUTB_internal<IGPUDescriptorSetLayout>(outBindings, fsamplersBinding, isamplersBinding, usamplersBinding);
        }

        std::pair<uint32_t, uint32_t> getDescriptorSetWritesForUTB(IGPUDescriptorSet::SWriteDescriptorSet* outWrites, IGPUDescriptorSet::SDescriptorInfo* outInfo, IGPUDescriptorSet* dstSet, uint32_t fBuffersBinding = 0u, uint32_t iBuffersBinding = 1u, uint32_t uBuffersBinding = 2u) const
        {
            auto createBufferView = [&](E_FORMAT format) -> core::smart_refctd_ptr<IGPUBufferView>
            {
                return m_driver->createGPUBufferView(m_packerDataStore.vertexBuffer.get(), format);
            };

            return getDescriptorSetWritesForUTB_internal<IGPUDescriptorSet, IGPUBufferView>(outWrites, outInfo, dstSet, createBufferView, fBuffersBinding, iBuffersBinding, uBuffersBinding);
        }

        uint32_t getDSlayoutBindingsForSSBO(IGPUDescriptorSetLayout::SBinding* outBindings, uint32_t uintBufferBinding = 0u, uint32_t uvec2BufferBinding = 1u, uint32_t uvec3BufferBinding = 2u, uint32_t uvec4BufferBinding = 3u) const
        {
            return getDSlayoutBindingsForSSBO_internal<IGPUDescriptorSetLayout>(outBindings, uintBufferBinding, uvec2BufferBinding, uvec3BufferBinding, uvec4BufferBinding);
        }

        uint32_t getDescriptorSetWritesForSSBO(IGPUDescriptorSet::SWriteDescriptorSet* outWrites, IGPUDescriptorSet::SDescriptorInfo* outInfo, IGPUDescriptorSet* dstSet, uint32_t uintBufferBinding = 0u, uint32_t uvec2BufferBinding = 1u, uint32_t uvec3BufferBinding = 2u, uint32_t uvec4BufferBinding = 3u) const
        {
            return getDescriptorSetWritesForSSBO_internal<IGPUDescriptorSet>(outWrites, outInfo, dstSet, m_packerDataStore.vertexBuffer, uintBufferBinding, uvec2BufferBinding, uvec3BufferBinding, uvec4BufferBinding);
        }

    private:
        video::IVideoDriver* m_driver;

};

template <typename MDIStructType>
void CGPUMeshPackerV2<MDIStructType>::instantiateDataStorage()
{
    m_packerDataStore.MDIDataBuffer = m_driver->createDeviceLocalGPUBufferOnDedMem(m_allocParams.MDIDataBuffSupportedCnt * sizeof(MDIStructType));
    m_packerDataStore.indexBuffer = m_driver->createDeviceLocalGPUBufferOnDedMem(m_allocParams.indexBuffSupportedCnt * sizeof(uint16_t));
    m_packerDataStore.vertexBuffer = m_driver->createDeviceLocalGPUBufferOnDedMem(m_allocParams.vertexBuffSupportedByteSize);
}

template <typename MDIStructType>
template <typename MeshBufferIterator>
bool CGPUMeshPackerV2<MDIStructType>::commit(IMeshPackerBase::PackedMeshBufferData* pmbdOut, ReservedAllocationMeshBuffers* rambIn, const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd)
{
    assert(0);
    return false;
}

}
}

#endif