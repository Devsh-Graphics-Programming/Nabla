// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_GPU_MESH_PACKER_V2_H_INCLUDED__
#define __NBL_ASSET_C_GPU_MESH_PACKER_V2_H_INCLUDED__

#include <nbl/video/IGPUMesh.h>
#include <nbl/asset/IMeshPackerV2.h>

namespace nbl
{
namespace asset
{

template <typename MDIStructType = DrawElementsIndirectCommand_t>
class CGPUMeshPackerV2 final : public IMeshPackerV2<video::IGPUBuffer, video::IGPUMeshBuffer, MDIStructType>
{
    using base_t = IMeshPackerV2<video::IGPUBuffer, video::IGPUMeshBuffer, MDIStructType>;
    using Triangle = typename base_t::Triangle;
    using TriangleBatch = typename base_t::TriangleBatch;

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

    void instantiateDataStorage();

    template <typename MeshBufferIterator>
    bool commit(IMeshPackerBase::PackedMeshBufferData* pmbdOut, ReservedAllocationMeshBuffers* rambIn, const MeshBufferIterator mbBegin, const MeshBufferIterator mbEnd);

private:
    video::IVideoDriver* m_driver;

};

template <typename MDIStructType>
void CGPUMeshPackerV2<MDIStructType>::instantiateDataStorage()
{
    m_packerDataStore.MDIDataBuffer = m_driver->createDeviceLocalGPUBufferOnDedMem(m_allocParams.MDIDataBuffSupportedCnt * sizeof(MDIStructType));
    m_packerDataStore.indexBuffer = m_driver->createDeviceLocalGPUBufferOnDedMem(m_allocParams.indexBuffSupportedCnt * sizeof(uint16_t));
    m_packerDataStore.vertexBuffer = m_driver->createDeviceLocalGPUBufferOnDedMem(m_allocParams.vertexBuffSupportedSize);
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