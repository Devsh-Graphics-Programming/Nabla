// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/video/CCUDAExportableMemory.h"
#include "nbl/video/CCUDADevice.h"

#ifdef _NBL_COMPILE_WITH_CUDA_
namespace nbl::video
{

core::smart_refctd_ptr<IDeviceMemoryAllocation> CCUDAExportableMemory::exportAsMemory(ILogicalDevice* device, IDeviceMemoryBacked* dedication) const
{
	auto pd = device->getPhysicalDevice();
	uint32_t memoryTypeBits = (1 << pd->getMemoryProperties().memoryTypeCount) - 1;
	uint32_t vram = pd->getDeviceLocalMemoryTypeBits();

	switch (m_params.location)
	{
    case CU_MEM_LOCATION_TYPE_HOST:   memoryTypeBits &= ~vram; break;
    case CU_MEM_LOCATION_TYPE_DEVICE: memoryTypeBits &=  vram; break;
      // TODO(Atil): Figure out how to handle these
    case CU_MEM_LOCATION_TYPE_HOST_NUMA:
    case CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT:
    default: break;
	}

	IDeviceMemoryBacked::SDeviceMemoryRequirements req = {};
	req.size = m_params.granularSize;
	req.memoryTypeBits = memoryTypeBits;
	req.prefersDedicatedAllocation  = nullptr != dedication;
	req.requiresDedicatedAllocation = nullptr != dedication;

	return device->allocate(req, 
		dedication, 
		IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_NONE, 
		CCUDADevice::EXTERNAL_MEMORY_HANDLE_TYPE, 
		m_params.externalHandle).memory;
}

CCUDAExportableMemory::~CCUDAExportableMemory()
{
	auto& cu = m_device->getHandler()->getCUDAFunctionTable();

	cu.pcuMemUnmap(m_params.ptr, m_params.granularSize);
	cu.pcuMemAddressFree(m_params.ptr, m_params.granularSize);
	cu.pcuMemRelease(m_allocationHandle);

	CloseExternalHandle(m_params.externalHandle);

}
}

#endif // _NBL_COMPILE_WITH_CUDA_