// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/video/CCUDADevice.h"

#ifdef _NBL_COMPILE_WITH_CUDA_
namespace nbl::video
{

core::smart_refctd_ptr<IDeviceMemoryAllocation> CCUDASharedMemory::exportAsMemory(ILogicalDevice* device, IDeviceMemoryBacked* dedication) const
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
		CCUDADevice::EXTERNAL_MEMORY_HANDLE_TYPE, m_params.osHandle, 
		std::make_unique<CCUDADevice::SCUDACleaner>(core::smart_refctd_ptr<const CCUDASharedMemory>(this))).memory;
}

core::smart_refctd_ptr<IGPUImage>  CCUDASharedMemory::createAndBindImage(ILogicalDevice* device, asset::IImage::SCreationParams&& params) const
{
	if (!device || !m_device->isMatchingDevice(device->getPhysicalDevice()))
		return nullptr;

	auto img = device->createImage({ std::move(params), { {.externalHandleTypes = CCUDADevice::EXTERNAL_MEMORY_HANDLE_TYPE } }, IGPUImage::TILING::LINEAR });
	
	if (exportAsMemory(device, img.get()))
		return img;
	
	return nullptr;
}

CCUDASharedMemory::~CCUDASharedMemory()
{
	auto& cu = m_device->getHandler()->getCUDAFunctionTable();

	CUresult re[] = {
		cu.pcuMemUnmap(m_params.ptr, m_params.granularSize),
	};
	CloseHandle(m_params.osHandle);

}
}

#endif // _NBL_COMPILE_WITH_CUDA_