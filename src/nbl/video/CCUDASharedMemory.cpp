// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/video/CCUDADevice.h"

#ifdef _NBL_COMPILE_WITH_CUDA_
namespace nbl::video
{

core::smart_refctd_ptr<IGPUBuffer> CCUDASharedMemory::exportAsBuffer(ILogicalDevice* device, core::bitflag<asset::IBuffer::E_USAGE_FLAGS> usage) const
{
	if (!device || !m_device->isMatchingDevice(device->getPhysicalDevice()))
		return nullptr;

	auto buf = device->createBuffer({{
			.size = m_params.size,
			.usage = usage }, {{
			.postDestroyCleanup = std::make_unique<CCUDADevice::SCUDACleaner>(core::smart_refctd_ptr<const CCUDASharedMemory>(this)),
			.externalHandleTypes = CCUDADevice::EXTERNAL_MEMORY_HANDLE_TYPE,
			.externalHandle = m_params.osHandle
		}}});

	auto req = buf->getMemoryReqs();
	auto pd = device->getPhysicalDevice();
	switch (m_params.location)
	{
	case CU_MEM_LOCATION_TYPE_DEVICE: req.memoryTypeBits &= pd->getDeviceLocalMemoryTypeBits(); break;
	case CU_MEM_LOCATION_TYPE_HOST: req.memoryTypeBits &= pd->getHostVisibleMemoryTypeBits(); break;
	// TODO(Atil): Figure out how to handle these
	case CU_MEM_LOCATION_TYPE_HOST_NUMA: 
	case CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT: 
	default: break;
	}

	if (!device->allocate(req, buf.get()).isValid())
		return nullptr;

	return buf;
}

core::smart_refctd_ptr<IGPUImage>  CCUDASharedMemory::exportAsImage(ILogicalDevice* device, asset::IImage::SCreationParams&& params) const
{
	if (!device || !m_device->isMatchingDevice(device->getPhysicalDevice()))
		return nullptr;

	auto img = device->createImage({
		std::move(params),
		{{
			.postDestroyCleanup = std::make_unique<CCUDADevice::SCUDACleaner>(core::smart_refctd_ptr<const CCUDASharedMemory>(this)),
			.externalHandleTypes = CCUDADevice::EXTERNAL_MEMORY_HANDLE_TYPE,
			.externalHandle = m_params.osHandle
		}},
		IGPUImage::ET_OPTIMAL,
		IGPUImage::EL_PREINITIALIZED,
	});

	auto req = img->getMemoryReqs();
	auto pd = device->getPhysicalDevice();
	switch (m_params.location)
	{
	case CU_MEM_LOCATION_TYPE_DEVICE: req.memoryTypeBits &= pd->getDeviceLocalMemoryTypeBits(); break;
	case CU_MEM_LOCATION_TYPE_HOST: req.memoryTypeBits &= pd->getHostVisibleMemoryTypeBits(); break;
		// TODO(Atil): Figure out how to handle these
	case CU_MEM_LOCATION_TYPE_HOST_NUMA:
	case CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT:
	default: break;
	}

	if (!device->allocate(req, img.get()).isValid())
		return nullptr;

	return img;
}

CCUDASharedMemory::~CCUDASharedMemory()
{
	auto& cu = m_device->getHandler()->getCUDAFunctionTable();
	auto& params = m_params;
	cu.pcuMemUnmap(params.ptr, params.size);
	cu.pcuMemAddressFree(params.ptr, params.size);
	cu.pcuMemRelease(params.mem);
	CloseHandle(params.osHandle);
}
}

#endif // _NBL_COMPILE_WITH_CUDA_
