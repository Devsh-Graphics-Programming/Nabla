// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/CUDAInterop/CUDAInterop.h"

#ifdef _NBL_COMPILE_WITH_CUDA_
#include "CUDAInteropNativeState.hpp"

namespace nbl::video
{

CCUDAExportableMemory::CCUDAExportableMemory(core::smart_refctd_ptr<CCUDADevice> device, SCachedCreationParams&& params, std::unique_ptr<SNativeState>&& nativeState)
	: m_device(std::move(device))
	, m_params(std::move(params))
	, m_native(std::move(nativeState))
{}

core::smart_refctd_ptr<IDeviceMemoryAllocation> CCUDAExportableMemory::exportAsMemory(ILogicalDevice* device, IDeviceMemoryBacked* dedication) const
{
	auto pd = device->getPhysicalDevice();
	uint32_t memoryTypeBits = (1 << pd->getMemoryProperties().memoryTypeCount) - 1;
	uint32_t vram = pd->getDeviceLocalMemoryTypeBits();

	if (m_params.deviceLocal)
		memoryTypeBits &= vram;
	else
		memoryTypeBits &= ~vram;

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
	const auto& cu = cuda_native::getCUDAFunctionTable(*m_device->getHandler());

  ASSERT_CUDA_SUCCESS(cu.pcuMemUnmap(m_native->ptr, m_params.granularSize), m_device->getHandler());

	ASSERT_CUDA_SUCCESS(cu.pcuMemAddressFree(m_native->ptr, m_params.granularSize), m_device->getHandler());

  bool closeSucceed = CloseExternalHandle(m_params.externalHandle);
	assert(closeSucceed);

}

namespace cuda_native
{

CUdeviceptr getDeviceptr(const CCUDAExportableMemory& memory)
{
	return SAccess::native(memory).ptr;
}

}
}

#else

namespace nbl::video
{

// CUDA OFF stub keeps the clean public API linkable and reports feature absence with nullptr instead of unresolved symbols.
struct CCUDAExportableMemory::SNativeState {};

CCUDAExportableMemory::CCUDAExportableMemory(core::smart_refctd_ptr<CCUDADevice> device, SCachedCreationParams&& params, std::unique_ptr<SNativeState>&& nativeState)
	: m_device(std::move(device))
	, m_params(std::move(params))
	, m_native(std::move(nativeState))
{}

CCUDAExportableMemory::~CCUDAExportableMemory() = default;

core::smart_refctd_ptr<IDeviceMemoryAllocation> CCUDAExportableMemory::exportAsMemory(ILogicalDevice*, IDeviceMemoryBacked*) const
{
	return nullptr;
}

}

#endif // _NBL_COMPILE_WITH_CUDA_
