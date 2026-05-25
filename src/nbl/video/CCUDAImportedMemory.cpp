// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/video/CUDAInterop.h"

#ifdef _NBL_COMPILE_WITH_CUDA_
#include "CUDAInteropNativeState.hpp"

namespace nbl::video
{

CCUDAImportedMemory::CCUDAImportedMemory(core::smart_refctd_ptr<CCUDADevice> device, core::smart_refctd_ptr<nbl::video::IDeviceMemoryAllocation> src, std::unique_ptr<SNativeState>&& nativeState)
	: m_device(std::move(device))
	, m_src(std::move(src))
	, m_native(std::move(nativeState))
{
	assert(m_native);
}

cuda_interop::SCUexternalMemory CCUDAImportedMemory::getInternalObject() const
{
	return m_native->handle;
}

bool CCUDAImportedMemory::getMappedBuffer(cuda_interop::SOutput<cuda_interop::SCUdeviceptr> mappedBuffer, size_t size, size_t offset) const
{
	if (!mappedBuffer)
		return false;

	const auto allocationSize = m_src->getAllocationSize();

	if (offset > allocationSize)
	{
		m_device->getHandler()->getLogger().log("Offset must not be more than allocation size!", system::ILogger::ELL_ERROR);
		return false;
	}


	CUDA_EXTERNAL_MEMORY_BUFFER_DESC bufferDesc = {};
	bufferDesc.offset = offset;
	bufferDesc.size = size == WholeSize ? (allocationSize - offset) : size;

	CUdeviceptr nativeMappedBuffer = 0;
	const auto& handler = m_device->getHandler();
	const auto& cu = handler->getCUDAFunctionTable();
	if (!handler->defaultHandleResult(cu.pcuExternalMemoryGetMappedBuffer(&nativeMappedBuffer, m_native->handle, &bufferDesc), "Fail to get mapped buffer!"))
		return false;

	*mappedBuffer = nativeMappedBuffer;
	return true;
}

CCUDAImportedMemory::~CCUDAImportedMemory()
{
	const auto& handler = m_device->getHandler();
	auto& cu = handler->getCUDAFunctionTable();
	handler->defaultHandleResult(cu.pcuDestroyExternalMemory(m_native->handle), "Fail to destroy external memory!");
}

}

#else

namespace nbl::video
{

// CUDA OFF stub keeps the clean public API linkable and reports feature absence with nullptr instead of unresolved symbols.
struct CCUDAImportedMemory::SNativeState {};

CCUDAImportedMemory::CCUDAImportedMemory(core::smart_refctd_ptr<CCUDADevice> device, core::smart_refctd_ptr<IDeviceMemoryAllocation> src, std::unique_ptr<SNativeState>&& nativeState)
	: m_device(std::move(device))
	, m_src(std::move(src))
	, m_native(std::move(nativeState))
{
	assert(false);
}

CCUDAImportedMemory::~CCUDAImportedMemory() = default;

cuda_interop::SCUexternalMemory CCUDAImportedMemory::getInternalObject() const
{
	return {};
}

bool CCUDAImportedMemory::getMappedBuffer(cuda_interop::SOutput<cuda_interop::SCUdeviceptr>) const
{
	return false;
}

}

#endif
