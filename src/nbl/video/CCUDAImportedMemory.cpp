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
	return cuda_interop::SNativeHandle<cuda_interop::SCUexternalMemory,CUexternalMemory>(m_native->handle);
}

bool CCUDAImportedMemory::getMappedBuffer(cuda_interop::SCUdeviceptr* mappedBuffer) const
{
	if (!mappedBuffer)
		return false;

	CUDA_EXTERNAL_MEMORY_BUFFER_DESC bufferDesc = {};
	bufferDesc.offset = 0;
	bufferDesc.size = m_src->getAllocationSize();

	CUdeviceptr nativeMappedBuffer = 0;
	const auto& cu = m_device->getHandler()->getCUDAFunctionTable();
	const auto result = cu.pcuExternalMemoryGetMappedBuffer(&nativeMappedBuffer, m_native->handle, &bufferDesc);
	if (!cuda_native::defaultHandleResult(*m_device->getHandler(),result))
		return false;

	*mappedBuffer = cuda_interop::SNativeHandle<cuda_interop::SCUdeviceptr,CUdeviceptr>(nativeMappedBuffer);
	return true;
}

CCUDAImportedMemory::~CCUDAImportedMemory()
{
	auto& cu = m_device->getHandler()->getCUDAFunctionTable();
	cuda_native::defaultHandleResult(*m_device->getHandler(), cu.pcuDestroyExternalMemory(m_native->handle));
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
	assert(m_native);
}

CCUDAImportedMemory::~CCUDAImportedMemory() = default;

cuda_interop::SCUexternalMemory CCUDAImportedMemory::getInternalObject() const
{
	return {};
}

bool CCUDAImportedMemory::getMappedBuffer(cuda_interop::SCUdeviceptr*) const
{
	return false;
}

}

#endif
