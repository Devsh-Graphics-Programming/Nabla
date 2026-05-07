// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/CUDAInterop/CUDAInterop.h"

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

namespace cuda_native
{

CUexternalMemory getInternalObject(const CCUDAImportedMemory& memory)
{
  return SAccess::native(memory).handle;
}

CUresult getMappedBuffer(const CCUDAImportedMemory& memory, CUdeviceptr* mappedBuffer)
{
  CUDA_EXTERNAL_MEMORY_BUFFER_DESC bufferDesc = {};
  bufferDesc.offset = 0;
  bufferDesc.size = SAccess::source(memory)->getAllocationSize();

  const auto& cu = getCUDAFunctionTable(*SAccess::device(memory)->getHandler());
  return cu.pcuExternalMemoryGetMappedBuffer(mappedBuffer, SAccess::native(memory).handle, &bufferDesc);
  
}

}

CCUDAImportedMemory::~CCUDAImportedMemory()
{
  auto& cu = cuda_native::getCUDAFunctionTable(*m_device->getHandler());
  ASSERT_CUDA_SUCCESS(cu.pcuDestroyExternalMemory(m_native->handle), m_device->getHandler());
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

}

#endif
