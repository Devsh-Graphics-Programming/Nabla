// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "CUDAInteropNativeState.hpp"

#ifdef _NBL_COMPILE_WITH_CUDA_

namespace nbl::video
{

CCUDAImportedMemory::CCUDAImportedMemory(core::smart_refctd_ptr<CCUDADevice> device, core::smart_refctd_ptr<nbl::video::IDeviceMemoryAllocation> src, std::unique_ptr<SNativeState>&& nativeState)
	: m_device(std::move(device))
	, m_src(std::move(src))
	, m_native(std::move(nativeState))
{}

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

#endif
