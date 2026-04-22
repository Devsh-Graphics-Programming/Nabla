// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/video/CCUDAImportedMemory.h"
#include "nbl/video/CCUDADevice.h"

#ifdef _NBL_COMPILE_WITH_CUDA_

namespace nbl::video
{

CUresult CCUDAImportedMemory::getMappedBuffer(CUdeviceptr* mappedBuffer)
{
  CUDA_EXTERNAL_MEMORY_BUFFER_DESC bufferDesc = {};
  bufferDesc.offset = 0;
  bufferDesc.size = m_src->getAllocationSize();

  auto& cu = m_device->getHandler()->getCUDAFunctionTable();
  return cu.pcuExternalMemoryGetMappedBuffer(mappedBuffer, m_handle, &bufferDesc);
  
}

CCUDAImportedMemory::~CCUDAImportedMemory()
{
  auto& cu = m_device->getHandler()->getCUDAFunctionTable();
  if (cu.pcuDestroyExternalMemory(m_handle) != CUDA_SUCCESS)
    assert(!"Invalid code path");
}

}

#endif