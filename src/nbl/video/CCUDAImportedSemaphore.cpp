// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/video/CCUDAImportedSemaphore.h"
#include "nbl/video/CCUDADevice.h"

#ifdef _NBL_COMPILE_WITH_CUDA_
namespace nbl::video
{
CCUDAImportedSemaphore::~CCUDAImportedSemaphore()
{
	auto& cu = m_device->getHandler()->getCUDAFunctionTable();
	if (cu.pcuDestroyExternalSemaphore(m_handle) != CUDA_SUCCESS)
    assert(!"Invalid code path.");
}
}

#endif // _NBL_COMPILE_WITH_CUDA_