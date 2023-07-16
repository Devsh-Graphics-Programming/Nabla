// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/video/CCUDASharedSemaphore.h"

#ifdef _NBL_COMPILE_WITH_CUDA_
namespace nbl::video
{
CCUDASharedSemaphore::~CCUDASharedSemaphore()
{
	m_device->destroyExternalSemaphore(this);
}
}

#endif // _NBL_COMPILE_WITH_CUDA_
