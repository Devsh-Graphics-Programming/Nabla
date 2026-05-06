// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "CUDAInteropNativeState.hpp"

#ifdef _NBL_COMPILE_WITH_CUDA_
namespace nbl::video
{
CCUDAImportedSemaphore::CCUDAImportedSemaphore(core::smart_refctd_ptr<CCUDADevice> device, core::smart_refctd_ptr<ISemaphore> src, std::unique_ptr<SNativeState>&& nativeState)
	: m_device(std::move(device))
	, m_src(std::move(src))
	, m_native(std::move(nativeState))
{}

namespace cuda_native
{

CUexternalSemaphore getInternalObject(const CCUDAImportedSemaphore& semaphore)
{
	return SAccess::native(semaphore).handle;
}

}

CCUDAImportedSemaphore::~CCUDAImportedSemaphore()
{
	auto& cu = cuda_native::getCUDAFunctionTable(*m_device->getHandler());
	ASSERT_CUDA_SUCCESS(cu.pcuDestroyExternalSemaphore(m_native->handle), m_device->getHandler());
}
}

#endif // _NBL_COMPILE_WITH_CUDA_
