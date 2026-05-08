// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/video/CUDAInterop.h"

#ifdef _NBL_COMPILE_WITH_CUDA_
#include "CUDAInteropNativeState.hpp"

namespace nbl::video
{
CCUDAImportedSemaphore::CCUDAImportedSemaphore(core::smart_refctd_ptr<CCUDADevice> device, core::smart_refctd_ptr<ISemaphore> src, std::unique_ptr<SNativeState>&& nativeState)
	: m_device(std::move(device))
	, m_src(std::move(src))
	, m_native(std::move(nativeState))
{
	assert(m_native);
}

cuda_interop::SCUexternalSemaphore CCUDAImportedSemaphore::getInternalObject() const
{
	return cuda_native::SCUexternalSemaphore(m_native->handle).asOpaque();
}

CCUDAImportedSemaphore::~CCUDAImportedSemaphore()
{
	auto& cu = cuda_native::getCUDAFunctionTable(*m_device->getHandler());
	if (!cuda_native::defaultHandleResult(*m_device->getHandler(), cu.pcuDestroyExternalSemaphore(m_native->handle)))
		assert(false);
}
}

#else

namespace nbl::video
{

// CUDA OFF stub keeps the clean public API linkable and reports feature absence with nullptr instead of unresolved symbols.
struct CCUDAImportedSemaphore::SNativeState {};

CCUDAImportedSemaphore::CCUDAImportedSemaphore(core::smart_refctd_ptr<CCUDADevice> device, core::smart_refctd_ptr<ISemaphore> src, std::unique_ptr<SNativeState>&& nativeState)
	: m_device(std::move(device))
	, m_src(std::move(src))
	, m_native(std::move(nativeState))
{
	assert(m_native);
}

CCUDAImportedSemaphore::~CCUDAImportedSemaphore() = default;

cuda_interop::SCUexternalSemaphore CCUDAImportedSemaphore::getInternalObject() const
{
	return {};
}

}

#endif // _NBL_COMPILE_WITH_CUDA_
