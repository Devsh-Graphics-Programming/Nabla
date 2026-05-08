// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_C_CUDA_IMPORTED_SEMAPHORE_H_
#define _NBL_VIDEO_C_CUDA_IMPORTED_SEMAPHORE_H_

#include "nbl/video/declarations.h"
#include "nbl/video/CUDAInteropHandles.h"

#include <memory>
#include <utility>

namespace nbl::video
{

class CCUDADevice;

namespace cuda_native
{
struct SAccess;
}

class NBL_API2 CCUDAImportedSemaphore : public core::IReferenceCounted
{
	public:
		~CCUDAImportedSemaphore() override;
		cuda_interop::SCUexternalSemaphore getInternalObject() const;

	private:
		friend class CCUDADevice;
		friend struct cuda_native::SAccess;

		struct SNativeState;
		CCUDAImportedSemaphore(core::smart_refctd_ptr<CCUDADevice> device, core::smart_refctd_ptr<ISemaphore> src, std::unique_ptr<SNativeState>&& nativeState);

		core::smart_refctd_ptr<CCUDADevice> m_device;
		core::smart_refctd_ptr<ISemaphore> m_src;
		std::unique_ptr<SNativeState> m_native;
};

}

#endif
