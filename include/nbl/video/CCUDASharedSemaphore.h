// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_C_CUDA_SHARED_SEMAPHORE_H_
#define _NBL_VIDEO_C_CUDA_SHARED_SEMAPHORE_H_

#include "nbl/video/CCUDADevice.h"

#ifdef _NBL_COMPILE_WITH_CUDA_

#include "cuda.h"
#include "nvrtc.h"
#if CUDA_VERSION < 9000
	#error "Need CUDA 9.0 SDK or higher."
#endif

// useful includes in the future
//#include "cudaEGL.h"
//#include "cudaVDPAU.h"

namespace nbl::video
{

class CCUDASharedSemaphore : public core::IReferenceCounted
{
public:
    friend class CCUDADevice;

    CUexternalSemaphore getInternalObject() const { return m_handle; }

protected:
   
    CCUDASharedSemaphore(core::smart_refctd_ptr<CCUDADevice> device, core::smart_refctd_ptr<IGPUSemaphore> src, CUexternalSemaphore semaphore, void* osHandle)
        : m_device(std::move(device))
        , m_src(std::move(m_src))
        , m_handle(semaphore)
        , m_osHandle(osHandle)
    {}
    ~CCUDASharedSemaphore() override;

    core::smart_refctd_ptr<CCUDADevice> m_device;
    core::smart_refctd_ptr<IGPUSemaphore> m_src;
    CUexternalSemaphore m_handle;
    void* m_osHandle;
};

}

#endif // _NBL_COMPILE_WITH_CUDA_

#endif
