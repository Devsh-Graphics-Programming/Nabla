// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_C_CUDA_SHARED_MEMORY_H_
#define _NBL_VIDEO_C_CUDA_SHARED_MEMORY_H_

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

class CCUDASharedMemory : public core::IReferenceCounted
{
public:
    friend class CCUDADevice;
    CUdeviceptr getDevicePtr() const { return m_ptr; }
protected:

    CCUDASharedMemory(core::smart_refctd_ptr<CCUDADevice> device, size_t size, CUdeviceptr ptr, CUmemGenericAllocationHandle memory, void* osHandle)
        : m_device(std::move(device))
        , m_size(size)
        , m_ptr(ptr)
        , m_handle(memory)
        , m_osHandle(osHandle)
    {}
    ~CCUDASharedMemory() override;

    core::smart_refctd_ptr<CCUDADevice> m_device;
    size_t m_size;
    CUdeviceptr m_ptr;
    CUmemGenericAllocationHandle m_handle;
    void* m_osHandle;
};

}

#endif // _NBL_COMPILE_WITH_CUDA_

#endif
