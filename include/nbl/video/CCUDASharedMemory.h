// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_C_CUDA_SHARED_MEMORY_H_
#define _NBL_VIDEO_C_CUDA_SHARED_MEMORY_H_


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
    CUdeviceptr getDevicePtr() const { return m_params.ptr; }

    struct SCreationParams
    {
        size_t            size;
        uint32_t          alignment;
        CUmemLocationType location;
    };

    struct SCachedCreationParams : SCreationParams
    {
        size_t granularSize;
        CUmemGenericAllocationHandle mem;
        CUdeviceptr ptr;
        union
        {
            void* osHandle;
            int fd;
        };
    };

    const SCreationParams& getCreationParams() const { return m_params; }

    core::smart_refctd_ptr<IGPUBuffer> exportAsBuffer(ILogicalDevice* device, core::bitflag<asset::IBuffer::E_USAGE_FLAGS> usage) const;
    core::smart_refctd_ptr<IGPUImage>  exportAsImage(ILogicalDevice* device, asset::IImage::SCreationParams&& params) const;

protected:

    CCUDASharedMemory(core::smart_refctd_ptr<CCUDADevice> device, SCachedCreationParams&& params)
        : m_device(std::move(device))
        , m_params(std::move(params))
    {}
    ~CCUDASharedMemory() override;

    core::smart_refctd_ptr<CCUDADevice> m_device;
    SCachedCreationParams m_params;
};

}

#endif // _NBL_COMPILE_WITH_CUDA_

#endif
