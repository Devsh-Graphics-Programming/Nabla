#ifndef _NBL_EXT_CUDA_INTEROP_C_CUDA_IMPORTED_MEMORY_H_
#define _NBL_EXT_CUDA_INTEROP_C_CUDA_IMPORTED_MEMORY_H_

#ifdef _NBL_COMPILE_WITH_CUDA_

#include "nbl/video/declarations.h"

#include "cuda.h"
#include "nvrtc.h"
#if CUDA_VERSION < 9000
  #error "Need CUDA 9.0 SDK or higher."
#endif

namespace nbl::video
{

class CCUDADevice;

class CCUDAImportedMemory : public core::IReferenceCounted
{
    public:

      CCUDAImportedMemory(core::smart_refctd_ptr<CCUDADevice> device, core::smart_refctd_ptr<nbl::video::IDeviceMemoryAllocation> src,
        CUexternalMemory cuExtMem) : 
        m_device(device),
        m_src(src),
        m_handle(cuExtMem) {}

      ~CCUDAImportedMemory() override;

      CUexternalMemory getInternalObject() const { return m_handle; }
      CUresult getMappedBuffer(CUdeviceptr* mappedBuffer);

    private:

      core::smart_refctd_ptr<CCUDADevice> m_device;
      core::smart_refctd_ptr<IDeviceMemoryAllocation> m_src;
      CUexternalMemory m_handle;

};

}

#endif // _NBL_COMPILE_WITH_CUDA_

#endif
