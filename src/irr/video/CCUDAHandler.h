#ifndef __C_CUDA_HANDLER_H__
#define __C_CUDA_HANDLER_H__

#include "irr/macros.h"
#include "irr/system/system.h"


#ifdef _IRR_COMPILE_WITH_CUDA_

#include "cuda.h"
#include "nvrtc.h"
#if CUDA_VERSION < 9000
	#error "Need CUDA 9.0 SDK or higher."
#endif

#ifdef _IRR_COMPILE_WITH_OPENGL_
	#include "cudaGL.h"
    #include "COpenGLExtensionHandler.h"
#endif // _IRR_COMPILE_WITH_OPENGL_

// useful includes in the future
//#include "cudaEGL.h"
//#include "cudaVDPAU.h"

#include "os.h"

namespace irr
{
namespace cuda
{

#if 0
static const char* const OpenCLFeatureStrings[] = {
    "cl_khr_gl_sharing",
	"cl_khr_gl_event"
};


#define IRR_MAX_OCL_PLATFORMS 5
#define IRR_MAX_OCL_DEVICES 8
#endif




class CCUDAHandler
{
        CCUDAHandler() = delete;
    public:
		using LibLoader = system::DefaultFuncPtrLoader;

		IRR_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(CUDA, LibLoader
			,cuCtxCreate_v2
			,cuDevicePrimaryCtxRetain
			,cuDevicePrimaryCtxRelease
			,cuDevicePrimaryCtxSetFlags
			,cuDevicePrimaryCtxGetState
			,cuCtxDestroy_v2
			,cuCtxEnablePeerAccess
			,cuCtxGetCurrent
			,cuCtxGetDevice
			,cuCtxGetSharedMemConfig
			,cuCtxPopCurrent_v2
			,cuCtxSetCurrent
			,cuCtxSetSharedMemConfig
			,cuCtxSynchronize
			,cuDeviceComputeCapability
			,cuDeviceCanAccessPeer
			,cuDeviceGet
			,cuDeviceGetAttribute
			,cuDeviceGetCount
			,cuDeviceGetName
			,cuDeviceGetPCIBusId
			,cuDeviceGetProperties
			,cuDeviceTotalMem_v2
			,cuDriverGetVersion
			,cuEventCreate
			,cuEventDestroy_v2
			,cuEventElapsedTime
			,cuEventQuery
			,cuEventRecord
			,cuEventSynchronize
			,cuFuncGetAttribute
			,cuFuncSetCacheConfig
			,cuGetErrorName
			,cuGetErrorString
			,cuInit
			,cuLaunchKernel
			,cuMemAlloc_v2
			,cuMemcpyDtoD_v2
			,cuMemcpyDtoH_v2
			,cuMemcpyHtoD_v2
			,cuMemcpyDtoDAsync_v2
			,cuMemcpyDtoHAsync_v2
			,cuMemcpyHtoDAsync_v2
			,cuMemGetAddressRange_v2
			,cuMemFree_v2
			,cuMemFreeHost
			,cuMemGetInfo_v2
			,cuMemHostAlloc
			,cuMemHostRegister_v2
			,cuMemHostUnregister
			,cuMemsetD32_v2
			,cuMemsetD32Async
			,cuMemsetD8_v2
			,cuMemsetD8Async
			,cuModuleGetFunction
			,cuModuleGetGlobal_v2
			,cuModuleLoadDataEx
			,cuModuleLoadFatBinary
			,cuModuleUnload
			,cuOccupancyMaxActiveBlocksPerMultiprocessor
			,cuPointerGetAttribute
			,cuStreamAddCallback
			,cuStreamCreate
			,cuStreamDestroy_v2
			,cuStreamQuery
			,cuStreamSynchronize
			,cuStreamWaitEvent 
		);
		static CUDA cuda;
		IRR_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(NVRTC, LibLoader,
			nvrtcGetErrorString,
			nvrtcVersion,
			nvrtcAddNameExpression,
			nvrtcCompileProgram,
			nvrtcCreateProgram,
			nvrtcDestroyProgram,
			nvrtcGetLoweredName,
			nvrtcGetPTX,
			nvrtcGetPTXSize,
			nvrtcGetProgramLog,
			nvrtcGetProgramLogSize
		);
		static NVRTC nvrtc;

		static void init()
		{
			cuda = CUDA("cuda");
			nvrtc = NVRTC("nvrtc");
		}
		static void deinit()
		{
			cuda = CUDA();
			nvrtc = NVRTC();
		}

		// note: enable `-arch=latest` `-use-fast-math`
		// note: allow to control `-G` `-lineinfo` `-maxrregcount` `-rdc` `-ewp` `-restrict` then includes and defines
};

}
}

#endif // _IRR_COMPILE_WITH_CUDA_

#endif
