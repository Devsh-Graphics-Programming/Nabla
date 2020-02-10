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


class CCUDAHandler
{
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
			,cuDeviceGetCount
			,cuDeviceGet
			,cuDeviceGetAttribute
			,cuDeviceGetUuid
			,cuDeviceTotalMem_v2
			,cuDeviceGetName
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

		struct Device
		{
			Device() {}
			Device(int ordinal) : Device()
			{
				CUresult result;

				if (cuda.pcuDeviceGet(&handle, ordinal)!=CUDA_SUCCESS)
				{
					handle = 0;
					return;
				}

				if (cuda.pcuDeviceGetName(name,sizeof(name),handle)!=CUDA_SUCCESS)
					return;

				if (cuda.pcuDeviceGetUuid(&uuid,handle)!=CUDA_SUCCESS)
					return;

				if (cuda.pcuDeviceTotalMem_v2(&vram_size,handle)!=CUDA_SUCCESS)
					return;

				for (int i=0; i<CU_DEVICE_ATTRIBUTE_MAX; i++)
					cuda.pcuDeviceGetAttribute(attributes+i,static_cast<CUdevice_attribute>(i),handle);
			}
			~Device()
			{
			}

			CUdevice handle = 0;
			char name[124] = {};
			CUuuid uuid = {};
			size_t vram_size = 0ull;
			int attributes[CU_DEVICE_ATTRIBUTE_MAX] = {};
		};

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

	protected:
        CCUDAHandler() = default;

		_IRR_STATIC_INLINE int CudaVersion = 0;
		_IRR_STATIC_INLINE int DeviceCount = 0;
		static core::vector<Device> devices;

	public:
		static CUresult init()
		{
			CUresult result;


			cuda = CUDA(
				#if defined(_IRR_WINDOWS_API_)
					"nvcuda"
				#elif defined(_IRR_POSIX_API_)
					"cuda"
				#endif
			);
			#define SAFE_CUDA_CALL(NO_PTR_ERROR,FUNC,...) \
			{\
				if (!cuda.p ## FUNC)\
					return NO_PTR_ERROR;\
				result = cuda.p ## FUNC ## (__VA_ARGS__);\
				if (result!=CUDA_SUCCESS)\
					return result;\
			}
			
			SAFE_CUDA_CALL(CUDA_ERROR_NOT_SUPPORTED,cuInit,0)
				
			SAFE_CUDA_CALL(CUDA_ERROR_NOT_SUPPORTED,cuDriverGetVersion,&CudaVersion)
			if (CudaVersion<9000)
				return CUDA_ERROR_SYSTEM_DRIVER_MISMATCH;
			
			SAFE_CUDA_CALL(CUDA_ERROR_NOT_SUPPORTED,cuDeviceGetCount,&DeviceCount)

			devices.resize(DeviceCount);
			for (int i=0; i<DeviceCount; i++)
				devices[i] = Device(i);
			
			#undef SAFE_CUDA_CALL

			nvrtc = NVRTC("nvrtc");


			return CUDA_SUCCESS;
		}
		static void deinit()
		{
			devices.resize(0u);

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
