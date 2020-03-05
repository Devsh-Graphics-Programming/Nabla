#ifndef __C_CUDA_HANDLER_H__
#define __C_CUDA_HANDLER_H__

#include "irr/macros.h"
#include "IReadFile.h"
#include "irr/system/system.h"


#ifdef _IRR_COMPILE_WITH_CUDA_

#include "cuda.h"
#include "nvrtc.h"
#if CUDA_VERSION < 9000
	#error "Need CUDA 9.0 SDK or higher."
#endif

#ifdef _IRR_COMPILE_WITH_OPENGL_
	#include "cudaGL.h"
    #include "../source/Irrlicht/COpenGLDriver.h"
    #include "../source/Irrlicht/COpenGLTexture.h"
#endif // _IRR_COMPILE_WITH_OPENGL_

// useful includes in the future
//#include "cudaEGL.h"
//#include "cudaVDPAU.h"

#include "os.h"

namespace irr
{
namespace cuda
{

#define _IRR_DEFAULT_NVRTC_OPTIONS "--std=c++14",virtualCUDAArchitecture,"-dc","-use_fast_math"


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
			,cuCtxGetApiVersion
			,cuCtxGetCurrent
			,cuCtxGetDevice
			,cuCtxGetSharedMemConfig
			,cuCtxPopCurrent_v2
			,cuCtxPushCurrent_v2
			,cuCtxSetCacheConfig
			,cuCtxSetCurrent
			,cuCtxSetSharedMemConfig
			,cuCtxSynchronize
			,cuDeviceComputeCapability
			,cuDeviceCanAccessPeer
			,cuDeviceGetCount
			,cuDeviceGet
			,cuDeviceGetAttribute
			,cuDeviceGetLuid
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
			,cuGraphicsGLRegisterBuffer
			,cuGraphicsGLRegisterImage
			,cuGraphicsMapResources
			,cuGraphicsResourceGetMappedPointer_v2
			,cuGraphicsResourceGetMappedMipmappedArray
			,cuGraphicsSubResourceGetMappedArray
			,cuGraphicsUnmapResources
			,cuGraphicsUnregisterResource
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
			,cuGLGetDevices_v2
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

				if (cuda.pcuDeviceGetLuid(&luid,&deviceNodeMask,handle))
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

			CUdevice handle = -1;
			char name[122] = {};
			char luid = -1;
			unsigned int deviceNodeMask = 0;
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
		/*
		IRR_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(NVRTC_BUILTINS, LibLoader
		);
		static NVRTC_BUILTINS nvrtc_builtins;
		*/
	protected:
        CCUDAHandler() = default;

		_IRR_STATIC_INLINE int CudaVersion = 0;
		_IRR_STATIC_INLINE int DeviceCount = 0;
		static core::vector<Device> devices;

		_IRR_STATIC_INLINE_CONSTEXPR const char* virtualCUDAArchitectures[] = {	"-arch=compute_30",
																				"-arch=compute_32",
																				"-arch=compute_35",
																				"-arch=compute_37",
																				"-arch=compute_50",
																				"-arch=compute_52",
																				"-arch=compute_53",
																				"-arch=compute_60",
																				"-arch=compute_61",
																				"-arch=compute_62",
																				"-arch=compute_70",
																				"-arch=compute_72",
																				"-arch=compute_75",
																				"-arch=compute_80"};
		_IRR_STATIC_INLINE const char* virtualCUDAArchitecture = nullptr;

	public:
		static CUresult init()
		{
			CUresult result = CUDA_ERROR_UNKNOWN;
			auto cleanup = core::makeRAIIExiter([&result]() -> void
				{
					if (result != CUDA_SUCCESS)
					{
						CudaVersion = 0;
						DeviceCount = 0;
						devices.clear();
					}
				}
			);


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
					return result=NO_PTR_ERROR;\
				result = cuda.p ## FUNC ## (__VA_ARGS__);\
				if (result!=CUDA_SUCCESS)\
					return result;\
			}
			
			SAFE_CUDA_CALL(CUDA_ERROR_NOT_SUPPORTED,cuInit,0)
				
			SAFE_CUDA_CALL(CUDA_ERROR_NOT_SUPPORTED,cuDriverGetVersion,&CudaVersion)
			if (CudaVersion<9000)
				return result=CUDA_ERROR_SYSTEM_DRIVER_MISMATCH;
			
			SAFE_CUDA_CALL(CUDA_ERROR_NOT_SUPPORTED,cuDeviceGetCount,&DeviceCount)

			devices.resize(DeviceCount);
			int j = 0;
			for (int i=0; i<DeviceCount; i++)
			{
				auto tmp = Device(i);
				
				const int archVersion[2] = {tmp.attributes[CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR],tmp.attributes[CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR]};
				if (archVersion[0]>8 || archVersion[0]==8&&archVersion[1]>0)
				{
					assert(strcmp(virtualCUDAArchitectures[13],"-arch=compute_80")==0);
					if (!virtualCUDAArchitecture)
						virtualCUDAArchitecture = virtualCUDAArchitectures[13];
				}
				else
				{
					const std::string virtualArchString = "-arch=compute_"+std::to_string(archVersion[0])+std::to_string(archVersion[1]);
				
					int32_t i = sizeof(virtualCUDAArchitectures)/sizeof(const char*);
					while (i>0)
					if (virtualCUDAArchitecture==virtualCUDAArchitectures[--i] || !virtualCUDAArchitecture)
						break;
					
					if (!virtualCUDAArchitecture || virtualArchString!=virtualCUDAArchitecture)
					{
						i++;
						while (i>0)
						if (virtualArchString==virtualCUDAArchitectures[--i])
						{
							virtualCUDAArchitecture = virtualCUDAArchitectures[i];
							break;
						}
					}
				}

				devices[j++] = tmp;
			}
			devices.resize(j);

			if (!virtualCUDAArchitecture)
				return result=CUDA_ERROR_INVALID_DEVICE;
			
			#undef SAFE_CUDA_CALL

			#if defined(_IRR_WINDOWS_API_)
			const char* nvrtc64_versions[] = {"nvrtc64_102","nvrtc64_101","nvrtc64_100","nvrtc64_92","nvrtc64_91","nvrtc64_90","nvrtc64_80","nvrtc64_75","nvrtc64_70",nullptr};
			const char* nvrtc64_suffices[] = {"","_","_0","_1","_2",nullptr};
			for (auto verpath=nvrtc64_versions; *verpath; verpath++)
			{
				for (auto suffix=nvrtc64_suffices; *suffix; suffix++)
				{
					std::string path(*verpath);
					path += *suffix;
					nvrtc = NVRTC(path.c_str());
					if (nvrtc.pnvrtcVersion)
						break;
				}
				if (nvrtc.pnvrtcVersion)
					break;
			}
			#elif defined(_IRR_POSIX_API_)
			nvrtc = NVRTC("nvrtc");
			//nvrtc_builtins = NVRTC("nvrtc-builtins");
			#endif

			int nvrtcVersion[2] = {-1,-1};
			cuda::CCUDAHandler::nvrtc.pnvrtcVersion(nvrtcVersion+0,nvrtcVersion+1);
			if (nvrtcVersion[0]<9)
				return result=CUDA_ERROR_SYSTEM_DRIVER_MISMATCH;

			return result=CUDA_SUCCESS;
		}
		static void deinit()
		{
			devices.resize(0u);

			cuda = CUDA();


			nvrtc = NVRTC();
		}

		static const char* getCommonVirtualCUDAArchitecture() {return virtualCUDAArchitecture;}

		static bool defaultHandleResult(CUresult result)
		{
			switch (result)
			{
				case CUDA_SUCCESS:
					return true;
					break;
				case CUDA_ERROR_INVALID_VALUE:
					printf(R"===(CCUDAHandler:
						This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.
					)===");
					break;
				case CUDA_ERROR_OUT_OF_MEMORY:
					printf(R"===(CCUDAHandler:
						The API call failed because it was unable to allocate enough memory to perform the requested operation.
					)===");
					break;
				case CUDA_ERROR_NOT_INITIALIZED:
					printf(R"===(CCUDAHandler:
						This indicates that the CUDA driver has not been initialized with cuInit() or that initialization has failed. 
					)===");
					break;
				case CUDA_ERROR_DEINITIALIZED:
					printf(R"===(CCUDAHandler:
						This indicates that the CUDA driver is in the process of shutting down.
					)===");
					break;
				case CUDA_ERROR_PROFILER_DISABLED:
					printf(R"===(CCUDAHandler:
						This indicates profiler is not initialized for this run. This can happen when the application is running with external profiling tools like visual profiler.
					)===");
					break;
				case CUDA_ERROR_NO_DEVICE:
					printf(R"===(CCUDAHandler:
						This indicates that no CUDA-capable devices were detected by the installed CUDA driver. 
					)===");
					break;
				case CUDA_ERROR_INVALID_DEVICE:
					printf(R"===(CCUDAHandler:
						This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device. 
					)===");
					break;
				case CUDA_ERROR_INVALID_IMAGE:
					printf(R"===(CCUDAHandler:
						This indicates that the device kernel image is invalid. This can also indicate an invalid CUDA module. 
					)===");
					break;
				case CUDA_ERROR_INVALID_CONTEXT:
					printf(R"===(CCUDAHandler:
						This most frequently indicates that there is no context bound to the current thread. This can also be returned if the context passed to an API call is not a valid handle (such as a context that has had cuCtxDestroy() invoked on it). This can also be returned if a user mixes different API versions (i.e. 3010 context with 3020 API calls). See cuCtxGetApiVersion() for more details.
					)===");
					break;
				case CUDA_ERROR_MAP_FAILED:
					printf(R"===(CCUDAHandler:
						This indicates that a map or register operation has failed.
					)===");
					break;
				case CUDA_ERROR_UNMAP_FAILED:
					printf(R"===(CCUDAHandler:
						This indicates that an unmap or unregister operation has failed.
					)===");
					break;
				case CUDA_ERROR_ARRAY_IS_MAPPED:
					printf(R"===(CCUDAHandler:
						This indicates that the specified array is currently mapped and thus cannot be destroyed.
					)===");
					break;
				case CUDA_ERROR_ALREADY_MAPPED:
					printf(R"===(CCUDAHandler:
						This indicates that the resource is already mapped.
					)===");
					break;
				case CUDA_ERROR_NO_BINARY_FOR_GPU:
					printf(R"===(CCUDAHandler:
						This indicates that there is no kernel image available that is suitable for the device. This can occur when a user specifies code generation options for a particular CUDA source file that do not include the corresponding device configuration.
					)===");
					break;
				case CUDA_ERROR_ALREADY_ACQUIRED:
					printf(R"===(CCUDAHandler:
						This indicates that a resource has already been acquired. 
					)===");
					break;
				case CUDA_ERROR_NOT_MAPPED:
					printf(R"===(CCUDAHandler:
						This indicates that a resource is not mapped.
					)===");
					break;
				case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
					printf(R"===(CCUDAHandler:
						This indicates that a mapped resource is not available for access as an array. 
					)===");
					break;
				case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
					printf(R"===(CCUDAHandler:
						This indicates that a mapped resource is not available for access as a pointer. 
					)===");
					break;
				case CUDA_ERROR_ECC_UNCORRECTABLE:
					printf(R"===(CCUDAHandler:
						This indicates that an uncorrectable ECC error was detected during execution. 
					)===");
					break;
				case CUDA_ERROR_UNSUPPORTED_LIMIT:
					printf(R"===(CCUDAHandler:
						This indicates that the CUlimit passed to the API call is not supported by the active device. 
					)===");
					break;
				case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
					printf(R"===(CCUDAHandler:
						This indicates that the CUcontext passed to the API call can only be bound to a single CPU thread at a time but is already bound to a CPU thread. 
					)===");
					break;
				case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
					printf(R"===(CCUDAHandler:
						This indicates that peer access is not supported across the given devices. 
					)===");
					break;
				case CUDA_ERROR_INVALID_PTX:
					printf(R"===(CCUDAHandler:
						This indicates that a PTX JIT compilation failed. 
					)===");
					break;
				case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
					printf(R"===(CCUDAHandler:
						This indicates an error with OpenGL or DirectX context. 
					)===");
					break;
				case CUDA_ERROR_NVLINK_UNCORRECTABLE:
					printf(R"===(CCUDAHandler:
						This indicates that an uncorrectable NVLink error was detected during the execution. 
					)===");
					break;
				case CUDA_ERROR_JIT_COMPILER_NOT_FOUND:
					printf(R"===(CCUDAHandler:
						This indicates that the PTX JIT compiler library was not found. 
					)===");
					break;
				case CUDA_ERROR_INVALID_SOURCE:
					printf(R"===(CCUDAHandler:
						This indicates that the device kernel source is invalid. 
					)===");
					break;
				case CUDA_ERROR_FILE_NOT_FOUND:
					printf(R"===(CCUDAHandler:
						This indicates that the file specified was not found. 
					)===");
					break;
				case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
					printf(R"===(CCUDAHandler:
						This indicates that a link to a shared object failed to resolve.
					)===");
					break;
				case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
					printf(R"===(CCUDAHandler:
						This indicates that initialization of a shared object failed.
					)===");
					break;
				case CUDA_ERROR_OPERATING_SYSTEM:
					printf(R"===(CCUDAHandler:
						This indicates that an OS call failed. 
					)===");
					break;
				case CUDA_ERROR_INVALID_HANDLE:
					printf(R"===(CCUDAHandler:
						This indicates that a resource handle passed to the API call was not valid. Resource handles are opaque types like CUstream and CUevent. 
					)===");
					break;
				case CUDA_ERROR_ILLEGAL_STATE:
					printf(R"===(CCUDAHandler:
						This indicates that a resource required by the API call is not in a valid state to perform the requested operation. 
					)===");
					break;
				case CUDA_ERROR_NOT_FOUND:
					printf(R"===(CCUDAHandler:
						This indicates that a named symbol was not found. Examples of symbols are global/constant variable names, texture names, and surface names. 
					)===");
					break;
				case CUDA_ERROR_NOT_READY:
					printf(R"===(CCUDAHandler:
						This indicates that asynchronous operations issued previously have not completed yet. This result is not actually an error, but must be indicated differently than CUDA_SUCCESS (which indicates completion). Calls that may return this value include cuEventQuery() and cuStreamQuery().
					)===");
					break;
				case CUDA_ERROR_ILLEGAL_ADDRESS:
					printf(R"===(CCUDAHandler:
						While executing a kernel, the device encountered a load or store instruction on an invalid memory address. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched. 
					)===");
					break;
				case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
					printf(R"===(CCUDAHandler:
						This indicates that a launch did not occur because it did not have appropriate resources. This error usually indicates that the user has attempted to pass too many arguments to the device kernel, or the kernel launch specifies too many threads for the kernel's register count. Passing arguments of the wrong size (i.e. a 64-bit pointer when a 32-bit int is expected) is equivalent to passing too many arguments and can also result in this error. 
					)===");
					break;
				case CUDA_ERROR_LAUNCH_TIMEOUT:
					printf(R"===(CCUDAHandler:
						This indicates that the device kernel took too long to execute. This can only occur if timeouts are enabled - see the device attribute CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched. 
					)===");
					break;
				case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
					printf(R"===(CCUDAHandler:
						This error indicates a kernel launch that uses an incompatible texturing mode. 
					)===");
					break;
				case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
					printf(R"===(CCUDAHandler:
						This error indicates that a call to cuCtxEnablePeerAccess() is trying to re-enable peer access to a context which has already had peer access to it enabled. 
					)===");
					break;
				case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
					printf(R"===(CCUDAHandler:
						This error indicates that cuCtxDisablePeerAccess() is trying to disable peer access which has not been enabled yet via cuCtxEnablePeerAccess(). 
					)===");
					break;
				case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
					printf(R"===(CCUDAHandler:
						This error indicates that the primary context for the specified device has already been initialized. 
					)===");
					break;
				case CUDA_ERROR_CONTEXT_IS_DESTROYED:
					printf(R"===(CCUDAHandler:
						This error indicates that the context current to the calling thread has been destroyed using cuCtxDestroy, or is a primary context which has not yet been initialized. 
					)===");
					break;
				case CUDA_ERROR_ASSERT:
					printf(R"===(CCUDAHandler:
						A device-side assert triggered during kernel execution. The context cannot be used anymore, and must be destroyed. All existing device memory allocations from this context are invalid and must be reconstructed if the program is to continue using CUDA. 
					)===");
					break;
				case CUDA_ERROR_TOO_MANY_PEERS:
					printf(R"===(CCUDAHandler:
						This error indicates that the hardware resources required to enable peer access have been exhausted for one or more of the devices passed to cuCtxEnablePeerAccess(). 
					)===");
					break;
				case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
					printf(R"===(CCUDAHandler:
						This error indicates that the memory range passed to cuMemHostRegister() has already been registered. 
					)===");
					break;
				case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
					printf(R"===(CCUDAHandler:
						This error indicates that the pointer passed to cuMemHostUnregister() does not correspond to any currently registered memory region. 
					)===");
					break;
				case CUDA_ERROR_HARDWARE_STACK_ERROR:
					printf(R"===(CCUDAHandler:
						While executing a kernel, the device encountered a stack error. This can be due to stack corruption or exceeding the stack size limit. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
					)===");
					break;
				case CUDA_ERROR_ILLEGAL_INSTRUCTION:
					printf(R"===(CCUDAHandler:
						While executing a kernel, the device encountered an illegal instruction. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
					)===");
					break;
				case CUDA_ERROR_MISALIGNED_ADDRESS:
					printf(R"===(CCUDAHandler:
						While executing a kernel, the device encountered a load or store instruction on a memory address which is not aligned. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched. 
					)===");
					break;
				case CUDA_ERROR_INVALID_ADDRESS_SPACE:
					printf(R"===(CCUDAHandler:
						While executing a kernel, the device encountered an instruction which can only operate on memory locations in certain address spaces (global, shared, or local), but was supplied a memory address not belonging to an allowed address space. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
					)===");
					break;				
				case CUDA_ERROR_INVALID_PC:
					printf(R"===(CCUDAHandler:
						While executing a kernel, the device program counter wrapped its address space. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
					)===");
					break;				
				case CUDA_ERROR_LAUNCH_FAILED:
					printf(R"===(CCUDAHandler:
						An exception occurred on the device while executing a kernel. Common causes include dereferencing an invalid device pointer and accessing out of bounds shared memory. Less common cases can be system specific - more information about these cases can be found in the system specific user guide. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched. 
					)===");
					break;
				case CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE:
					printf(R"===(CCUDAHandler:
						This error indicates that the number of blocks launched per grid for a kernel that was launched via either cuLaunchCooperativeKernel or cuLaunchCooperativeKernelMultiDevice exceeds the maximum number of blocks as allowed by cuOccupancyMaxActiveBlocksPerMultiprocessor or cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors as specified by the device attribute CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.
					)===");
					break;			
				case CUDA_ERROR_NOT_PERMITTED:
					printf(R"===(CCUDAHandler:
						This error indicates that the attempted operation is not permitted.
					)===");
					break;
				case CUDA_ERROR_NOT_SUPPORTED:
					printf(R"===(CCUDAHandler:
						This error indicates that the attempted operation is not supported on the current system or device.
					)===");
					break;				
				case CUDA_ERROR_SYSTEM_NOT_READY:
					printf(R"===(CCUDAHandler:
						This error indicates that the system is not yet ready to start any CUDA work. To continue using CUDA, verify the system configuration is in a valid state and all required driver daemons are actively running. More information about this error can be found in the system specific user guide.
					)===");
					break;
				case CUDA_ERROR_SYSTEM_DRIVER_MISMATCH:
					printf(R"===(CCUDAHandler:
						This error indicates that there is a mismatch between the versions of the display driver and the CUDA driver. Refer to the compatibility documentation for supported versions. 
					)===");
					break;
				case CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE:
					printf(R"===(CCUDAHandler:
						This error indicates that the system was upgraded to run with forward compatibility but the visible hardware detected by CUDA does not support this configuration. Refer to the compatibility documentation for the supported hardware matrix or ensure that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES environment variable.
					)===");
					break;
				case CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED:
					printf(R"===(CCUDAHandler:
						This error indicates that the operation is not permitted when the stream is capturing. 
					)===");
					break;
				case CUDA_ERROR_STREAM_CAPTURE_INVALIDATED:
					printf(R"===(CCUDAHandler:
						This error indicates that the current capture sequence on the stream has been invalidated due to a previous error. 
					)===");
					break;
				case CUDA_ERROR_STREAM_CAPTURE_MERGE:
					printf(R"===(CCUDAHandler:
						This error indicates that the operation would have resulted in a merge of two independent capture sequences. 
					)===");
					break;
				case CUDA_ERROR_STREAM_CAPTURE_UNMATCHED:
					printf(R"===(CCUDAHandler:
						This error indicates that the capture was not initiated in this stream. 
					)===");
					break;
				case CUDA_ERROR_STREAM_CAPTURE_UNJOINED:
					printf(R"===(CCUDAHandler:
						This error indicates that the capture sequence contains a fork that was not joined to the primary stream.
					)===");
					break;
				case CUDA_ERROR_STREAM_CAPTURE_ISOLATION:
					printf(R"===(CCUDAHandler:
						This error indicates that a dependency would have been created which crosses the capture sequence boundary. Only implicit in-stream ordering dependencies are allowed to cross the boundary.
					)===");
					break;
				case CUDA_ERROR_STREAM_CAPTURE_IMPLICIT:
					printf(R"===(CCUDAHandler:
						This error indicates a disallowed implicit dependency on a current capture sequence from cudaStreamLegacy. 
					)===");
					break;
				case CUDA_ERROR_CAPTURED_EVENT:
					printf(R"===(CCUDAHandler:
						This error indicates that the operation is not permitted on an event which was last recorded in a capturing stream. 
					)===");
					break;
				case CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD:
					printf(R"===(CCUDAHandler:
						A stream capture sequence not initiated with the CU_STREAM_CAPTURE_MODE_RELAXED argument to cuStreamBeginCapture was passed to cuStreamEndCapture in a different thread. 
					)===");
					break;
				case CUDA_ERROR_TIMEOUT:
					printf(R"===(CCUDAHandler:
						This error indicates that the timeout specified for the wait operation has lapsed. 
					)===");
					break;
				case CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE:
					printf(R"===(CCUDAHandler:
						This error indicates that the graph update was not performed because it included changes which violated constraints specific to instantiated graph update. 
					)===");
					break;
				case CUDA_ERROR_UNKNOWN:
				default:
					printf("CCUDAHandler: Unknown CUDA Error!\n");
					break;
			}
			_IRR_DEBUG_BREAK_IF(true);
			return false;
		}

		static CUresult getDefaultGLDevices(uint32_t* foundCount, CUdevice* pCudaDevices, uint32_t cudaDeviceCount)
		{
			return cuda.pcuGLGetDevices_v2(foundCount,pCudaDevices,cudaDeviceCount,CU_GL_DEVICE_LIST_ALL);
		}

		template<typename ObjType>
		struct GraphicsAPIObjLink
		{
			core::smart_refctd_ptr<ObjType> obj;
			CUgraphicsResource cudaHandle;

			GraphicsAPIObjLink() : obj(nullptr), cudaHandle(nullptr) {}

			GraphicsAPIObjLink(const GraphicsAPIObjLink& other) = delete;
			GraphicsAPIObjLink& operator=(const GraphicsAPIObjLink& other) = delete;

			~GraphicsAPIObjLink()
			{
				if (obj)
					CCUDAHandler::cuda.pcuGraphicsUnregisterResource(cudaHandle);
			}
		};
		static CUresult registerBuffer(GraphicsAPIObjLink<video::IGPUBuffer>* link, uint32_t flags=CU_GRAPHICS_REGISTER_FLAGS_NONE)
		{
			assert(link->obj);
			auto glbuf = static_cast<video::COpenGLBuffer*>(link->obj.get());
			auto retval = cuda.pcuGraphicsGLRegisterBuffer(&link->cudaHandle,glbuf->getOpenGLName(),flags);
			if (retval!=CUDA_SUCCESS)
				link->obj = nullptr;
			return retval;
		}
		static CUresult registerImage(GraphicsAPIObjLink<video::ITexture>* link, uint32_t flags=CU_GRAPHICS_REGISTER_FLAGS_NONE)
		{
			assert(link->obj);
			
			auto format = link->obj->getColorFormat();
			if (asset::isBlockCompressionFormat(format) || asset::isDepthOrStencilFormat(format) || asset::isScaledFormat(format) || asset::isPlanarFormat(format))
				return CUDA_ERROR_INVALID_IMAGE;

			auto glimg = static_cast<video::COpenGLFilterableTexture*>(link->obj.get());
			GLenum target = glimg->getOpenGLTextureType();
			switch (target)
			{
				case GL_TEXTURE_2D:
				case GL_TEXTURE_2D_ARRAY:
				case GL_TEXTURE_CUBE_MAP:
				case GL_TEXTURE_3D:
					break;
				default:
					return CUDA_ERROR_INVALID_IMAGE;
					break;
			}
			auto retval = cuda.pcuGraphicsGLRegisterImage(&link->cudaHandle,glimg->getOpenGLName(),target,flags);
			if (retval != CUDA_SUCCESS)
				link->obj = nullptr;
			return retval;
		}

		

		static bool defaultHandleResult(nvrtcResult result)
		{
			switch (result)
			{
				case NVRTC_SUCCESS:
					return true;
					break;
				default:
					if (nvrtc.pnvrtcGetErrorString)
						printf("%s\n",nvrtc.pnvrtcGetErrorString(result));
					else
						printf(R"===(CudaHandler: `pnvrtcGetErrorString` is nullptr, the nvrtc library probably not found on the system.\n)===");
					break;
			}
			_IRR_DEBUG_BREAK_IF(true);
			return false;
		}

		static nvrtcResult createProgram(	nvrtcProgram* prog, const char* source, const char* name,
											const char* const* headersBegin=nullptr, const char* const* headersEnd=nullptr,
											const char* const* includeNamesBegin=nullptr, const char* const* includeNamesEnd=nullptr)
		{
			auto headerCount = std::distance(headersBegin, headersEnd);
			if (headerCount)
			{
				if (std::distance(includeNamesBegin,includeNamesEnd)!=headerCount)
					return NVRTC_ERROR_INVALID_INPUT;
				return nvrtc.pnvrtcCreateProgram(prog, source, name, headerCount, headersBegin, includeNamesBegin);
			}
			else
				return nvrtc.pnvrtcCreateProgram(prog, source, name, 0u, nullptr, nullptr);
		}

		template<typename HeaderFileIt>
		static nvrtcResult createProgram(	nvrtcProgram* prog, irr::io::IReadFile* main,
											const HeaderFileIt includesBegin, const HeaderFileIt includesEnd)
		{
			int numHeaders = std::distance(includesBegin,includesEnd);
			core::vector<const char*> headers(numHeaders);
			core::vector<const char*> includeNames(numHeaders);
			size_t sourceSize = main->getSize();
			size_t sourceIt = sourceSize;
			sourceSize++;
			for (auto it=includesBegin; it!=includesEnd; it++)
			{
				sourceSize += it->getSize()+1u;
				includeNames.emplace_back(it->getFileName().c_str());
			}
			core::vector<char> sources(sourceSize);
			main->read(sources.data(),sourceIt);
			sources[sourceIt++] = 0;
			for (auto it=includesBegin; it!=includesEnd; it++)
			{
				auto oldpos = it->getPos();
				it->seek(0ull);

				auto ptr = sources.data()+sourceIt;
				headers.push_back(ptr);
				auto filesize = it->getSize();
				it->read(ptr,filesize);
				sourceIt += filesize;
				sources[sourceIt++] = 0;

				it->seek(oldpos);
			}
			return nvrtc.pnvrtcCreateProgram(prog, sources.data(), main->getFileName().c_str(), numHeaders, headers.data(), includeNames.data());
		}

		static nvrtcResult compileProgram(nvrtcProgram prog, const std::initializer_list<const char*>& options={_IRR_DEFAULT_NVRTC_OPTIONS})
		{
			return nvrtc.pnvrtcCompileProgram(prog, options.size(), options.begin());
		}

		static nvrtcResult compileProgram(nvrtcProgram prog, const std::vector<const char*>& options)
		{
			return nvrtc.pnvrtcCompileProgram(prog, options.size(), options.data());
		}
		// note: allow to control  `-restrict` then includes and defines

		static nvrtcResult getProgramLog(nvrtcProgram prog, std::string& log)
		{
			size_t _size = 0ull;
			nvrtcResult sizeRes = nvrtc.pnvrtcGetProgramLogSize(prog, &_size);
			if (sizeRes != NVRTC_SUCCESS)
				return sizeRes;
			if (_size == 0ull)
				return NVRTC_ERROR_INVALID_INPUT;

			log.resize(_size);
			return nvrtc.pnvrtcGetProgramLog(prog, log.data());
		}

		static nvrtcResult getPTX(nvrtcProgram prog, std::string& ptx)
		{
			size_t _size = 0ull;
			nvrtcResult sizeRes = nvrtc.pnvrtcGetPTXSize(prog, &_size);
			if (sizeRes!=NVRTC_SUCCESS)
				return sizeRes;
			if (_size==0ull)
				return NVRTC_ERROR_INVALID_INPUT;

			ptx.resize(_size);
			return nvrtc.pnvrtcGetPTX(prog,ptx.data());
		}

		template<typename OptionsT = const std::initializer_list<const char*>&>
		static nvrtcResult compileDirectlyToPTX(std::string& ptx, irr::io::IReadFile* main,
			const char* const* headersBegin = nullptr, const char* const* headersEnd = nullptr,
			const char* const* includeNamesBegin = nullptr, const char* const* includeNamesEnd = nullptr,
			OptionsT options = { _IRR_DEFAULT_NVRTC_OPTIONS },
			std::string* log = nullptr)
		{
			nvrtcProgram program = nullptr;
			nvrtcResult result = NVRTC_ERROR_PROGRAM_CREATION_FAILURE;
			auto cleanup = core::makeRAIIExiter([&program, &result]() -> void {
				if (result != NVRTC_SUCCESS && program)
					cuda::CCUDAHandler::nvrtc.pnvrtcDestroyProgram(&program);
				});
			
			char* data = new char[main->getSize()+1ull];
			main->read(data, main->getSize());
			data[main->getSize()] = 0;
			result = cuda::CCUDAHandler::createProgram(&program, data, main->getFileName().c_str(), headersBegin, headersEnd, includeNamesBegin, includeNamesEnd);
			delete[] data;

			if (result != NVRTC_SUCCESS)
				return result;

			return result = compileDirectlyToPTX_helper<OptionsT>(ptx, program, std::forward<OptionsT>(options), log);
		}

		template<typename CompileArgsT, typename OptionsT=const std::initializer_list<const char*>&>
		static nvrtcResult compileDirectlyToPTX(std::string& ptx, irr::io::IReadFile* main,
												CompileArgsT includesBegin, CompileArgsT includesEnd,
												OptionsT options={_IRR_DEFAULT_NVRTC_OPTIONS},
												std::string* log=nullptr)
		{
			nvrtcProgram program = nullptr;
			nvrtcResult result = NVRTC_ERROR_PROGRAM_CREATION_FAILURE;
			auto cleanup = core::makeRAIIExiter([&program,&result]() -> void {
				if (result!=NVRTC_SUCCESS && program)
					cuda::CCUDAHandler::nvrtc.pnvrtcDestroyProgram(&program);
			});
			result = cuda::CCUDAHandler::createProgram(&program, main, includesBegin, includesEnd);
			if (result!=NVRTC_SUCCESS)
				return result;

			return result = compileDirectlyToPTX_helper<OptionsT>(ptx,program,std::forward<OptionsT>(options),log);
		}


	protected:
		template<typename OptionsT = const std::initializer_list<const char*>&>
		static nvrtcResult compileDirectlyToPTX_helper(std::string& ptx, nvrtcProgram program, OptionsT options, std::string* log=nullptr)
		{
			nvrtcResult result = cuda::CCUDAHandler::compileProgram(program,options);
			if (log)
				cuda::CCUDAHandler::getProgramLog(program, *log);
			if (result!=NVRTC_SUCCESS)
				return result;

			return cuda::CCUDAHandler::getPTX(program, ptx);
		}
};

}
}

#endif // _IRR_COMPILE_WITH_CUDA_

#endif
