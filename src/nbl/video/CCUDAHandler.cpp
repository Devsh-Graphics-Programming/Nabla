// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/video/CCUDAHandler.h"

#ifdef _NBL_COMPILE_WITH_CUDA_
#include "jitify/jitify.hpp"


namespace nbl::video
{
	
bool CCUDAHandler::defaultHandleResult(CUresult result, const system::logger_opt_ptr& logger)
{
	switch (result)
	{
		case CUDA_SUCCESS:
			return true;
			break;
		case CUDA_ERROR_INVALID_VALUE:
			logger.log(R"===(CCUDAHandler:
				This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_OUT_OF_MEMORY:
			logger.log(R"===(CCUDAHandler:
				The API call failed because it was unable to allocate enough memory to perform the requested operation.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_NOT_INITIALIZED:
			logger.log(R"===(CCUDAHandler:
				This indicates that the CUDA driver has not been initialized with cuInit() or that initialization has failed. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_DEINITIALIZED:
			logger.log(R"===(CCUDAHandler:
				This indicates that the CUDA driver is in the process of shutting down.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_PROFILER_DISABLED:
			logger.log(R"===(CCUDAHandler:
				This indicates profiler is not initialized for this run. This can happen when the application is running with external profiling tools like visual profiler.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_NO_DEVICE:
			logger.log(R"===(CCUDAHandler:
				This indicates that no CUDA-capable devices were detected by the installed CUDA driver. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_INVALID_DEVICE:
			logger.log(R"===(CCUDAHandler:
				This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_INVALID_IMAGE:
			logger.log(R"===(CCUDAHandler:
				This indicates that the device kernel image is invalid. This can also indicate an invalid CUDA module. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_INVALID_CONTEXT:
			logger.log(R"===(CCUDAHandler:
				This most frequently indicates that there is no context bound to the current thread. This can also be returned if the context passed to an API call is not a valid handle (such as a context that has had cuCtxDestroy() invoked on it). This can also be returned if a user mixes different API versions (i.e. 3010 context with 3020 API calls). See cuCtxGetApiVersion() for more details.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_MAP_FAILED:
			logger.log(R"===(CCUDAHandler:
				This indicates that a map or register operation has failed.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_UNMAP_FAILED:
			logger.log(R"===(CCUDAHandler:
				This indicates that an unmap or unregister operation has failed.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_ARRAY_IS_MAPPED:
			logger.log(R"===(CCUDAHandler:
				This indicates that the specified array is currently mapped and thus cannot be destroyed.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_ALREADY_MAPPED:
			logger.log(R"===(CCUDAHandler:
				This indicates that the resource is already mapped.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_NO_BINARY_FOR_GPU:
			logger.log(R"===(CCUDAHandler:
				This indicates that there is no kernel image available that is suitable for the device. This can occur when a user specifies code generation options for a particular CUDA source file that do not include the corresponding device configuration.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_ALREADY_ACQUIRED:
			logger.log(R"===(CCUDAHandler:
				This indicates that a resource has already been acquired. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_NOT_MAPPED:
			logger.log(R"===(CCUDAHandler:
				This indicates that a resource is not mapped.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
			logger.log(R"===(CCUDAHandler:
				This indicates that a mapped resource is not available for access as an array. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
			logger.log(R"===(CCUDAHandler:
				This indicates that a mapped resource is not available for access as a pointer. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_ECC_UNCORRECTABLE:
			logger.log(R"===(CCUDAHandler:
				This indicates that an uncorrectable ECC error was detected during execution. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_UNSUPPORTED_LIMIT:
			logger.log(R"===(CCUDAHandler:
				This indicates that the CUlimit passed to the API call is not supported by the active device. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
			logger.log(R"===(CCUDAHandler:
				This indicates that the CUcontext passed to the API call can only be bound to a single CPU thread at a time but is already bound to a CPU thread. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
			logger.log(R"===(CCUDAHandler:
				This indicates that peer access is not supported across the given devices. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_INVALID_PTX:
			logger.log(R"===(CCUDAHandler:
				This indicates that a PTX JIT compilation failed. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
			logger.log(R"===(CCUDAHandler:
				This indicates an error with OpenGL or DirectX context. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_NVLINK_UNCORRECTABLE:
			logger.log(R"===(CCUDAHandler:
				This indicates that an uncorrectable NVLink error was detected during the execution. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_JIT_COMPILER_NOT_FOUND:
			logger.log(R"===(CCUDAHandler:
				This indicates that the PTX JIT compiler library was not found. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_INVALID_SOURCE:
			logger.log(R"===(CCUDAHandler:
				This indicates that the device kernel source is invalid. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_FILE_NOT_FOUND:
			logger.log(R"===(CCUDAHandler:
				This indicates that the file specified was not found. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
			logger.log(R"===(CCUDAHandler:
				This indicates that a link to a shared object failed to resolve.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
			logger.log(R"===(CCUDAHandler:
				This indicates that initialization of a shared object failed.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_OPERATING_SYSTEM:
			logger.log(R"===(CCUDAHandler:
				This indicates that an OS call failed. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_INVALID_HANDLE:
			logger.log(R"===(CCUDAHandler:
				This indicates that a resource handle passed to the API call was not valid. Resource handles are opaque types like CUstream and CUevent. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_ILLEGAL_STATE:
			logger.log(R"===(CCUDAHandler:
				This indicates that a resource required by the API call is not in a valid state to perform the requested operation. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_NOT_FOUND:
			logger.log(R"===(CCUDAHandler:
				This indicates that a named symbol was not found. Examples of symbols are global/constant variable names, texture names, and surface names. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_NOT_READY:
			logger.log(R"===(CCUDAHandler:
				This indicates that asynchronous operations issued previously have not completed yet. This result is not actually an error, but must be indicated differently than CUDA_SUCCESS (which indicates completion). Calls that may return this value include cuEventQuery() and cuStreamQuery().
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_ILLEGAL_ADDRESS:
			logger.log(R"===(CCUDAHandler:
				While executing a kernel, the device encountered a load or store instruction on an invalid memory address. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
			logger.log(R"===(CCUDAHandler:
				This indicates that a launch did not occur because it did not have appropriate resources. This error usually indicates that the user has attempted to pass too many arguments to the device kernel, or the kernel launch specifies too many threads for the kernel's register count. Passing arguments of the wrong size (i.e. a 64-bit pointer when a 32-bit int is expected) is equivalent to passing too many arguments and can also result in this error. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_LAUNCH_TIMEOUT:
			logger.log(R"===(CCUDAHandler:
				This indicates that the device kernel took too long to execute. This can only occur if timeouts are enabled - see the device attribute CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
			logger.log(R"===(CCUDAHandler:
				This error indicates a kernel launch that uses an incompatible texturing mode. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
			logger.log(R"===(CCUDAHandler:
				This error indicates that a call to cuCtxEnablePeerAccess() is trying to re-enable peer access to a context which has already had peer access to it enabled. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
			logger.log(R"===(CCUDAHandler:
				This error indicates that cuCtxDisablePeerAccess() is trying to disable peer access which has not been enabled yet via cuCtxEnablePeerAccess(). 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the primary context for the specified device has already been initialized. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_CONTEXT_IS_DESTROYED:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the context current to the calling thread has been destroyed using cuCtxDestroy, or is a primary context which has not yet been initialized. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_ASSERT:
			logger.log(R"===(CCUDAHandler:
				A device-side assert triggered during kernel execution. The context cannot be used anymore, and must be destroyed. All existing device memory allocations from this context are invalid and must be reconstructed if the program is to continue using CUDA. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_TOO_MANY_PEERS:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the hardware resources required to enable peer access have been exhausted for one or more of the devices passed to cuCtxEnablePeerAccess(). 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the memory range passed to cuMemHostRegister() has already been registered. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the pointer passed to cuMemHostUnregister() does not correspond to any currently registered memory region. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_HARDWARE_STACK_ERROR:
			logger.log(R"===(CCUDAHandler:
				While executing a kernel, the device encountered a stack error. This can be due to stack corruption or exceeding the stack size limit. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_ILLEGAL_INSTRUCTION:
			logger.log(R"===(CCUDAHandler:
				While executing a kernel, the device encountered an illegal instruction. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_MISALIGNED_ADDRESS:
			logger.log(R"===(CCUDAHandler:
				While executing a kernel, the device encountered a load or store instruction on a memory address which is not aligned. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_INVALID_ADDRESS_SPACE:
			logger.log(R"===(CCUDAHandler:
				While executing a kernel, the device encountered an instruction which can only operate on memory locations in certain address spaces (global, shared, or local), but was supplied a memory address not belonging to an allowed address space. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
			)===",system::ILogger::ELL_ERROR);
			break;				
		case CUDA_ERROR_INVALID_PC:
			logger.log(R"===(CCUDAHandler:
				While executing a kernel, the device program counter wrapped its address space. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
			)===",system::ILogger::ELL_ERROR);
			break;				
		case CUDA_ERROR_LAUNCH_FAILED:
			logger.log(R"===(CCUDAHandler:
				An exception occurred on the device while executing a kernel. Common causes include dereferencing an invalid device pointer and accessing out of bounds shared memory. Less common cases can be system specific - more information about these cases can be found in the system specific user guide. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the number of blocks launched per grid for a kernel that was launched via either cuLaunchCooperativeKernel or cuLaunchCooperativeKernelMultiDevice exceeds the maximum number of blocks as allowed by cuOccupancyMaxActiveBlocksPerMultiprocessor or cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors as specified by the device attribute CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.
			)===",system::ILogger::ELL_ERROR);
			break;			
		case CUDA_ERROR_NOT_PERMITTED:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the attempted operation is not permitted.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_NOT_SUPPORTED:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the attempted operation is not supported on the current system or device.
			)===",system::ILogger::ELL_ERROR);
			break;				
		case CUDA_ERROR_SYSTEM_NOT_READY:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the system is not yet ready to start any CUDA work. To continue using CUDA, verify the system configuration is in a valid state and all required driver daemons are actively running. More information about this error can be found in the system specific user guide.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_SYSTEM_DRIVER_MISMATCH:
			logger.log(R"===(CCUDAHandler:
				This error indicates that there is a mismatch between the versions of the display driver and the CUDA driver. Refer to the compatibility documentation for supported versions. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the system was upgraded to run with forward compatibility but the visible hardware detected by CUDA does not support this configuration. Refer to the compatibility documentation for the supported hardware matrix or ensure that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES environment variable.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the operation is not permitted when the stream is capturing. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_STREAM_CAPTURE_INVALIDATED:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the current capture sequence on the stream has been invalidated due to a previous error. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_STREAM_CAPTURE_MERGE:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the operation would have resulted in a merge of two independent capture sequences. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_STREAM_CAPTURE_UNMATCHED:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the capture was not initiated in this stream. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_STREAM_CAPTURE_UNJOINED:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the capture sequence contains a fork that was not joined to the primary stream.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_STREAM_CAPTURE_ISOLATION:
			logger.log(R"===(CCUDAHandler:
				This error indicates that a dependency would have been created which crosses the capture sequence boundary. Only implicit in-stream ordering dependencies are allowed to cross the boundary.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_STREAM_CAPTURE_IMPLICIT:
			logger.log(R"===(CCUDAHandler:
				This error indicates a disallowed implicit dependency on a current capture sequence from cudaStreamLegacy. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_CAPTURED_EVENT:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the operation is not permitted on an event which was last recorded in a capturing stream. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD:
			logger.log(R"===(CCUDAHandler:
				A stream capture sequence not initiated with the CU_STREAM_CAPTURE_MODE_RELAXED argument to cuStreamBeginCapture was passed to cuStreamEndCapture in a different thread. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_TIMEOUT:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the timeout specified for the wait operation has lapsed. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the graph update was not performed because it included changes which violated constraints specific to instantiated graph update. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_UNKNOWN:
		default:
			logger.log("CCUDAHandler: Unknown CUDA Error!\n",system::ILogger::ELL_ERROR);
			break;
	}
	_NBL_DEBUG_BREAK_IF(true);
	return false;
}

bool CCUDAHandler::defaultHandleResult(nvrtcResult result)
{
	switch (result)
	{
		case NVRTC_SUCCESS:
			return true;
			break;
		default:
			if (m_nvrtc.pnvrtcGetErrorString)
				m_logger.log("%s\n",system::ILogger::ELL_ERROR,m_nvrtc.pnvrtcGetErrorString(result));
			else
				m_logger.log(R"===(CudaHandler: `pnvrtcGetErrorString` is nullptr, the nvrtc library probably not found on the system.\n)===",system::ILogger::ELL_ERROR);
			break;
	}
	_NBL_DEBUG_BREAK_IF(true);
	return false;
}

core::smart_refctd_ptr<CCUDAHandler> CCUDAHandler::create(system::ISystem* system, core::smart_refctd_ptr<system::ILogger>&& _logger)
{
	CUDA cuda = CUDA(
		#if defined(_NBL_WINDOWS_API_)
			"nvcuda"
		#elif defined(_NBL_POSIX_API_)
			"cuda"
		#endif
	);
	
	NVRTC nvrtc = {};
	#if defined(_NBL_WINDOWS_API_)
	// Perpetual TODO: any new CUDA releases we need to account for?
	const char* nvrtc64_versions[] = { "nvrtc64_111","nvrtc64_110","nvrtc64_102","nvrtc64_101","nvrtc64_100","nvrtc64_92","nvrtc64_91","nvrtc64_90","nvrtc64_80","nvrtc64_75","nvrtc64_70",nullptr };
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
	#elif defined(_NBL_POSIX_API_)
	nvrtc = NVRTC("nvrtc");
	//nvrtc_builtins = NVRTC("nvrtc-builtins");
	#endif
	

	// need a complex safe calling chain because DLL/SO might not have loaded
	#define SAFE_CUDA_CALL(FUNC,...) \
	{\
		if (!cuda.p ## FUNC)\
			return nullptr;\
		auto result = cuda.p ## FUNC ## (__VA_ARGS__);\
		if (result!=CUDA_SUCCESS)\
			return nullptr;\
	}
	
	SAFE_CUDA_CALL(cuInit,0)
				
	int cudaVersion = 0;
	SAFE_CUDA_CALL(cuDriverGetVersion,&cudaVersion)
	if (cudaVersion<9000)
		return nullptr;

	// stop the pollution
	#undef SAFE_CUDA_CALL


	// check nvrtc existence and compatibility
	if (!nvrtc.pnvrtcVersion)
		return nullptr;
	int nvrtcVersion[2] = { -1,-1 };
	nvrtc.pnvrtcVersion(nvrtcVersion+0,nvrtcVersion+1);
	if (nvrtcVersion[0]<9)
		return nullptr;


	CCUDAHandler* handler = new CCUDAHandler(std::move(cuda), std::move(nvrtc),/*std::move(headers)*/{}, std::move(_logger), cudaVersion);
	return core::smart_refctd_ptr<CCUDAHandler>(handler,core::dont_grab);
}

#if 0
core::vector<core::smart_refctd_ptr<const io::IReadFile> > CCUDAHandler::headers;
core::vector<const char*> CCUDAHandler::headerContents;
core::vector<const char*> CCUDAHandler::headerNames;
#endif

nvrtcResult CCUDAHandler::getProgramLog(nvrtcProgram prog, std::string& log)
{
	size_t _size = 0ull;
	nvrtcResult sizeRes = m_nvrtc.pnvrtcGetProgramLogSize(prog, &_size);
	if (sizeRes != NVRTC_SUCCESS)
		return sizeRes;
	if (_size == 0ull)
		return NVRTC_ERROR_INVALID_INPUT;

	log.resize(_size);
	return m_nvrtc.pnvrtcGetProgramLog(prog,log.data());
}

std::pair<core::smart_refctd_ptr<asset::ICPUBuffer>,nvrtcResult> CCUDAHandler::getPTX(nvrtcProgram prog)
{
	size_t _size = 0ull;
	nvrtcResult sizeRes = m_nvrtc.pnvrtcGetPTXSize(prog,&_size);
	if (sizeRes!=NVRTC_SUCCESS)
		return {nullptr,sizeRes};
	if (_size==0ull)
		return {nullptr,NVRTC_ERROR_INVALID_INPUT};

	auto ptx = core::make_smart_refctd_ptr<asset::ICPUBuffer>(_size);
	return {std::move(ptx),m_nvrtc.pnvrtcGetPTX(prog,reinterpret_cast<char*>(ptx->getPointer()))};
}

#if 0
	const int archVersion[2] = { tmp.attributes[CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR],tmp.attributes[CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR] };
	if (archVersion[0] > 8 || archVersion[0] == 8 && archVersion[1] > 0)
	{
		assert(strcmp(virtualCUDAArchitectures[13], "-arch=compute_80") == 0);
		if (!virtualCUDAArchitecture)
			virtualCUDAArchitecture = virtualCUDAArchitectures[13];
	}
	else
	{
		const std::string virtualArchString = "-arch=compute_" + std::to_string(archVersion[0]) + std::to_string(archVersion[1]);

		int32_t i = sizeof(virtualCUDAArchitectures) / sizeof(const char*);
		while (i > 0)
			if (virtualCUDAArchitecture == virtualCUDAArchitectures[--i] || !virtualCUDAArchitecture)
				break;

		if (!virtualCUDAArchitecture || virtualArchString != virtualCUDAArchitecture)
		{
			i++;
			while (i > 0)
				if (virtualArchString == virtualCUDAArchitectures[--i])
				{
					virtualCUDAArchitecture = virtualCUDAArchitectures[i];
					break;
				}
		}
	}

if (!virtualCUDAArchitecture)
return result = CUDA_ERROR_INVALID_DEVICE;
#endif

core::smart_refctd_ptr<CCUDADevice> CCUDAHandler::createDevice(core::smart_refctd_ptr<CVulkanConnection>&& vulkanConnection, IPhysicalDevice* physicalDevice)
{
	if (!vulkanConnection)
		return nullptr;
	const auto devices = vulkanConnection->getPhysicalDevices();
	if (std::find(devices.begin(),devices.end(),physicalDevice)==devices.end())
		return nullptr;

    int deviceCount = 0;
    if (m_cuda.pcuDeviceGetCount(&deviceCount)!=CUDA_SUCCESS || deviceCount<=0)
		return nullptr;

    for (int ordinal=0; ordinal<deviceCount; ordinal++)
	{
		CUdevice handle = -1;
		if (m_cuda.pcuDeviceGet(&handle,ordinal)!=CUDA_SUCCESS || handle<0)
			continue;

		CUuuid uuid = {};
		if (m_cuda.pcuDeviceGetUuid(&uuid,handle)!=CUDA_SUCCESS)
			continue;
        if (!memcmp(&uuid,&physicalDevice->getLimits().deviceUUID,VK_UUID_SIZE))
		{
			int attributes[CU_DEVICE_ATTRIBUTE_MAX] = {};
			for (int i=0; i<CU_DEVICE_ATTRIBUTE_MAX; i++)
				m_cuda.pcuDeviceGetAttribute(attributes+i,static_cast<CUdevice_attribute>(i),handle);

			CCUDADevice::E_VIRTUAL_ARCHITECTURE arch = CCUDADevice::EVA_COUNT;
			const int& archMajor = attributes[CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR];
			const int& archMinor = attributes[CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR];
			switch (archMajor)
			{
				case 3:
					switch (archMinor)
					{
						case 0:
							arch = CCUDADevice::EVA_30;
							break;
						case 2:
							arch = CCUDADevice::EVA_32;
							break;
						case 5:
							arch = CCUDADevice::EVA_35;
							break;
						case 7:
							arch = CCUDADevice::EVA_37;
							break;
						default:
							break;
					}
					break;
				case 5:
					switch (archMinor)
					{
						case 0:
							arch = CCUDADevice::EVA_50;
							break;
						case 2:
							arch = CCUDADevice::EVA_52;
							break;
						case 3:
							arch = CCUDADevice::EVA_53;
							break;
						default:
							break;
					}
					break;
				case 6:
					switch (archMinor)
					{
						case 0:
							arch = CCUDADevice::EVA_60;
							break;
						case 1:
							arch = CCUDADevice::EVA_61;
							break;
						case 2:
							arch = CCUDADevice::EVA_62;
							break;
						default:
							break;
					}
					break;
				case 7:
					switch (archMinor)
					{
						case 0:
							arch = CCUDADevice::EVA_70;
							break;
						case 2:
							arch = CCUDADevice::EVA_72;
							break;
						case 5:
							arch = CCUDADevice::EVA_75;
							break;
						default:
							break;
					}
					break;
				default:
					if (archMajor>=8)
						arch = CCUDADevice::EVA_80;
					break;
			}
			if (arch==CCUDADevice::EVA_COUNT)
				continue;

			auto device = new CCUDADevice(std::move(vulkanConnection),physicalDevice,arch);
            return core::smart_refctd_ptr<CCUDADevice>(device,core::dont_grab);
        }
    }
	return nullptr;
}

}

#endif // _NBL_COMPILE_WITH_CUDA_
