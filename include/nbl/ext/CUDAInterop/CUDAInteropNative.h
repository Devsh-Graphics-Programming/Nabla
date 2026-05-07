// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_CUDA_INTEROP_NATIVE_H_INCLUDED_
#define _NBL_EXT_CUDA_INTEROP_NATIVE_H_INCLUDED_

#include "nbl/video/CUDAInterop.h"

#include "nbl/asset/ICPUBuffer.h"
#include "nbl/system/DynamicFunctionCaller.h"

#include <concepts>
#include <string>
#include <type_traits>
#include <utility>

#include "cuda.h"
#include "nvrtc.h"
#if CUDA_VERSION < 13000
	#error "Need CUDA 13.0 SDK or higher."
#endif

namespace nbl::video::cuda_native
{

inline constexpr int MinimumCUDADriverVersion = 13000;
inline constexpr int MinimumNVRTCMajorVersion = MinimumCUDADriverVersion/1000;

using LibLoader = system::DefaultFuncPtrLoader;

NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(CUDA,LibLoader
	,cuCtxCreate_v4
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
	,cuDeviceGetUuid_v2
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
	,cuSurfObjectCreate
	,cuSurfObjectDestroy
	,cuTexObjectCreate
	,cuTexObjectDestroy
	,cuImportExternalMemory
	,cuDestroyExternalMemory
	,cuExternalMemoryGetMappedBuffer
	,cuMemUnmap
	,cuMemAddressFree
	,cuMemGetAllocationGranularity
	,cuMemAddressReserve
	,cuMemCreate
	,cuMemExportToShareableHandle
	,cuMemMap
	,cuMemRelease
	,cuMemSetAccess
	,cuMemImportFromShareableHandle
	,cuLaunchHostFunc
	,cuDestroyExternalSemaphore
	,cuImportExternalSemaphore
	,cuSignalExternalSemaphoresAsync
	,cuWaitExternalSemaphoresAsync
	,cuLogsRegisterCallback
);

NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(NVRTC,LibLoader,
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

struct SCUDADeviceInfo
{
	CUdevice handle = {};
	CUuuid uuid = {};
	int attributes[CU_DEVICE_ATTRIBUTE_MAX] = {};
};

struct SExportableMemoryCreationParams
{
	size_t size;
	uint32_t alignment;
	CUmemLocationType location;
};

namespace detail
{

template<typename Source>
concept program_text_source = std::same_as<std::remove_cvref_t<Source>, std::string> ||
	std::convertible_to<Source, const char*>;

}

NBL_API2 const CUDA& getCUDAFunctionTable(const CCUDAHandler& handler);
NBL_API2 const NVRTC& getNVRTCFunctionTable(const CCUDAHandler& handler);

template<typename Handler>
requires core::const_dereferenceable_to<Handler, CCUDAHandler>
inline const CUDA& getCUDAFunctionTable(Handler&& handler)
{
	return getCUDAFunctionTable(core::dereference(std::forward<Handler>(handler)));
}

template<typename Handler>
requires core::const_dereferenceable_to<Handler, CCUDAHandler>
inline const NVRTC& getNVRTCFunctionTable(Handler&& handler)
{
	return getNVRTCFunctionTable(core::dereference(std::forward<Handler>(handler)));
}

NBL_API2 bool defaultHandleResult(CUresult result, const system::logger_opt_ptr& logger);
NBL_API2 bool defaultHandleResult(const CCUDAHandler& handler, CUresult result);
NBL_API2 bool defaultHandleResult(const CCUDAHandler& handler, nvrtcResult result);

template<typename T>
T* cast_CUDA_ptr(CUdeviceptr ptr) { return reinterpret_cast<T*>(ptr); }

NBL_API2 const core::vector<SCUDADeviceInfo>& getAvailableDevices(const CCUDAHandler& handler);

template<typename Handler>
requires core::const_dereferenceable_to<Handler, CCUDAHandler>
inline const core::vector<SCUDADeviceInfo>& getAvailableDevices(Handler&& handler)
{
	return getAvailableDevices(core::dereference(std::forward<Handler>(handler)));
}

NBL_API2 nvrtcResult createProgram(CCUDAHandler& handler, nvrtcProgram* prog, std::string&& source, const char* name, const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr);
inline nvrtcResult createProgram(CCUDAHandler& handler, nvrtcProgram* prog, const char* source, const char* name, const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr)
{
	return createProgram(handler,prog,std::string(source),name,headerCount,headerContents,includeNames);
}
NBL_API2 nvrtcResult createProgram(CCUDAHandler& handler, nvrtcProgram* prog, system::IFile* file, const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr);

template<typename Handler, typename Source>
requires core::dereferenceable_to<Handler, CCUDAHandler> && detail::program_text_source<Source>
inline nvrtcResult createProgram(Handler&& handler, nvrtcProgram* prog, Source&& source, const char* name, const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr)
{
	auto& handlerRef = core::dereference(std::forward<Handler>(handler));
	if constexpr (std::same_as<std::remove_cvref_t<Source>, std::string>)
		return createProgram(handlerRef,prog,std::string(std::forward<Source>(source)),name,headerCount,headerContents,includeNames);
	else
	{
		const char* sourceText = source;
		return createProgram(handlerRef,prog,sourceText,name,headerCount,headerContents,includeNames);
	}
}

template<typename Handler, typename File>
requires core::dereferenceable_to<Handler, CCUDAHandler> && std::convertible_to<File, system::IFile*>
inline nvrtcResult createProgram(Handler&& handler, nvrtcProgram* prog, File file, const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr)
{
	return createProgram(core::dereference(std::forward<Handler>(handler)),prog,static_cast<system::IFile*>(file),headerCount,headerContents,includeNames);
}
NBL_API2 nvrtcResult compileProgram(const CCUDAHandler& handler, nvrtcProgram prog, core::SRange<const char* const> options);
NBL_API2 nvrtcResult getProgramLog(const CCUDAHandler& handler, nvrtcProgram prog, std::string& log);

struct ptx_and_nvrtcResult_t
{
	core::smart_refctd_ptr<asset::ICPUBuffer> ptx;
	nvrtcResult result;
};

NBL_API2 ptx_and_nvrtcResult_t getPTX(const CCUDAHandler& handler, nvrtcProgram prog);
NBL_API2 ptx_and_nvrtcResult_t compileDirectlyToPTX(
	CCUDAHandler& handler, std::string&& source, const char* filename, core::SRange<const char* const> nvrtcOptions,
	const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr,
	std::string* log=nullptr
);
inline ptx_and_nvrtcResult_t compileDirectlyToPTX(
	CCUDAHandler& handler, const char* source, const char* filename, core::SRange<const char* const> nvrtcOptions,
	const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr,
	std::string* log=nullptr
)
{
	return compileDirectlyToPTX(handler,std::string(source),filename,nvrtcOptions,headerCount,headerContents,includeNames,log);
}
NBL_API2 ptx_and_nvrtcResult_t compileDirectlyToPTX(
	CCUDAHandler& handler, system::IFile* file, core::SRange<const char* const> nvrtcOptions,
	const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr,
	std::string* log=nullptr
);

template<typename Handler, typename Source>
requires core::dereferenceable_to<Handler, CCUDAHandler> && detail::program_text_source<Source>
inline ptx_and_nvrtcResult_t compileDirectlyToPTX(
	Handler&& handler, Source&& source, const char* filename, core::SRange<const char* const> nvrtcOptions,
	const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr,
	std::string* log=nullptr
)
{
	auto& handlerRef = core::dereference(std::forward<Handler>(handler));
	if constexpr (std::same_as<std::remove_cvref_t<Source>, std::string>)
		return compileDirectlyToPTX(handlerRef,std::string(std::forward<Source>(source)),filename,nvrtcOptions,headerCount,headerContents,includeNames,log);
	else
	{
		const char* sourceText = source;
		return compileDirectlyToPTX(handlerRef,sourceText,filename,nvrtcOptions,headerCount,headerContents,includeNames,log);
	}
}

template<typename Handler, typename File>
requires core::dereferenceable_to<Handler, CCUDAHandler> && std::convertible_to<File, system::IFile*>
inline ptx_and_nvrtcResult_t compileDirectlyToPTX(
	Handler&& handler, File file, core::SRange<const char* const> nvrtcOptions,
	const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr,
	std::string* log=nullptr
)
{
	return compileDirectlyToPTX(core::dereference(std::forward<Handler>(handler)),static_cast<system::IFile*>(file),nvrtcOptions,headerCount,headerContents,includeNames,log);
}

NBL_API2 CUdevice getInternalObject(const CCUDADevice& device);
NBL_API2 CUcontext getContext(const CCUDADevice& device);
NBL_API2 size_t roundToGranularity(const CCUDADevice& device, CUmemLocationType location, size_t size);
NBL_API2 core::smart_refctd_ptr<CCUDAExportableMemory> createExportableMemory(CCUDADevice& device, SExportableMemoryCreationParams&& params);
NBL_API2 CUdeviceptr getDeviceptr(const CCUDAExportableMemory& memory);
NBL_API2 CUexternalMemory getInternalObject(const CCUDAImportedMemory& memory);
NBL_API2 CUresult getMappedBuffer(const CCUDAImportedMemory& memory, CUdeviceptr* mappedBuffer);
NBL_API2 CUexternalSemaphore getInternalObject(const CCUDAImportedSemaphore& semaphore);

template<typename Object>
requires (
	core::const_dereferenceable_to<Object, CCUDADevice> ||
	core::const_dereferenceable_to<Object, CCUDAImportedMemory> ||
	core::const_dereferenceable_to<Object, CCUDAImportedSemaphore>
)
inline auto getInternalObject(Object&& object)
{
	return getInternalObject(core::dereference(std::forward<Object>(object)));
}

template<typename Device>
requires core::const_dereferenceable_to<Device, CCUDADevice>
inline CUcontext getContext(Device&& device)
{
	return getContext(core::dereference(std::forward<Device>(device)));
}

template<typename Device>
requires core::const_dereferenceable_to<Device, CCUDADevice>
inline size_t roundToGranularity(Device&& device, CUmemLocationType location, size_t size)
{
	return roundToGranularity(core::dereference(std::forward<Device>(device)),location,size);
}

template<typename Device>
requires core::dereferenceable_to<Device, CCUDADevice>
inline core::smart_refctd_ptr<CCUDAExportableMemory> createExportableMemory(Device&& device, SExportableMemoryCreationParams&& params)
{
	return createExportableMemory(core::dereference(std::forward<Device>(device)),std::move(params));
}

template<typename Memory>
requires core::const_dereferenceable_to<Memory, CCUDAExportableMemory>
inline CUdeviceptr getDeviceptr(Memory&& memory)
{
	return getDeviceptr(core::dereference(std::forward<Memory>(memory)));
}

template<typename Memory>
requires core::const_dereferenceable_to<Memory, CCUDAImportedMemory>
inline CUresult getMappedBuffer(Memory&& memory, CUdeviceptr* mappedBuffer)
{
	return getMappedBuffer(core::dereference(std::forward<Memory>(memory)),mappedBuffer);
}

}

#define ASSERT_CUDA_SUCCESS(expr, handler) \
	do { \
		const auto cudaResult = (expr); \
		if (!nbl::video::cuda_native::defaultHandleResult(*(handler), cudaResult)) { \
			assert(false); \
		} \
	} while(0)

#endif
