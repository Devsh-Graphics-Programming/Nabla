// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_CUDA_INTEROP_NATIVE_H_INCLUDED_
#define _NBL_EXT_CUDA_INTEROP_NATIVE_H_INCLUDED_

#include "nbl/ext/CUDAInterop/CUDAInterop.h"

#include "nbl/asset/ICPUBuffer.h"
#include "nbl/system/DynamicFunctionCaller.h"

#include "cuda.h"
#include "nvrtc.h"
#if CUDA_VERSION < 13000
	#error "Need CUDA 13.0 SDK or higher."
#endif

namespace nbl::video::cuda_native
{

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

NBL_API2 const CUDA& getCUDAFunctionTable(const CCUDAHandler& handler);
NBL_API2 const NVRTC& getNVRTCFunctionTable(const CCUDAHandler& handler);

inline const CUDA& getCUDAFunctionTable(const CCUDAHandler* handler)
{
	return getCUDAFunctionTable(*handler);
}

inline const CUDA& getCUDAFunctionTable(const core::smart_refctd_ptr<CCUDAHandler>& handler)
{
	return getCUDAFunctionTable(*handler);
}

inline const NVRTC& getNVRTCFunctionTable(const CCUDAHandler* handler)
{
	return getNVRTCFunctionTable(*handler);
}

inline const NVRTC& getNVRTCFunctionTable(const core::smart_refctd_ptr<CCUDAHandler>& handler)
{
	return getNVRTCFunctionTable(*handler);
}

NBL_API2 bool defaultHandleResult(CUresult result, const system::logger_opt_ptr& logger);
NBL_API2 bool defaultHandleResult(const CCUDAHandler& handler, CUresult result);
NBL_API2 bool defaultHandleResult(const CCUDAHandler& handler, nvrtcResult result);

template<typename T>
T* cast_CUDA_ptr(CUdeviceptr ptr) { return reinterpret_cast<T*>(ptr); }

NBL_API2 const core::vector<SCUDADeviceInfo>& getAvailableDevices(const CCUDAHandler& handler);

inline const core::vector<SCUDADeviceInfo>& getAvailableDevices(const CCUDAHandler* handler)
{
	return getAvailableDevices(*handler);
}

inline const core::vector<SCUDADeviceInfo>& getAvailableDevices(const core::smart_refctd_ptr<CCUDAHandler>& handler)
{
	return getAvailableDevices(*handler);
}

NBL_API2 nvrtcResult createProgram(CCUDAHandler& handler, nvrtcProgram* prog, std::string&& source, const char* name, const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr);
inline nvrtcResult createProgram(CCUDAHandler& handler, nvrtcProgram* prog, const char* source, const char* name, const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr)
{
	return createProgram(handler,prog,std::string(source),name,headerCount,headerContents,includeNames);
}
NBL_API2 nvrtcResult createProgram(CCUDAHandler& handler, nvrtcProgram* prog, system::IFile* file, const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr);
inline nvrtcResult createProgram(CCUDAHandler* handler, nvrtcProgram* prog, std::string&& source, const char* name, const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr)
{
	return createProgram(*handler,prog,std::move(source),name,headerCount,headerContents,includeNames);
}
inline nvrtcResult createProgram(const core::smart_refctd_ptr<CCUDAHandler>& handler, nvrtcProgram* prog, std::string&& source, const char* name, const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr)
{
	return createProgram(*handler,prog,std::move(source),name,headerCount,headerContents,includeNames);
}
inline nvrtcResult createProgram(CCUDAHandler* handler, nvrtcProgram* prog, const char* source, const char* name, const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr)
{
	return createProgram(*handler,prog,source,name,headerCount,headerContents,includeNames);
}
inline nvrtcResult createProgram(const core::smart_refctd_ptr<CCUDAHandler>& handler, nvrtcProgram* prog, const char* source, const char* name, const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr)
{
	return createProgram(*handler,prog,source,name,headerCount,headerContents,includeNames);
}
inline nvrtcResult createProgram(CCUDAHandler* handler, nvrtcProgram* prog, system::IFile* file, const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr)
{
	return createProgram(*handler,prog,file,headerCount,headerContents,includeNames);
}
inline nvrtcResult createProgram(const core::smart_refctd_ptr<CCUDAHandler>& handler, nvrtcProgram* prog, system::IFile* file, const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr)
{
	return createProgram(*handler,prog,file,headerCount,headerContents,includeNames);
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
inline ptx_and_nvrtcResult_t compileDirectlyToPTX(
	CCUDAHandler* handler, std::string&& source, const char* filename, core::SRange<const char* const> nvrtcOptions,
	const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr,
	std::string* log=nullptr
)
{
	return compileDirectlyToPTX(*handler,std::move(source),filename,nvrtcOptions,headerCount,headerContents,includeNames,log);
}
inline ptx_and_nvrtcResult_t compileDirectlyToPTX(
	const core::smart_refctd_ptr<CCUDAHandler>& handler, std::string&& source, const char* filename, core::SRange<const char* const> nvrtcOptions,
	const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr,
	std::string* log=nullptr
)
{
	return compileDirectlyToPTX(*handler,std::move(source),filename,nvrtcOptions,headerCount,headerContents,includeNames,log);
}
inline ptx_and_nvrtcResult_t compileDirectlyToPTX(
	CCUDAHandler* handler, const char* source, const char* filename, core::SRange<const char* const> nvrtcOptions,
	const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr,
	std::string* log=nullptr
)
{
	return compileDirectlyToPTX(*handler,source,filename,nvrtcOptions,headerCount,headerContents,includeNames,log);
}
inline ptx_and_nvrtcResult_t compileDirectlyToPTX(
	const core::smart_refctd_ptr<CCUDAHandler>& handler, const char* source, const char* filename, core::SRange<const char* const> nvrtcOptions,
	const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr,
	std::string* log=nullptr
)
{
	return compileDirectlyToPTX(*handler,source,filename,nvrtcOptions,headerCount,headerContents,includeNames,log);
}
inline ptx_and_nvrtcResult_t compileDirectlyToPTX(
	CCUDAHandler* handler, system::IFile* file, core::SRange<const char* const> nvrtcOptions,
	const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr,
	std::string* log=nullptr
)
{
	return compileDirectlyToPTX(*handler,file,nvrtcOptions,headerCount,headerContents,includeNames,log);
}
inline ptx_and_nvrtcResult_t compileDirectlyToPTX(
	const core::smart_refctd_ptr<CCUDAHandler>& handler, system::IFile* file, core::SRange<const char* const> nvrtcOptions,
	const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr,
	std::string* log=nullptr
)
{
	return compileDirectlyToPTX(*handler,file,nvrtcOptions,headerCount,headerContents,includeNames,log);
}

NBL_API2 CUdevice getInternalObject(const CCUDADevice& device);
NBL_API2 CUcontext getContext(const CCUDADevice& device);
NBL_API2 size_t roundToGranularity(const CCUDADevice& device, CUmemLocationType location, size_t size);
NBL_API2 CUdeviceptr getDeviceptr(const CCUDAExportableMemory& memory);
NBL_API2 CUexternalMemory getInternalObject(const CCUDAImportedMemory& memory);
NBL_API2 CUresult getMappedBuffer(const CCUDAImportedMemory& memory, CUdeviceptr* mappedBuffer);
NBL_API2 CUexternalSemaphore getInternalObject(const CCUDAImportedSemaphore& semaphore);

inline CUdevice getInternalObject(const CCUDADevice* device)
{
	return getInternalObject(*device);
}

inline CUdevice getInternalObject(const core::smart_refctd_ptr<CCUDADevice>& device)
{
	return getInternalObject(*device);
}

inline CUcontext getContext(const CCUDADevice* device)
{
	return getContext(*device);
}

inline CUcontext getContext(const core::smart_refctd_ptr<CCUDADevice>& device)
{
	return getContext(*device);
}

inline size_t roundToGranularity(const CCUDADevice* device, CUmemLocationType location, size_t size)
{
	return roundToGranularity(*device,location,size);
}

inline size_t roundToGranularity(const core::smart_refctd_ptr<CCUDADevice>& device, CUmemLocationType location, size_t size)
{
	return roundToGranularity(*device,location,size);
}

inline CUdeviceptr getDeviceptr(const CCUDAExportableMemory* memory)
{
	return getDeviceptr(*memory);
}

inline CUdeviceptr getDeviceptr(const core::smart_refctd_ptr<CCUDAExportableMemory>& memory)
{
	return getDeviceptr(*memory);
}

inline CUexternalMemory getInternalObject(const CCUDAImportedMemory* memory)
{
	return getInternalObject(*memory);
}

inline CUexternalMemory getInternalObject(const core::smart_refctd_ptr<CCUDAImportedMemory>& memory)
{
	return getInternalObject(*memory);
}

inline CUresult getMappedBuffer(const CCUDAImportedMemory* memory, CUdeviceptr* mappedBuffer)
{
	return getMappedBuffer(*memory,mappedBuffer);
}

inline CUresult getMappedBuffer(const core::smart_refctd_ptr<CCUDAImportedMemory>& memory, CUdeviceptr* mappedBuffer)
{
	return getMappedBuffer(*memory,mappedBuffer);
}

inline CUexternalSemaphore getInternalObject(const CCUDAImportedSemaphore* semaphore)
{
	return getInternalObject(*semaphore);
}

inline CUexternalSemaphore getInternalObject(const core::smart_refctd_ptr<CCUDAImportedSemaphore>& semaphore)
{
	return getInternalObject(*semaphore);
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
