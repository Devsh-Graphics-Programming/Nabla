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

const CUDA& getCUDAFunctionTable(const CCUDAHandler& handler);
const NVRTC& getNVRTCFunctionTable(const CCUDAHandler& handler);

bool defaultHandleResult(CUresult result, const system::logger_opt_ptr& logger);
bool defaultHandleResult(const CCUDAHandler& handler, CUresult result);
bool defaultHandleResult(const CCUDAHandler& handler, nvrtcResult result);

template<typename T>
T* cast_CUDA_ptr(CUdeviceptr ptr) { return reinterpret_cast<T*>(ptr); }

const core::vector<SCUDADeviceInfo>& getAvailableDevices(const CCUDAHandler& handler);

nvrtcResult createProgram(CCUDAHandler& handler, nvrtcProgram* prog, std::string&& source, const char* name, const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr);
inline nvrtcResult createProgram(CCUDAHandler& handler, nvrtcProgram* prog, const char* source, const char* name, const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr)
{
	return createProgram(handler,prog,std::string(source),name,headerCount,headerContents,includeNames);
}
nvrtcResult createProgram(CCUDAHandler& handler, nvrtcProgram* prog, system::IFile* file, const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr);
nvrtcResult compileProgram(const CCUDAHandler& handler, nvrtcProgram prog, core::SRange<const char* const> options);
nvrtcResult getProgramLog(const CCUDAHandler& handler, nvrtcProgram prog, std::string& log);

struct ptx_and_nvrtcResult_t
{
	core::smart_refctd_ptr<asset::ICPUBuffer> ptx;
	nvrtcResult result;
};

ptx_and_nvrtcResult_t getPTX(const CCUDAHandler& handler, nvrtcProgram prog);
ptx_and_nvrtcResult_t compileDirectlyToPTX(
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
ptx_and_nvrtcResult_t compileDirectlyToPTX(
	CCUDAHandler& handler, system::IFile* file, core::SRange<const char* const> nvrtcOptions,
	const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr,
	std::string* log=nullptr
);

CUdevice getInternalObject(const CCUDADevice& device);
CUcontext getContext(const CCUDADevice& device);
size_t roundToGranularity(const CCUDADevice& device, CUmemLocationType location, size_t size);
CUdeviceptr getDeviceptr(const CCUDAExportableMemory& memory);
CUexternalMemory getInternalObject(const CCUDAImportedMemory& memory);
CUresult getMappedBuffer(const CCUDAImportedMemory& memory, CUdeviceptr* mappedBuffer);
CUexternalSemaphore getInternalObject(const CCUDAImportedSemaphore& semaphore);

}

#define ASSERT_CUDA_SUCCESS(expr, handler) \
	do { \
		const auto cudaResult = (expr); \
		if (!nbl::video::cuda_native::defaultHandleResult(*(handler), cudaResult)) { \
			assert(false); \
		} \
	} while(0)

#endif
