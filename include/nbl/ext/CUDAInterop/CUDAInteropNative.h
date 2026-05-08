// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXT_CUDA_INTEROP_NATIVE_H_INCLUDED_
#define _NBL_EXT_CUDA_INTEROP_NATIVE_H_INCLUDED_

#include "nbl/video/CUDAInterop.h"

#include "nbl/asset/ICPUBuffer.h"
#include "nbl/system/DynamicFunctionCaller.h"

#include <string>

#include "cuda.h"
#include "nvrtc.h"
#include <type_traits>
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
};

struct SExportableMemoryCreationParams
{
	size_t size;
	uint32_t alignment;
	CUmemLocationType location;
};

struct SPTXResult
{
	core::smart_refctd_ptr<asset::ICPUBuffer> ptx;
	nvrtcResult result;
};

template<typename Opaque, typename Native>
concept cuda_opaque_handle =
	std::is_trivially_copyable_v<Opaque> &&
	std::is_trivially_copyable_v<Native> &&
	sizeof(Opaque)==sizeof(Native) &&
	alignof(Opaque)==alignof(Native);

template<typename Opaque>
struct SOpaqueCUDAType;

template<> struct SOpaqueCUDAType<cuda_interop::SCUdevice> { using type = CUdevice; };
template<> struct SOpaqueCUDAType<cuda_interop::SCUcontext> { using type = CUcontext; };
template<> struct SOpaqueCUDAType<cuda_interop::SCUdeviceptr> { using type = CUdeviceptr; };
template<> struct SOpaqueCUDAType<cuda_interop::SCUexternalMemory> { using type = CUexternalMemory; };
template<> struct SOpaqueCUDAType<cuda_interop::SCUexternalSemaphore> { using type = CUexternalSemaphore; };

template<typename Opaque>
struct SNativeHandle
{
	using cuda_t = typename SOpaqueCUDAType<Opaque>::type;
	static_assert(cuda_opaque_handle<Opaque,cuda_t>);

	SNativeHandle() = default;
	SNativeHandle(const SNativeHandle&) = default;
	SNativeHandle(const cuda_t& native) { operator=(native); }
	SNativeHandle(const Opaque& opaque) { operator=(opaque); }

	SNativeHandle& operator=(const SNativeHandle&) = default;
	SNativeHandle& operator=(const cuda_t& native) { value = native; return *this; }
	SNativeHandle& operator=(const Opaque& opaque) { operator Opaque&() = opaque; return *this; }

	operator cuda_t&() { return value; }
	operator const cuda_t&() const { return value; }
	operator Opaque&() { return reinterpret_cast<Opaque&>(value); }
	operator const Opaque&() const { return reinterpret_cast<const Opaque&>(value); }

	Opaque* opaque() { return &static_cast<Opaque&>(*this); }
	const Opaque* opaque() const { return &static_cast<const Opaque&>(*this); }
	Opaque asOpaque() const { return static_cast<const Opaque&>(*this); }

	cuda_t value = {};
};

using SCUdevice = SNativeHandle<cuda_interop::SCUdevice>;
using SCUcontext = SNativeHandle<cuda_interop::SCUcontext>;
using SCUdeviceptr = SNativeHandle<cuda_interop::SCUdeviceptr>;
using SCUexternalMemory = SNativeHandle<cuda_interop::SCUexternalMemory>;
using SCUexternalSemaphore = SNativeHandle<cuda_interop::SCUexternalSemaphore>;

inline bool isBuildCUDAVersionCompatible()
{
	const auto buildVersion = CCUDAHandler::getBuildCUDAVersion();
	return buildVersion==0u || buildVersion==CUDA_VERSION;
}

inline bool isDeviceLocal(CUmemLocationType location)
{
	return location==CU_MEM_LOCATION_TYPE_DEVICE;
}

// Opt-in native CUDA declarations. Nabla owns the definitions.
NBL_API2 const CUDA& getCUDAFunctionTable(const CCUDAHandler& handler);
NBL_API2 const NVRTC& getNVRTCFunctionTable(const CCUDAHandler& handler);
NBL_API2 bool defaultHandleResult(CUresult result, const system::logger_opt_ptr& logger);
NBL_API2 bool defaultHandleResult(const CCUDAHandler& handler, CUresult result);
NBL_API2 bool defaultHandleResult(const CCUDAHandler& handler, nvrtcResult result);
NBL_API2 const core::vector<SCUDADeviceInfo>& getAvailableDevices(const CCUDAHandler& handler);
NBL_API2 nvrtcResult createProgram(CCUDAHandler& handler, nvrtcProgram* prog, std::string&& source, const char* name, const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr);
NBL_API2 nvrtcResult compileProgram(const CCUDAHandler& handler, nvrtcProgram prog, core::SRange<const char* const> options);
NBL_API2 nvrtcResult getProgramLog(const CCUDAHandler& handler, nvrtcProgram prog, std::string& log);
NBL_API2 SPTXResult getPTX(const CCUDAHandler& handler, nvrtcProgram prog);
NBL_API2 SPTXResult compileDirectlyToPTX(
	CCUDAHandler& handler, std::string&& source, const char* filename, core::SRange<const char* const> nvrtcOptions,
	std::string& log, const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr
);

inline size_t roundToGranularity(const CCUDADevice& device, CUmemLocationType location, size_t size)
{
	return device.roundToGranularity(static_cast<uint32_t>(location),size);
}

inline core::smart_refctd_ptr<CCUDAExportableMemory> createExportableMemory(CCUDADevice& device, SExportableMemoryCreationParams&& params)
{
	return device.createExportableMemory({params.size,params.alignment,static_cast<uint32_t>(params.location)});
}

}

#endif
