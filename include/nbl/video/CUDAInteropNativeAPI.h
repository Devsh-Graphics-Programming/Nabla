// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_CUDA_INTEROP_NATIVE_API_H_INCLUDED_
#define _NBL_VIDEO_CUDA_INTEROP_NATIVE_API_H_INCLUDED_

#include <cassert>
#include <string>

#include "nbl/video/CUDAInterop.h"
#include "nbl/asset/ICPUBuffer.h"
#include "nbl/system/DynamicFunctionCaller.h"

#include "cuda.h"
#include "nvrtc.h"

namespace nbl::video::cuda_native
{

inline constexpr int MinimumCUDADriverVersion = 13000;
inline constexpr int MinimumNVRTCMajorVersion = MinimumCUDADriverVersion/1000;
static_assert(CUDA_VERSION >= MinimumCUDADriverVersion, "Need CUDA 13.0 SDK or higher.");

/*
	Low-level CUDA SDK boundary shared by Nabla's CUDA implementation and explicit CUDA interop opt-in users.

	This file lives under include/ because it is shared with nbl/ext/CUDAInterop/CUDAInteropNative.h, the public
	opt-in header for consumers that explicitly accept CUDA SDK types. Its physical location does not make it part
	of the default Nabla public interface: nbl/video/CCUDA*.h headers, Nabla::Nabla public requirements, and PCH
	do not include it, so normal Nabla consumers do not need cuda.h or nvrtc.h.

	The declarations below intentionally use CUDA/NVRTC SDK types because they describe the SDK-typed glue between
	raw CUDA code and Nabla's exported CUDA interop objects: dynamic function tables, NVRTC helpers, error handling,
	and runtime header discovery integration. Consumers enter this surface only by linking Nabla::ext::CUDAInterop
	and including nbl/ext/CUDAInterop/CUDAInteropNative.h.
*/
using LibLoader = system::DefaultFuncPtrLoader;

/*
	The CUDA/NVRTC table classes contain the calls used and tested by Nabla's interop implementation. SDK opt-in
	consumers can load additional Driver API or NVRTC symbols from the same table without changing Nabla's ABI:

	auto pcuNewCall = NBL_SYSTEM_LOAD_DYNLIB_FUNCPTR(handler->getCUDAFunctionTable(), cuNewCall);

	The requested symbol must be declared by the CUDA SDK visible to that translation unit because the helper uses
	decltype(cuNewCall) to preserve the native function signature.
*/
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

struct SPTXResult
{
	core::smart_refctd_ptr<asset::ICPUBuffer> ptx;
	nvrtcResult result;
};

/*
	Exported Nabla glue declarations with CUDA SDK signatures.

	These are not a CUDA wrapper. They are the small boundary surface used for error handling, NVRTC helpers,
	runtime header discovery integration, and dynamic CUDA/NVRTC table access. Nabla owns the definitions.
	The signatures mention CUDA SDK types, so they are intentionally unavailable to consumers that only parse
	SDK-free nbl/video/CCUDA*.h headers.
*/
NBL_API2 bool defaultHandleResult(CUresult result, const system::logger_opt_ptr& logger);
NBL_API2 bool defaultHandleResult(const CCUDAHandler& handler, CUresult result);
NBL_API2 bool defaultHandleResult(const CCUDAHandler& handler, nvrtcResult result);
NBL_API2 nvrtcResult createProgram(CCUDAHandler& handler, nvrtcProgram* prog, std::string&& source, const char* name, const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr);
NBL_API2 nvrtcResult compileProgram(const CCUDAHandler& handler, nvrtcProgram prog, core::SRange<const char* const> options);
NBL_API2 nvrtcResult getProgramLog(const CCUDAHandler& handler, nvrtcProgram prog, std::string& log);
NBL_API2 SPTXResult getPTX(const CCUDAHandler& handler, nvrtcProgram prog);
NBL_API2 SPTXResult compileDirectlyToPTX(
	CCUDAHandler& handler, std::string&& source, const char* filename, core::SRange<const char* const> nvrtcOptions,
	std::string& log, const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr
);

#define NBL_CUDA_INTEROP_ASSERT_SUCCESS(expr, handler) \
	do { \
		const auto nblCudaInteropResult = (expr); \
		if (!::nbl::video::cuda_native::defaultHandleResult((handler),nblCudaInteropResult)) \
			assert(false); \
	} while (false)

}

#endif
