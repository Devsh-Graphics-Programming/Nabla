// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
/*
	CUDA SDK opt-in boundary for Nabla CUDA interop.

	Public nbl/video CUDA interop headers expose SDK-free cuda_interop::SCU* opaque handles. This header is the
	explicit boundary where a consumer accepts CUDA/NVRTC SDK headers, raw CU* types, and Nabla helper APIs whose
	signatures use CUDA SDK types. This happens by linking Nabla::ext::CUDAInterop and including this file, which
	includes cuda.h and nvrtc.h. The CUDA SDK becomes a compile-time requirement only for that SDK opt-in
	consumer.

	The exported definitions stay in Nabla because they are glue between the Nabla world and the CUDA world:
	dynamic Driver API/NVRTC loader access, NVRTC program helpers, error handling, runtime header discovery, and
	CUDA/Vulkan resource interop lifetime. This header only exposes the CUDA-typed signatures for that glue after
	the consumer explicitly opts in. Nabla::ext::CUDAInterop is the build-system edge for this SDK-typed surface.
	It is not a separate owner of these definitions. Code that only consumes Nabla::Nabla does not need CUDA SDK
	headers and does not parse CUDA/NVRTC declarations.

	Keeping SDK-defined types out of Nabla's public ABI is intentional. CUDA headers have changed observable
	compile-time constants across SDK versions:
	- CUDA Toolkit 9.0 documented CU_CTX_FLAGS_MASK as 0x1f. CUDA 12.1, 12.5, and 13.2 define it as 0xff.
	- CUDA Toolkit 9.0 documented CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS as 93. CUDA 12.1, 12.5,
	  and 13.2 keep 93 as CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1 and define the unsuffixed name
	  as 122.
	- CUDA Toolkit 9.0 documented CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR as 94. CUDA 12.1, 12.5,
	  and 13.2 keep 94 as CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1 and define the unsuffixed name
	  as 123.

	If these SDK declarations leak through public Nabla headers, consumers can silently compile against a
	different CUDA interpretation than the one used to build the interop implementation. That is especially
	problematic for installed packages, plugins, and separately built downstream projects. The opaque handles
	keep Nabla's public ABI independent from CUDA SDK headers. This opt-in header then validates handle
	size/alignment against the SDK selected by the SDK opt-in consumer.
*/
#ifndef _NBL_EXT_CUDA_INTEROP_NATIVE_H_INCLUDED_
#define _NBL_EXT_CUDA_INTEROP_NATIVE_H_INCLUDED_
#include <string>
#include "nbl/video/CUDAInterop.h"
#include "nbl/asset/ICPUBuffer.h"
#include "nbl/system/DynamicFunctionCaller.h"

#include "cuda.h"
#include "nvrtc.h"
#include <cassert>
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
/*
	Map Nabla opaque handles to CUDA SDK handle types.

	This is deliberately small. It is not an attempt to wrap CUDA. It only gives SDK opt-in code a convenient
	way to pass Nabla-owned opaque handles to CUDA C APIs while checking that the public opaque type has the same
	layout as the CUDA type visible in this translation unit. If a future SDK changes one of these handle layouts,
	the SDK opt-in build fails here instead of letting ABI drift propagate through packaged Nabla headers.
*/
template<typename Opaque>
struct SOpaqueCUDAType;

template<> struct SOpaqueCUDAType<cuda_interop::SCUdevice> { using type = CUdevice; };
template<> struct SOpaqueCUDAType<cuda_interop::SCUcontext> { using type = CUcontext; };
template<> struct SOpaqueCUDAType<cuda_interop::SCUdeviceptr> { using type = CUdeviceptr; };
template<> struct SOpaqueCUDAType<cuda_interop::SCUexternalMemory> { using type = CUexternalMemory; };
template<> struct SOpaqueCUDAType<cuda_interop::SCUexternalSemaphore> { using type = CUexternalSemaphore; };
/*
	CUDA SDK view of an SDK-free opaque handle.

	The conversions are intentionally available only after including this header. Public Nabla headers expose
	only the opaque SCU* values. Once a consumer opts in, SNativeHandle restores the CUDA spelling and ergonomics
	for raw Driver API calls without adding accessors to every interop operation.
*/
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
/*
	Nabla interop API declarations with CUDA SDK signatures.

	These declarations belong to the Nabla interop API. They live behind Nabla::ext::CUDAInterop because their
	signatures mention CUDA/NVRTC SDK types directly. Keeping them out of nbl/video/CCUDA*.h means Nabla's public
	API can be parsed and packaged without CUDA SDK headers. Nabla still owns the exported glue definitions.
	Consumers accept this SDK-typed API surface only by including this header and linking the explicit interop
	target.
*/
NBL_API2 bool defaultHandleResult(CUresult result, const system::logger_opt_ptr& logger);
NBL_API2 bool defaultHandleResult(const CCUDAHandler& handler, CUresult result);
NBL_API2 bool defaultHandleResult(const CCUDAHandler& handler, nvrtcResult result);
#define NBL_CUDA_INTEROP_ASSERT_SUCCESS(expr, handler) \
	do { \
		const auto nblCudaInteropResult = (expr); \
		if (!::nbl::video::cuda_native::defaultHandleResult(*(handler), nblCudaInteropResult)) \
			assert(false); \
	} while(0)
NBL_API2 nvrtcResult createProgram(CCUDAHandler& handler, nvrtcProgram* prog, std::string&& source, const char* name, const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr);
NBL_API2 nvrtcResult compileProgram(const CCUDAHandler& handler, nvrtcProgram prog, core::SRange<const char* const> options);
NBL_API2 nvrtcResult getProgramLog(const CCUDAHandler& handler, nvrtcProgram prog, std::string& log);
NBL_API2 SPTXResult getPTX(const CCUDAHandler& handler, nvrtcProgram prog);
NBL_API2 SPTXResult compileDirectlyToPTX(
	CCUDAHandler& handler, std::string&& source, const char* filename, core::SRange<const char* const> nvrtcOptions,
	std::string& log, const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr
);

}

#endif
