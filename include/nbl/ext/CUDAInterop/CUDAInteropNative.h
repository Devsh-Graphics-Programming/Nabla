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
#include "nbl/video/CUDAInteropNativeAPI.h"
namespace nbl::video::cuda_native
{

/*
	This header specializes the SDK-free opaque handles from nbl/video/CUDAInteropHandles.h for the CUDA SDK
	visible to this translation unit. After that opt-in, Nabla interop methods can be called with native CUDA/NVRTC
	types such as CUdeviceptr, CUexternalSemaphore, nvrtcProgram, CUresult, and nvrtcResult.

	The size/alignment checks live in nbl/video/CUDAInteropNativeAPI.h. This exact version check is a policy helper
	for SDK-typed code that wants to warn about or reject compatible-but-different SDK headers.
*/
inline bool isBuildCUDASDKVersionExactMatch()
{
	const auto buildVersion = CCUDAHandler::getBuildCUDASDKVersion();
	return buildVersion==CUDA_VERSION;
}

}

#endif
