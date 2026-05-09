// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_CUDA_INTEROP_HANDLES_H_INCLUDED_
#define _NBL_VIDEO_CUDA_INTEROP_HANDLES_H_INCLUDED_

#include <cstddef>
#include <cstdint>

namespace nbl::video::cuda_interop
{

/*
	SDK-free CUDA handle surrogates used by Nabla's public video API.

	These types are the small glue layer between Nabla and SDK-typed CUDA interop code. They let nbl/video/CCUDA*.h
	expose CUDA-related objects without including cuda.h or nvrtc.h, so consumers that only link Nabla::Nabla do
	not inherit CUDA SDK as a public compile-time dependency. CUDAInteropNative.h maps these opaque handles back
	to the real CU* types and checks their size/alignment against the SDK selected by the opt-in consumer.
*/
template<typename Storage>
struct alignas(alignof(Storage)) SOpaqueCUDAHandle
{
	uint8_t value[sizeof(Storage)] = {};
};

struct SCUdevice : SOpaqueCUDAHandle<int32_t> {};
struct SCUcontext : SOpaqueCUDAHandle<void*> {};
struct SCUdeviceptr : SOpaqueCUDAHandle<uintptr_t> {};
struct SCUexternalMemory : SOpaqueCUDAHandle<void*> {};
struct SCUexternalSemaphore : SOpaqueCUDAHandle<void*> {};

}

#endif
