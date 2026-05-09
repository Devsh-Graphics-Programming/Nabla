// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_CUDA_INTEROP_HANDLES_H_INCLUDED_
#define _NBL_VIDEO_CUDA_INTEROP_HANDLES_H_INCLUDED_

#include <cstddef>
#include <cstdint>
#include <type_traits>

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

template<typename Opaque, typename Native>
concept cuda_opaque_handle =
	std::is_trivially_copyable_v<Opaque> &&
	std::is_trivially_copyable_v<Native> &&
	sizeof(Opaque)==sizeof(Native) &&
	alignof(Opaque)==alignof(Native);

/*
	Native view of an SDK-free opaque handle.

	This template does not depend on CUDA SDK types by itself. CUDAInteropNative.h binds it to concrete CU* types
	after the consumer opts into CUDA SDK headers. The layout check keeps the public opaque handle and the native
	SDK handle compatible in that translation unit while preserving Nabla's SDK-free public headers.
*/
template<typename Opaque, typename Native>
struct SNativeHandle
{
	using cuda_t = Native;
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

}

#endif
