// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_CUDA_INTEROP_HANDLES_H_INCLUDED_
#define _NBL_VIDEO_CUDA_INTEROP_HANDLES_H_INCLUDED_

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
template<typename Opaque>
struct SOpaqueCUDANativeType;

template<typename Opaque, typename Native>
concept cuda_opaque_handle =
	std::is_trivially_copyable_v<Opaque> &&
	std::is_trivially_copyable_v<Native> &&
	sizeof(Opaque)==sizeof(Native) &&
	alignof(Opaque)==alignof(Native);

template<typename Opaque, typename Native>
concept cuda_native_handle_for =
	requires { typename SOpaqueCUDANativeType<Opaque>::type; } &&
	std::same_as<std::remove_cv_t<Native>,typename SOpaqueCUDANativeType<Opaque>::type> &&
	cuda_opaque_handle<Opaque,std::remove_cv_t<Native>>;

template<typename Derived, typename Storage>
struct alignas(alignof(Storage)) SOpaqueCUDAHandle
{
	uint8_t value[sizeof(Storage)] = {};

	SOpaqueCUDAHandle() = default;

	template<typename Native>
	requires cuda_native_handle_for<Derived,Native>
	SOpaqueCUDAHandle(const Native& native)
	{
		operator=(native);
	}

	template<typename Native>
	requires cuda_native_handle_for<Derived,Native>
	Derived& operator=(const Native& native)
	{
		std::memcpy(value,&native,sizeof(native));
		return static_cast<Derived&>(*this);
	}

	template<typename Native>
	requires cuda_native_handle_for<Derived,Native>
	operator Native() const
	{
		Native native = {};
		std::memcpy(&native,value,sizeof(native));
		return native;
	}

	template<typename Native>
	requires cuda_native_handle_for<Derived,Native>
	friend bool operator==(const Derived& lhs, const Native& rhs)
	{
		return static_cast<Native>(lhs)==rhs;
	}

	template<typename Native>
	requires cuda_native_handle_for<Derived,Native>
	friend bool operator==(const Native& lhs, const Derived& rhs)
	{
		return lhs==static_cast<Native>(rhs);
	}

	template<typename Native>
	requires cuda_native_handle_for<Derived,Native>
	friend bool operator!=(const Derived& lhs, const Native& rhs)
	{
		return !(lhs==rhs);
	}

	template<typename Native>
	requires cuda_native_handle_for<Derived,Native>
	friend bool operator!=(const Native& lhs, const Derived& rhs)
	{
		return !(lhs==rhs);
	}
};

#define NBL_CUDA_INTEROP_DECLARE_OPAQUE_HANDLE(NAME, STORAGE) \
	struct NAME : SOpaqueCUDAHandle<NAME,STORAGE> \
	{ \
		using SOpaqueCUDAHandle<NAME,STORAGE>::SOpaqueCUDAHandle; \
		using SOpaqueCUDAHandle<NAME,STORAGE>::operator=; \
	}

NBL_CUDA_INTEROP_DECLARE_OPAQUE_HANDLE(SCUdevice, int32_t);
NBL_CUDA_INTEROP_DECLARE_OPAQUE_HANDLE(SCUcontext, void*);
NBL_CUDA_INTEROP_DECLARE_OPAQUE_HANDLE(SCUdeviceptr, uintptr_t);
NBL_CUDA_INTEROP_DECLARE_OPAQUE_HANDLE(SCUexternalMemory, void*);
NBL_CUDA_INTEROP_DECLARE_OPAQUE_HANDLE(SCUexternalSemaphore, void*);
NBL_CUDA_INTEROP_DECLARE_OPAQUE_HANDLE(SCUresult, int32_t);
NBL_CUDA_INTEROP_DECLARE_OPAQUE_HANDLE(SNVRTCResult, int32_t);
NBL_CUDA_INTEROP_DECLARE_OPAQUE_HANDLE(SNVRTCProgram, void*);

#undef NBL_CUDA_INTEROP_DECLARE_OPAQUE_HANDLE

}

#endif
