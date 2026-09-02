// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_CUDA_INTEROP_HANDLES_H_INCLUDED_
#define _NBL_VIDEO_CUDA_INTEROP_HANDLES_H_INCLUDED_

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace nbl::video::cuda_interop
{

/*
	SDK-free CUDA interop boundary.

	Public nbl/video/CCUDA*.h headers cannot include cuda.h or nvrtc.h, but they still need to carry CUDA interop
	state and write CUDA/NVRTC handles for opt-in users. The split below keeps those two roles explicit:
	- SOpaqueCUDAHandle owns handle bits and is used in Nabla object layout, parameters, and return values.
	- SOutput is a non-owning output adapter. C++ does not apply user-defined conversions through T* or mutable T&,
	  so output parameters need a small bridge to write directly into either SCU* storage or native SDK storage.

	CUDAInteropNative.h is the only header that maps these opaque types back to CUDA/NVRTC SDK types. These helpers
	are class templates with in-class member definitions, so they are inline by the language rules and add no exported
	symbols.
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
	std::same_as<std::remove_cvref_t<Native>,typename SOpaqueCUDANativeType<Opaque>::type> &&
	cuda_opaque_handle<Opaque,std::remove_cvref_t<Native>>;

/*
	Non-owning output bridge for SDK-free APIs. It keeps one Nabla signature while opt-in callers can pass raw
	CUDA/NVRTC output variables directly, e.g. `CUdeviceptr ptr; memory->getMappedBuffer(ptr);`.
*/
template<typename Opaque>
struct SOutput
{
	SOutput(std::nullptr_t) : ptr(nullptr) {}
	SOutput(Opaque& opaque) : ptr(&opaque) {}
	SOutput(Opaque* opaque) : ptr(opaque) {}

	template<typename Native>
	requires cuda_native_handle_for<Opaque,Native>
	SOutput(Native& native) : ptr(reinterpret_cast<Opaque*>(&native)) {}

	template<typename Native>
	requires cuda_native_handle_for<Opaque,Native>
	SOutput(Native* native) : ptr(reinterpret_cast<Opaque*>(native)) {}

	Opaque& operator*() const { return *ptr; }
	operator Opaque*() const { return ptr; }
	explicit operator bool() const { return ptr!=nullptr; }

	private:
		Opaque* ptr;
};

/*
	Owned opaque value used in public Nabla ABI. Native reference conversions become available only after the opt-in
	header specializes SOpaqueCUDANativeType for the selected CUDA SDK.
*/
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
	operator Native&()
	{
		return *reinterpret_cast<Native*>(value);
	}

	template<typename Native>
	requires cuda_native_handle_for<Derived,Native>
	operator const Native&() const
	{
		return *reinterpret_cast<const Native*>(value);
	}

	template<typename Native>
	requires cuda_native_handle_for<Derived,Native>
	Derived& operator=(const Native& native)
	{
		static_cast<Native&>(*this) = native;
		return static_cast<Derived&>(*this);
	}

	template<typename Native>
	requires cuda_native_handle_for<Derived,Native>
	friend bool operator==(const Derived& lhs, const Native& rhs)
	{
		return static_cast<const Native&>(lhs)==rhs;
	}

	template<typename Native>
	requires cuda_native_handle_for<Derived,Native>
	friend bool operator==(const Native& lhs, const Derived& rhs)
	{
		return lhs==static_cast<const Native&>(rhs);
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
