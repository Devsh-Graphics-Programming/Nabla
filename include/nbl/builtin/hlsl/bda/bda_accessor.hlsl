// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BDA_ACCESSOR_INCLUDED_
#define _NBL_BUILTIN_HLSL_BDA_ACCESSOR_INCLUDED_

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/bda/__ptr.hlsl"

namespace nbl 
{
namespace hlsl
{

namespace impl {

struct BdaAccessorBase
{
    // Note: Its a funny quirk of the SPIR-V Vulkan Env spec that `MemorySemanticsUniformMemoryMask` means SSBO as well :facepalm: (and probably BDA)
	void workgroupExecutionAndMemoryBarrier() 
	{
		// we're only barriering the workgroup and trading memory within a workgroup
		spirv::controlBarrier(spv::ScopeWorkgroup, spv::ScopeWorkgroup, spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsUniformMemoryMask);
    }

    void memoryBarrier() 
	{
		// By default it's device-wide access to the buffer
		spirv::memoryBarrier(spv::ScopeDevice, spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsUniformMemoryMask);
	}
};

} //namespace impl

template<typename T>
struct BdaAccessor : impl::BdaAccessorBase
{
    using type_t = T;
    static BdaAccessor<T> create(const bda::__ptr<T> ptr)
    {
        BdaAccessor<T> accessor;
        accessor.ptr = ptr;
        return accessor;
    }

    T get(const uint64_t index)
    {
        bda::__ptr<T> target = ptr + index;
        return target.template deref().load();
    }

    void get(const uint64_t index, NBL_REF_ARG(T) value)
    {
        bda::__ptr<T> target = ptr + index;
        value = target.template deref().load();
    }

    void set(const uint64_t index, const T value)
    {
        bda::__ptr<T> target = ptr + index;
        return target.template deref().store(value);
    }

    template<typename S = T>
    enable_if_t<is_same_v<S,T> && is_integral<T>::value && (sizeof(T) == 4 || sizeof(T) == 8), T>
    atomicAdd(const uint64_t index, const T value)
    {
        bda::__ptr<T> target = ptr + index;
        return glsl::atomicAdd(target.template deref().get_ptr(), value);
    }

    template<typename S = T>
    enable_if_t<is_same_v<S,T> && is_integral<T>::value && (sizeof(T) == 4 || sizeof(T) == 8), T>
    atomicSub(const uint64_t index, const T value)
    {
        bda::__ptr<T> target = ptr + index;
        return glsl::atomicSub(target.template deref().get_ptr(), value);
    }

    bda::__ptr<T> ptr;
};

template<typename T>
struct DoubleBdaAccessor : impl::BdaAccessorBase
{
    using type_t = T;
    static DoubleBdaAccessor<T> create(const bda::__ptr<T> inputPtr, const bda::__ptr<T> outputPtr)
    {
        DoubleBdaAccessor<T> accessor;
        accessor.inputPtr = inputPtr;
        accessor.outputPtr = outputPtr;
        return accessor;
    }

    T get(const uint64_t index)
    {
        bda::__ptr<T> target = inputPtr + index;
        return target.template deref().load();
    }

    void get(const uint64_t index, NBL_REF_ARG(T) value)
    {
        bda::__ptr<T> target = inputPtr + index;
        value = target.template deref().load();
    }

    void set(const uint64_t index, const T value)
    {
        bda::__ptr<T> target = outputPtr + index;
        return target.template deref().store(value);
    }

    bda::__ptr<T> inputPtr, outputPtr;
};


}
}

#endif
