// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

// TODO: Remove all appearances of this class and refactor to use the real BDA once https://github.com/microsoft/DirectXShaderCompiler/issues/6541 is resolved
//       Then, delete this file altogether.


#ifndef _NBL_BUILTIN_HLSL_LEGACY_BDA_ACCESSOR_INCLUDED_
#define _NBL_BUILTIN_HLSL_LEGACY_BDA_ACCESSOR_INCLUDED_

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/bda/__ptr.hlsl"


namespace nbl 
{
namespace hlsl
{

namespace impl {

struct LegacyBdaAccessorBase
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
struct LegacyBdaAccessor : impl::LegacyBdaAccessorBase
{
    using type_t = T;
    static LegacyBdaAccessor<T> create(const uint64_t address)
    {
        LegacyBdaAccessor<T> accessor;
        accessor.address = address;
        return accessor;
    }

    T get(const uint64_t index)
    {
        return vk::RawBufferLoad<T>(address + index * sizeof(T));
    }

    void get(const uint64_t index, NBL_REF_ARG(T) value)
    {
        value = vk::RawBufferLoad<T>(address + index * sizeof(T));
    }

    void set(const uint64_t index, const T value)
    {
        vk::RawBufferStore<T>(address + index * sizeof(T), value);
    }

    uint64_t address;
};

template<typename T>
struct DoubleLegacyBdaAccessor : impl::LegacyBdaAccessorBase
{
    using type_t = T;
    static DoubleLegacyBdaAccessor<T> create(const uint64_t inputAddress, const uint64_t outputAddress)
    {
        DoubleLegacyBdaAccessor<T> accessor;
        accessor.inputAddress = inputAddress;
        accessor.outputAddress = outputAddress;
        return accessor;
    }

    T get(const uint64_t index)
    {
        return vk::RawBufferLoad<T>(inputAddress + index * sizeof(T));
    }

    void get(const uint64_t index, NBL_REF_ARG(T) value)
    {
        value = vk::RawBufferLoad<T>(inputAddress + index * sizeof(T));
    }

    void set(const uint64_t index, const T value)
    {
        vk::RawBufferStore<T>(outputAddress + index * sizeof(T), value);
    }

    uint64_t inputAddress, outputAddress;
};


}
}

#endif
