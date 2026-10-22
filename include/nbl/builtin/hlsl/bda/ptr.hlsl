// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BDA_PTR_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_BDA_PTR_HLSL_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat/matrix.hlsl"
#include "nbl/builtin/hlsl/type_traits.hlsl"

#ifndef __HLSL_VERSION
#include <cassert>
#include <concepts>
#endif

namespace nbl
{
namespace hlsl
{
namespace bda
{

namespace impl
{
// no `const` pointee: DXC silently drops stores through `vk::BufferPointer<const T>`
template<typename T>
struct pointee
{
    using type = T;
};
template<typename T>
struct pointee<const T> {};
}

#ifdef __HLSL_VERSION
// https://github.com/microsoft/hlsl-specs/blob/main/proposals/0010-vk-buffer-ref.md#buffer-pointers-and-aliasing
template<typename T, uint32_t Alignment=alignment_of<T>::value>
using ptr = vk::BufferPointer<typename impl::pointee<T>::type,Alignment>;
#else
template<typename T, uint32_t Alignment=alignment_of<T>::value>
struct ptr
{
    using pointee_t = typename impl::pointee<T>::type;

    uint64_t address;

    ptr() = default;
    constexpr explicit ptr(const uint64_t _address) : address(_address)
    {
        // https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#VUID-RuntimeSpirv-PhysicalStorageBuffer64-06315
        assert(Alignment==0u || (_address%Alignment)==0ull);
    }

    constexpr explicit operator uint64_t() const {return address;}

    // integers scroll by `sizeof(T)*arg`, byte offsets go through `offsetBytes`
    template<std::integral I>
    constexpr ptr operator+(const I elements) const {return ptr(address+static_cast<uint64_t>(elements)*sizeof(T));}
    template<std::integral I>
    constexpr ptr operator-(const I elements) const {return ptr(address-static_cast<uint64_t>(elements)*sizeof(T));}
    template<std::integral I>
    constexpr ptr& operator+=(const I elements) {return *this = *this+elements;}
    template<std::integral I>
    constexpr ptr& operator-=(const I elements) {return *this = *this-elements;}

    template<std::integral I>
    constexpr ptr offsetBytes(const I bytes) const {return ptr(address+static_cast<uint64_t>(bytes));}
};
static_assert(sizeof(ptr<uint32_t>)==sizeof(uint64_t) && alignof(ptr<uint32_t>)==alignof(uint64_t) && std::is_trivially_copyable_v<ptr<uint32_t> >);
#endif

}

#ifdef __HLSL_VERSION
namespace impl
{
template<typename T, uint32_t Alignment>
struct static_cast_helper<uint64_t,vk::BufferPointer<T,Alignment>,void>
{
    static uint64_t cast(vk::BufferPointer<T,Alignment> p) {return (uint64_t)p;}
};
}
#endif

}
}

#endif
