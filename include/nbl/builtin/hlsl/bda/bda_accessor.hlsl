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

template<typename T>
struct BdaAccessor
{
    using type_t = T;
    static BdaAccessor<T> create(const bda::__ptr<T> ptr)
    {
        BdaAccessor<T> accessor;
        accessor.ptr = ptr;
        return accessor;
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

    enable_if_t<is_integral<T>::value && (sizeof(T) == 4 || sizeof(T) == 8), T>
    atomicAdd(const uint64_t index, const T value)
    {
        bda::__ptr<T> target = ptr + index;
        return glsl::atomicAdd(target.template deref().get_ptr(), value);
    }

    enable_if_t<is_integral<T>::value && (sizeof(T) == 4 || sizeof(T) == 8), T>
    atomicSub(const uint64_t index, const T value)
    {
        bda::__ptr<T> target = ptr + index;
        return glsl::atomicSub(target.template deref().get_ptr(), value);
    }

    bda::__ptr<T> ptr;
};

}
}

#endif
