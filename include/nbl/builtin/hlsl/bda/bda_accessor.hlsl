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
namespace bda
{

template<typename T>
struct BdaAccessor
{
    static BdaAccessor<T> create(const uint64_t addr)
    {
        BdaAccessor<T> accessor;
        accessor.ptr = __ptr<T>::create(addr);
        return accessor;
    }

    T get(const uint64_t index)
    {
        __ptr<T> target = ptr + index;
        return target.template deref().load();
    }

    void set(const uint64_t index, const T value)
    {
        __ptr<T> target = ptr + index;
        return target.template deref().store(value);
    }

    enable_if_t<is_integral<T>::value && (sizeof(T) == 4 || sizeof(T) == 8), T>
    atomicAdd(const uint64_t index, const T value)
    {
        __ptr<T> target = ptr + index;
        return nbl::hlsl::glsl::atomicAdd(target.template deref().get_ptr(), value);
    }

    enable_if_t<is_integral<T>::value && (sizeof(T) == 4 || sizeof(T) == 8), T>
    atomicSub(const uint64_t index, const T value)
    {
        return atomicAdd(index, (T) (-1 * value));
    }

    __ptr<T> ptr;
};

}
}
}

#endif