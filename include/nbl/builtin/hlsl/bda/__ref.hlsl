// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/spirv_intrinsics/core.hlsl"

#ifndef _NBL_BUILTIN_HLSL_BDA_REF_INCLUDED_
#define _NBL_BUILTIN_HLSL_BDA_REF_INCLUDED_

namespace nbl
{
namespace hlsl
{
namespace bda
{
template<typename T>
using __spv_ptr_t = spirv::pointer_t<spv::StorageClassPhysicalStorageBuffer, T>;

template<typename T>
struct __ptr;

template<typename T, uint32_t alignment, bool _restrict>
struct __base_ref
{
// TODO:
// static_assert(alignment>=alignof(T));

    using spv_ptr_t = uint64_t;
    spv_ptr_t ptr;

    __spv_ptr_t<T> __get_spv_ptr()
    {
        return spirv::bitcast < __spv_ptr_t<T> > (ptr);
    }

    // TODO: Would like to use `spv_ptr_t` or OpAccessChain result instead of `uint64_t`
    void __init(const spv_ptr_t _ptr)
    {
        ptr = _ptr;
    }

    T load()
    {
        return spirv::load < T, __spv_ptr_t<T>, alignment > (__get_spv_ptr());
    }

    void store(const T val)
    {
        spirv::store < T, __spv_ptr_t<T>, alignment > (__get_spv_ptr(), val);
    }
};

template<typename T, uint32_t alignment=alignment_of_v<T>, bool _restrict = false>
struct __ref : __base_ref<T,alignment,_restrict>
{
    using base_t = __base_ref < T, alignment, _restrict>;
    using this_t = __ref < T, alignment, _restrict>;

    __spv_ptr_t<T> get_ptr()
    {
        return base_t::__get_spv_ptr();
    }
};
}
}
}
#endif