// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BDA_REF_INCLUDED_
#define _NBL_BUILTIN_HLSL_BDA_REF_INCLUDED_

#include "nbl/builtin/hlsl/functional.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bda
{

template<typename T, uint32_t alignment, bool _restrict>
struct __base_ref;
template<typename T, uint32_t alignment>
struct __base_ref<T,alignment,false>
{
    [[vk::ext_decorate(spv::DecorationAliasedPointer)]] spirv::bda_pointer_t<T> ptr;

    void __init(const spirv::bda_pointer_t<T> _ptr)
    {
        ptr = _ptr;
    }

    spirv::bda_pointer_t<T> __get_spv_ptr()
    {
        // BUG: if I don't launder the pointer through this I get ""
        return spirv::bitcast<spirv::bda_pointer_t<T> >(spirv::bitcast<uint32_t2>(ptr));
    }

    T load()
    {
        return spirv::load<T,alignment>(__get_spv_ptr());
    }

    void store(const T val)
    {
        spirv::store<T,alignment>(__get_spv_ptr(), val);
    }
};
template<typename T, uint32_t alignment>
struct __base_ref<T,alignment,true>
{
    [[vk::ext_decorate(spv::DecorationRestrictPointer)]] spirv::bda_pointer_t<T> ptr;

    void __init(const spirv::bda_pointer_t<T> _ptr)
    {
        ptr = _ptr;
    }

    spirv::bda_pointer_t<T> __get_spv_ptr()
    {
        // BUG: if I don't launder the pointer through this I get ""
        return spirv::bitcast<spirv::bda_pointer_t<T> >(spirv::bitcast<uint32_t2>(ptr));
    }

    T load()
    {
        return spirv::load<T,alignment>(__get_spv_ptr() );
    }

    void store(const T val)
    {
        spirv::store<T,alignment>(__get_spv_ptr(), val);
    }
};

// TODO: I wish HLSL had some things like C++ which would allow you to make a "stack only"/non-storable type
template<typename T, uint32_t alignment=alignment_of_v<T>, bool _restrict=false>
struct __ref : __base_ref<T,alignment,_restrict>
{
    using base_t = __base_ref< T,alignment,_restrict>;
    using this_t = __ref<T,alignment,_restrict>;
};
}
}
}

// time for some macros!
// Sequence of (variableName,Type)
#endif