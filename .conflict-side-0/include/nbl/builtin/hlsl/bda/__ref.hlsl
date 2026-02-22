// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BDA_REF_INCLUDED_
#define _NBL_BUILTIN_HLSL_BDA_REF_INCLUDED_

// TODO: this shouldn't be included IMHO
#include "nbl/builtin/hlsl/functional.hlsl"
#include "nbl/builtin/hlsl/spirv_intrinsics/core.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bda
{
template<typename T, bool _restrict>
struct __spv_ptr_t;
template<typename T>
struct __spv_ptr_t<T,false>
{
    [[vk::ext_decorate(spv::DecorationAliasedPointer)]] spirv::bda_pointer_t<T> value;
};
template<typename T>
struct __spv_ptr_t<T,true>
{
    [[vk::ext_decorate(spv::DecorationRestrictPointer)]] spirv::bda_pointer_t<T> value;
};

template<typename T, uint32_t alignment, bool _restrict>
struct __base_ref
{
    __spv_ptr_t<T,_restrict> ptr;

    void __init(const spirv::bda_pointer_t<T> _ptr)
    {
        ptr.value = _ptr;
    }
    
    spirv::bda_pointer_t<T> __get_spv_ptr()
    {
        // BUG: https://github.com/microsoft/DirectXShaderCompiler/issues/7184
        // if I don't launder the pointer through this I get "IsNonPtrAccessChain(ptrInst->opcode())" 
        return spirv::copyObject<spirv::bda_pointer_t<T> >(ptr.value);
    }

    T load()
    {
        return spirv::load<T,alignment>(__get_spv_ptr());
    }

    void store(const T val)
    {
        spirv::store<T,alignment>(__get_spv_ptr(),val);
    }
};

// TODO: I wish HLSL had some things like C++ which would allow you to make a "stack only"/non-storable type
// NOTE: I guess there's the Function/Private storage space variables?
template<typename T, uint32_t alignment=alignment_of_v<T>, bool _restrict=false>
struct __ref : __base_ref<T,alignment,_restrict>
{
    using base_t = __base_ref< T,alignment,_restrict>;
    using this_t = __ref<T,alignment,_restrict>;
};
}
}
}
#endif