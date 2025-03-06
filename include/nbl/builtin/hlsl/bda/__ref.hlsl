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
        // BUG: if I don't launder the pointer through this I get "IsNonPtrAccessChain(ptrInst->opcode())"
        //return ptr.value;
        // What to do!? OpCopyObject? trick the compiler into giving me an immediate value some other way!?
        // If I add `[[vk::ext_reference]]` to my OpLoad and OpStore, then compiler doesn't emit anything!?
        return spirv::bitcast<spirv::bda_pointer_t<T> >(spirv::bitcast<uint32_t2>(ptr.value));
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
// need to gen identical struct in HLSL and C++
/*
struct MyStruct
{
    // TODO: compute offsets from sizes and alignments
    [[vk::ext_decorate(spv::DecorationOffset,0)]] float32_t a;
    [[vk::ext_decorate(spv::DecorationOffset,4)]] int32_t b;
    [[vk::ext_decorate(spv::DecorationOffset,8)]] int16_t2 c;
};
template<>
struct nbl::hlsl::alignment_of<MyStruct>
{
    // TODO: compute alignment if not user specified
    NBL_CONSTEXPR_STATIC_INLINE uint32_t value = 4;
};
template<<>
struct nbl::hlsl::impl::member_info<MyStruct,0>
{
    using type = float32_t;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t offset = 0;
};
template<>
struct nbl::hlsl::impl::member_info<MyStruct,1>
{
    using type = int32_t;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t offset = nbl::hlsl::mpl::round_up<member_info<MyStruct,0>::offset+sizeof(member_info<MyStruct,0>::type),alignof<type> >::value;
};


template<uint32_t alignment, bool _restrict>
struct nbl::hlsl::bda::__ref<MyStruct,alignment,_restrict> : nbl::hlsl::bda::__base_ref<MyStruct,alignment,_restrict>
{
    using base_t = __base_ref<MyStruct,alignment,_restrict>;
    using this_t = __ref<MyStruct,alignment,_restrict>;

    // TODO: compute alignment as min of base alignment and offset alignment
    nbl::hlsl::bda::__ref<float32_t,1,_restrict> a;
    nbl::hlsl::bda::__ref<int32_t,1,_restrict> b;
    nbl::hlsl::bda::__ref<int16_t2,1,_restrict> c;

    void __init(const nbl::hlsl::spirv::bda_pointer_t<MyStruct> _ptr)
    {
        base_t::__init(_ptr);
        a.__init(spirv::accessChain<float32_t>(base_t::__get_spv_ptr(),0));
        b.__init(spirv::accessChain<int32_t>(base_t::__get_spv_ptr(),1));
        c.__init(spirv::accessChain<int16_t2>(base_t::__get_spv_ptr(),2));
    }
};
*/
#endif