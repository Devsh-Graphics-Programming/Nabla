// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/spirv_intrinsics/core.hlsl"

#ifndef _NBL_BUILTIN_HLSL_BDA_REF_INCLUDED_
#define _NBL_BUILTIN_HLSL_BDA_REF_INCLUDED_

namespace spirv
{
template<typename M, typename T, typename StorageClass>
[[vk::ext_instruction(spv::OpAccessChain)]]
vk::SpirvOpaqueType <spv::OpTypePointer,StorageClass,M> accessChain(
    [[vk::ext_reference]] vk::SpirvOpaqueType <spv::OpTypePointer,StorageClass,T>base,
    [[vk::ext_literal]] uint32_t index0
);
}

namespace bda
{
template<typename T>
using __spv_ptr_t = vk::SpirvOpaqueType<
    spv::OpTypePointer,
    vk::Literal< vk::integral_constant<uint, spv::StorageClassPhysicalStorageBuffer> >,
    T
>;

namespace impl
{
// this only exists to workaround DXC issue XYZW TODO https://github.com/microsoft/DirectXShaderCompiler/issues/6576
template<class T>
[[vk::ext_capability(spv::CapabilityPhysicalStorageBufferAddresses)]]
[[vk::ext_instruction(spv::OpBitcast)]]
T bitcast(uint64_t);

template<typename T, typename P, uint32_t alignment>
[[vk::ext_capability(spv::CapabilityPhysicalStorageBufferAddresses)]]
[[vk::ext_instruction(spv::OpLoad)]]
T load(P pointer, [[vk::ext_literal]] uint32_t __aligned = /*Aligned*/0x00000002, [[vk::ext_literal]] uint32_t __alignment = alignment);

template<typename T, typename P, uint32_t alignment >
[[vk::ext_capability(spv::CapabilityPhysicalStorageBufferAddresses)]]
[[vk::ext_instruction(spv::OpStore)]]
void store(P pointer, T obj, [[vk::ext_literal]] uint32_t __aligned = /*Aligned*/0x00000002, [[vk::ext_literal]] uint32_t __alignment = alignment);

// TODO: atomics for different types
template<typename T, typename P> // integers operate on 2s complement so same op for signed and unsigned
[[vk::ext_instruction(spv::OpAtomicIAdd)]]
T atomicIAdd(P ptr, uint32_t memoryScope, uint32_t memorySemantics, T value);
}

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
        return impl::bitcast < __spv_ptr_t<T> > (ptr);
    }

    // TODO: Would like to use `spv_ptr_t` or OpAccessChain result instead of `uint64_t`
    void __init(const spv_ptr_t _ptr)
    {
        ptr = _ptr;
    }

    __ptr<T> addrof()
    {
        __ptr<T> retval;
        retval.addr = nbl::hlsl::spirv::bitcast<uint64_t>(ptr);
        return retval;
    }

    T load()
    {
        return impl::load < T, __spv_ptr_t<T>, alignment > (__get_spv_ptr());
    }

    void store(const T val)
    {
        impl::store < T, __spv_ptr_t<T>, alignment > (__get_spv_ptr(), val);
    }
};

template<typename T, uint32_t alignment/*=alignof(T)*/, bool _restrict = false>
struct __ref : __base_ref<T,alignment,_restrict>
{
    using base_t = __base_ref < T, alignment, _restrict>;
    using this_t = __ref < T, alignment, _restrict>;
};

#define REF_INTEGRAL(Type)                                                      \
template<uint32_t alignment, bool _restrict>                                    \
struct __ref<Type,alignment,_restrict> : __base_ref<Type,alignment,_restrict>   \
{                                                                               \
    using base_t = __base_ref <Type, alignment, _restrict>;                     \
    using this_t = __ref <Type, alignment, _restrict>;                          \
                                                                                \
    [[vk::ext_capability(spv::CapabilityPhysicalStorageBufferAddresses)]]       \
    Type atomicAdd(const Type value)                                            \
    {                                                                           \
        return impl::atomicIAdd <Type> (base_t::__get_spv_ptr(), 1, 0, value);  \
    }                                                                           \
};

// TODO: specializations for simple builtin types that have atomics
// We are currently only supporting builtin types that work with atomicIAdd
REF_INTEGRAL(int16_t)
REF_INTEGRAL(uint16_t)
REF_INTEGRAL(int32_t)
REF_INTEGRAL(uint32_t)
REF_INTEGRAL(int64_t)
REF_INTEGRAL(uint64_t)
}

#endif