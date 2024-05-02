// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/builtin/hlsl/sort/common.hlsl"
#include "nbl/builtin/hlsl/workgroup/arithmetic.hlsl"

#ifndef _NBL_BUILTIN_HLSL_SORT_COUNTING_INCLUDED_
#define _NBL_BUILTIN_HLSL_SORT_COUNTING_INCLUDED_

namespace spirv
{
template<typename M, typename T, typename StorageClass>
[[vk::ext_instruction(/*spv::OpAccessChain*/65)]]
vk::SpirvOpaqueType </* OpTypePointer*/ 32,StorageClass,M> accessChain(
    [[vk::ext_reference]] vk::SpirvOpaqueType </* OpTypePointer*/ 32,
    StorageClass,T>base,
    [[vk::ext_literal]] uint32_t index0
);

template<class T, class U>
[[vk::ext_instruction( /*spv::OpBitcast*/124)]]
T bitcast(U);

template<typename T> // integers operate on 2s complement so same op for signed and unsigned
[[vk::ext_instruction(/*spv::OpAtomicIAdd*/234)]]
T atomicIAdd([[vk::ext_reference]] T ptr, uint32_t memoryScope, uint32_t memorySemantics, T value);
}

namespace bda
{
template<typename T>
using __spv_ptr_t = vk::SpirvOpaqueType<
    /* OpTypePointer */ 32,
    /* PhysicalStorageBuffer */ vk::Literal<vk::integral_constant<uint,5349> >,
    T
>;

namespace impl
{
// this only exists to workaround DXC issue XYZW TODO https://github.com/microsoft/DirectXShaderCompiler/issues/6576
template<class T>
[[vk::ext_capability(/*PhysicalStorageBufferAddresses */ 5347 )]]
[[vk::ext_instruction(/*spv::OpBitcast*/124)]]
T bitcast(uint64_t);

template<typename T, typename P, uint32_t alignment>
[[vk::ext_capability( /*PhysicalStorageBufferAddresses */5347)]]
[[vk::ext_instruction( /*OpLoad*/61)]]
T load(P pointer, [[vk::ext_literal]] uint32_t __aligned = /*Aligned*/0x00000002, [[vk::ext_literal]] uint32_t __alignment = alignment);

template<typename T, typename P, uint32_t alignment >
[[vk::ext_capability( /*PhysicalStorageBufferAddresses */5347)]]
[[vk::ext_instruction( /*OpStore*/62)]]
void store(P pointer, T obj, [[vk::ext_literal]] uint32_t __aligned = /*Aligned*/0x00000002, [[vk::ext_literal]] uint32_t __alignment = alignment);

// TODO: atomics for different types
template<typename T, typename P> // integers operate on 2s complement so same op for signed and unsigned
[[vk::ext_instruction( /*spv::OpAtomicIAdd*/234)]]
T atomicIAdd(P ptr, uint32_t memoryScope, uint32_t memorySemantics, T value);
}

// TODO: maybe make normal and restrict separate distinct types instead of templates
template<typename T, bool _restrict = false>
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

    __ptr<T,_restrict> addrof()
    {
        __ptr<T,_restrict> retval;
        retval.addr = spirv::bitcast<uint64_t>(ptr);
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
    [[vk::ext_capability(/*PhysicalStorageBufferAddresses */ 5347 )]]           \
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

template<typename T, bool _restrict>
struct __ptr
{
    using this_t = __ptr < T, _restrict>;
    uint64_t addr;

    static this_t create(const uint64_t _addr)
    {
        this_t retval;
        retval.addr = _addr;
        return retval;
    }

    template<uint32_t alignment>
    __ref<T,alignment,_restrict> deref()
    {
        // TODO: assert(addr&uint64_t(alignment-1)==0);
        using retval_t = __ref < T, alignment, _restrict>;
        retval_t retval;
        retval.__init(impl::bitcast<typename retval_t::spv_ptr_t>(addr));
        return retval;
    }
};
}

namespace nbl
{
namespace hlsl
{
namespace sort
{

NBL_CONSTEXPR uint32_t BucketsPerThread = ceil((float) BucketCount / WorkgroupSize);

groupshared uint32_t prefixScratch[BucketCount];

struct ScratchProxy
{
    uint32_t get(const uint32_t ix)
    {
        return prefixScratch[ix];
    }
    void set(const uint32_t ix, const uint32_t value)
    {
        prefixScratch[ix] = value;
    }

    void workgroupExecutionAndMemoryBarrier()
    {
        nbl::hlsl::glsl::barrier();
    }
};

static ScratchProxy arithmeticAccessor;

groupshared uint32_t sdata[BucketCount];

template<typename KeyAccessor, typename ValueAccessor, typename ScratchAccessor>
struct counting
{
    void init(
        const CountingPushData data
    ) {
        in_key_addr         = data.inputKeyAddress;
        out_key_addr        = data.outputKeyAddress;
        in_value_addr       = data.inputValueAddress;
        out_value_addr      = data.outputValueAddress;
        scratch_addr        = data.scratchAddress;
        data_element_count  = data.dataElementCount;
        minimum             = data.minimum;
        elements_per_wt     = data.elementsPerWT;
    }

    void histogram()
    {
        uint32_t tid = nbl::hlsl::workgroup::SubgroupContiguousIndex();

        [unroll]
        for (int i = 0; i < BucketsPerThread; i++)
            sdata[BucketsPerThread * tid + i] = 0;
        uint32_t index = (nbl::hlsl::glsl::gl_WorkGroupID().x * WorkgroupSize + tid) * elements_per_wt;

        nbl::hlsl::glsl::barrier();

        for (int i = 0; i < elements_per_wt; i++)
        {
            if (index + i >= data_element_count)
                break;
            uint32_t value = ValueAccessor(in_value_addr + sizeof(uint32_t) * (index + i)).template deref<4>().load();
            nbl::hlsl::glsl::atomicAdd(sdata[value - minimum], (uint32_t) 1);
        }

        nbl::hlsl::glsl::barrier();

        uint32_t sum = 0;
        uint32_t scan_sum = 0;

        for (int i = 0; i < BucketsPerThread; i++)
        {
            sum = nbl::hlsl::workgroup::exclusive_scan < nbl::hlsl::plus < uint32_t >, WorkgroupSize > ::
            template __call <ScratchProxy>
            (sdata[WorkgroupSize * i + tid], arithmeticAccessor);

            arithmeticAccessor.workgroupExecutionAndMemoryBarrier();

            ScratchAccessor(scratch_addr + sizeof(uint32_t) * (WorkgroupSize * i + tid)).template deref<4>().atomicAdd(sum);
            if ((tid == WorkgroupSize - 1) && i > 0)
                ScratchAccessor(scratch_addr + sizeof(uint32_t) * (WorkgroupSize * i)).template deref<4>().atomicAdd(scan_sum);

            arithmeticAccessor.workgroupExecutionAndMemoryBarrier();

            if ((tid == WorkgroupSize - 1) && i < (BucketsPerThread - 1))
            {
                scan_sum = sum + sdata[WorkgroupSize * i + tid];
                sdata[WorkgroupSize * (i + 1)] += scan_sum;
            }

            arithmeticAccessor.workgroupExecutionAndMemoryBarrier();
        }
    }
                
    void scatter()
    {
        uint32_t tid = nbl::hlsl::workgroup::SubgroupContiguousIndex();

        [unroll]
        for (int i = 0; i < BucketsPerThread; i++)
            sdata[BucketsPerThread * tid + i] = 0;
        uint32_t index = (nbl::hlsl::glsl::gl_WorkGroupID().x * WorkgroupSize + tid) * elements_per_wt;

        nbl::hlsl::glsl::barrier();

        [unroll]
        for (int i = 0; i < elements_per_wt; i++)
        {
            if (index + i >= data_element_count)
                break;
            uint32_t key = KeyAccessor(in_key_addr + sizeof(uint32_t) * (index + i)).template deref<4>().load();
            uint32_t value = ValueAccessor(in_value_addr + sizeof(uint32_t) * (index + i)).template deref<4>().load();
            nbl::hlsl::glsl::atomicAdd(sdata[value - minimum], (uint32_t) 1);
        }

        [unroll]
        for (int i = 0; i < elements_per_wt; i++)
        {
            if (index + i >= data_element_count)
                break;
            uint32_t key = KeyAccessor(in_key_addr + sizeof(uint32_t) * (index + i)).template deref<4>().load();
            uint32_t value = ValueAccessor(in_value_addr + sizeof(uint32_t) * (index + i)).template deref<4>().load();
            sdata[value - minimum] = ScratchAccessor(scratch_addr + sizeof(uint32_t) * (value - minimum)).template deref<4>().atomicAdd(1);
            KeyAccessor(out_key_addr + sizeof(uint32_t) * sdata[value - minimum]).template deref<4>().store(key);
            ValueAccessor(out_value_addr + sizeof(uint32_t) * sdata[value - minimum]).template deref<4>().store(value);
        }

        nbl::hlsl::glsl::barrier();
    }

    uint64_t in_key_addr, out_key_addr;
    uint64_t in_value_addr, out_value_addr;
    uint64_t scratch_addr;
    uint32_t data_element_count;
    uint32_t minimum;
    uint32_t elements_per_wt;
};

}
}
}

#endif