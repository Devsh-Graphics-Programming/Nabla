// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MEMORY_ACCESSOR_INCLUDED_
#define _NBL_BUILTIN_HLSL_MEMORY_ACCESSOR_INCLUDED_

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/member_test_macros.hlsl"

// weird namespace placing, see the comment where the macro is defined
GENERATE_METHOD_TESTER(atomicExchange)
GENERATE_METHOD_TESTER(atomicCompSwap)
GENERATE_METHOD_TESTER(atomicAnd)
GENERATE_METHOD_TESTER(atomicOr)
GENERATE_METHOD_TESTER(atomicXor)
GENERATE_METHOD_TESTER(atomicAdd)
GENERATE_METHOD_TESTER(atomicMin)
GENERATE_METHOD_TESTER(atomicMax)
GENERATE_METHOD_TESTER(workgroupExecutionAndMemoryBarrier)

namespace nbl 
{
namespace hlsl
{

// TODO: flesh out and move to `nbl/builtin/hlsl/utility.hlsl`
template<typename T1, typename T2>
struct pair
{
    using first_type = T1;
    using second_type = T2;

    first_type first;
    second_type second;
};

namespace accessor_adaptors
{
namespace impl
{
// only base class to use integral_constant because we need to use void to indicate a dynamic value and all values are valid
template<typename IndexType, typename Offset>
struct OffsetBase
{
    NBL_CONSTEXPR IndexType offset = Offset::value;
};
template<typename IndexType>
struct OffsetBase<IndexType,void>
{
    IndexType offset;
};

template<typename IndexType, uint64_t ElementStride, uint64_t SubElementStride, typename Offset>
struct StructureOfArraysStrides
{
    NBL_CONSTEXPR IndexType elementStride = ElementStride;
    NBL_CONSTEXPR IndexType subElementStride = SubElementStride;

    //static_assert(elementStride>0 && subElementStride>0);
};
template<typename IndexType, typename Offset>
struct StructureOfArraysStrides<IndexType,0,0,Offset> : OffsetBase<IndexType,Offset>
{
    IndexType elementStride;
    IndexType subElementStride;
};
#if 0 // don't seem to be able to specialize one at a time
template<typename IndexType, uint64_t ElementStride, typename Offset>
struct StructureOfArraysStrides<IndexType,ElementStride,0,Offset> : OffsetBase<IndexType,Offset>
{
    NBL_CONSTEXPR IndexType elementStride = ElementStride;
    IndexType subElementStride;
};
template<typename IndexType, uint64_t SubElementStride, typename Offset>
struct StructureOfArraysStrides<IndexType,0,SubElementStride,Offset> : OffsetBase<IndexType,Offset>
{
    IndexType elementStride;
    NBL_CONSTEXPR IndexType subElementStride = SubElementStride;
};
#endif


template<typename IndexType, uint64_t ElementStride, uint64_t SubElementStride, typename Offset>
struct StructureOfArraysBase : StructureOfArraysStrides<IndexType,ElementStride,SubElementStride,Offset>
{
    IndexType getIx(const IndexType ix, const IndexType el)
    {
        using base_t = StructureOfArraysStrides<IndexType,ElementStride,SubElementStride,Offset>;
        return base_t::elementStride*ix+base_t::subElementStride*el+OffsetBase<IndexType,Offset>::offset;
    }
};

// maybe we should have our own std::array
template<typename T, uint64_t count>
struct array
{
    T data[count];
};
}

// TODO: some CRTP thing to forward through atomics and barriers

// If you want static strides pass `Stride=pair<integral_constant<IndexType,ElementStride>,integral_constant<IndexType,SubElementStride> >`
template<class BaseAccessor, typename AccessType, typename IndexType=uint32_t, uint64_t ElementStride=0, uint64_t SubElementStride=0, typename _Offset=integral_constant<IndexType,0> >
struct StructureOfArrays : impl::StructureOfArraysBase<IndexType,ElementStride,SubElementStride,_Offset>
{
    using base_t = impl::StructureOfArraysBase<IndexType,ElementStride,SubElementStride,_Offset>;
    // Question: should the `BaseAccessor` let us know what this is?
    using access_t = AccessType;
    using index_t = IndexType;

    BaseAccessor accessor;

    // Question: shall we go back to requiring a `access_t get(index_t)` on the `BaseAccessor`, then we could `enable_if` check the return type (via `has_method_get`) matches and we won't get Nasty HLSL copy-in copy-out conversions
    template<typename T>
    enable_if_t<sizeof(T)%sizeof(access_t)==0,void> get(const index_t ix, NBL_REF_ARG(T) value)
    {
        NBL_CONSTEXPR uint64_t SubElementCount = sizeof(T)/sizeof(access_t);
        // `vector` for now, we'll use `array` later when `bit_cast` gets fixed
        vector<access_t,SubElementCount> aux;
        for (index_t i=0; i<SubElementCount; i++)
            accessor.get(base_t::getIx(ix,i),aux[i]);
        value = bit_cast<T,vector<access_t,SubElementCount> >(aux);
    }

    template<typename T>
    enable_if_t<sizeof(T)%sizeof(access_t)==0,void> set(const index_t ix, NBL_CONST_REF_ARG(T) value)
    { 
        NBL_CONSTEXPR uint64_t SubElementCount = sizeof(T)/sizeof(access_t);
        // `vector` for now, we'll use `array` later when `bit_cast` gets fixed
        vector<access_t,SubElementCount> aux;
        aux = bit_cast<vector<access_t,SubElementCount>,T>(value);
        for (index_t i=0; i<SubElementCount; i++)
            accessor.set(base_t::getIx(ix,i),aux[i]);

    }

    template<typename T, typename S=BaseAccessor>
    enable_if_t<
        sizeof(T)==sizeof(access_t) && is_same_v<S,BaseAccessor> && is_same_v<typename has_method_atomicExchange<S,index_t,access_t>::return_type,access_t>,void
    > atomicExchange(const index_t ix, const T value, NBL_REF_ARG(T) orig)
    {
       orig = bit_cast<T,access_t>(accessor.atomicExchange(getIx(ix),bit_cast<access_t,T>(value)));
    }
    template<typename T, typename S=BaseAccessor>
    enable_if_t<
        sizeof(T)==sizeof(access_t) && is_same_v<S,BaseAccessor> && is_same_v<typename has_method_atomicCompSwap<S,index_t,access_t,access_t>::return_type,access_t>,void
    > atomicCompSwap(const index_t ix, const T value, const T comp, NBL_REF_ARG(T) orig)
    {
       orig = bit_cast<T,access_t>(accessor.atomicCompSwap(getIx(ix),bit_cast<access_t,T>(comp),bit_cast<access_t,T>(value)));
    }

    template<typename T, typename S=BaseAccessor>
    enable_if_t<
        sizeof(T)==sizeof(access_t) && is_same_v<S,BaseAccessor> && is_same_v<typename has_method_atomicAnd<S,index_t,access_t>::return_type,access_t>,void
    > atomicAnd(const index_t ix, const T value, NBL_REF_ARG(T) orig)
    {
       orig = bit_cast<T,access_t>(accessor.atomicAnd(getIx(ix),bit_cast<access_t,T>(value)));
    }
    template<typename T, typename S=BaseAccessor>
    enable_if_t<
        sizeof(T)==sizeof(access_t) && is_same_v<S,BaseAccessor> && is_same_v<typename has_method_atomicOr<S,index_t,access_t>::return_type,access_t>,void
    > atomicOr(const index_t ix, const T value, NBL_REF_ARG(T) orig)
    {
       orig = bit_cast<T,access_t>(accessor.atomicOr(getIx(ix),bit_cast<access_t,T>(value)));
    }
    template<typename T, typename S=BaseAccessor>
    enable_if_t<
        sizeof(T)==sizeof(access_t) && is_same_v<S,BaseAccessor> && is_same_v<typename has_method_atomicXor<S,index_t,access_t>::return_type,access_t>,void
    > atomicXor(const index_t ix, const T value, NBL_REF_ARG(T) orig)
    {
       orig = bit_cast<T,access_t>(accessor.atomicXor(getIx(ix),bit_cast<access_t,T>(value)));
    }

    // This has the upside of never calling a `(uint32_t)(uint32_t,uint32_t)` overload of `atomicAdd` because it checks the return type!
    // If someone makes a `(float)(uint32_t,uint32_t)` they will break this detection code, but oh well.
    template<typename T>
    enable_if_t<is_same_v<typename has_method_atomicAdd<BaseAccessor,index_t,T>::return_type,T>,void> atomicAdd(const index_t ix, const T value, NBL_REF_ARG(T) orig)
    {
       orig = accessor.atomicAdd(getIx(ix),value);
    }
    template<typename T>
    enable_if_t<is_same_v<typename has_method_atomicMin<BaseAccessor,index_t,T>::return_type,T>,void> atomicMin(const index_t ix, const T value, NBL_REF_ARG(T) orig)
    {
       orig = accessor.atomicMin(getIx(ix),value);
    }
    template<typename T>
    enable_if_t<is_same_v<typename has_method_atomicMax<BaseAccessor,index_t,T>::return_type,T>,void> atomicMax(const index_t ix, const T value, NBL_REF_ARG(T) orig)
    {
        orig = accessor.atomicMax(getIx(ix),value);
    }
    
    template<typename S=BaseAccessor>
    enable_if_t<
        is_same_v<S,BaseAccessor> && is_same_v<typename has_method_workgroupExecutionAndMemoryBarrier<S>::return_type,void>,void
    > workgroupExecutionAndMemoryBarrier()
    {
        accessor.workgroupExecutionAndMemoryBarrier();
    }
};

// ---------------------------------------------- Offset Accessor ----------------------------------------------------

template<class BaseAccessor, typename IndexType=uint32_t, typename _Offset=void>
struct Offset : impl::OffsetBase<IndexType,_Offset>
{
    using base_t = impl::OffsetBase<IndexType,_Offset>;
    using index_t = IndexType;

    BaseAccessor accessor;

    template <typename T>
    void set(index_t idx, T value) {accessor.set(idx+base_t::offset,value); }

    template <typename T> 
    void get(index_t idx, NBL_REF_ARG(T) value) {accessor.get(idx+base_t::offset,value);}
    
    template<typename S=BaseAccessor>
    enable_if_t<
        is_same_v<S,BaseAccessor> && is_same_v<typename has_method_workgroupExecutionAndMemoryBarrier<S>::return_type,void>,void
    > workgroupExecutionAndMemoryBarrier()
    {
        accessor.workgroupExecutionAndMemoryBarrier();
    }
};

}
}
}
#endif