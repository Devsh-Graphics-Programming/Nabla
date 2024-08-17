// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MEMORY_ACCESSOR_INCLUDED_
#define _NBL_BUILTIN_HLSL_MEMORY_ACCESSOR_INCLUDED_

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"

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


// TODO: find some cool way to SFINAE the default into `_NBL_HLSL_WORKGROUP_SIZE_` if defined, and something like 1 otherwise
template<class BaseAccessor, typename AccessType, typename IndexType=uint32_t, typename Strides=pair<integral_constant<IndexType,1>,integral_constant<IndexType,_NBL_HLSL_WORKGROUP_SIZE_> > >
struct MemoryAdaptor // TODO: rename to something nicer like StructureOfArrays and add a `namespace accessor_adaptors`
{
    // Question: should the `BaseAccessor` let us know what this is?
    using access_t = AccessType;
    using index_t = IndexType;
    NBL_CONSTEXPR index_t ElementStride = Strides::first_type::value;
    NBL_CONSTEXPR index_t SubElementStride = Strides::second_type::value;

    BaseAccessor accessor;
    
    access_t get(const index_t ix)
    { 
        access_t retVal;
        get<access_t>(ix,retVal);
        return retVal; 
    }

    // Question: shall we go back to requiring a `access_t get(index_t)` on the `BaseAccessor`, then we could `enable_if` check the return type (via `has_method_get`) matches and we won't get Nasty HLSL copy-in copy-out conversions
    template<typename T>
    enable_if_t<sizeof(T)%sizeof(access_t)==0,void> get(const index_t ix, NBL_REF_ARG(T) value)
    { 
        NBL_CONSTEXPR uint64_t SubElementCount = sizeof(T)/sizeof(access_t);
        access_t aux[SubElementCount];
        for (uint64_t i=0; i<SubElementCount; i++)
            accessor.get(ix*ElementStride+i*SubElementStride,aux[i]);
        value = bit_cast<T,access_t[SubElementCount]>(aux);
    }

    template<typename T>
    enable_if_t<sizeof(T)%sizeof(access_t)==0,void> set(const index_t ix, NBL_CONST_REF_ARG(T) value)
    { 
        NBL_CONSTEXPR uint64_t SubElementCount = sizeof(T)/sizeof(access_t);
        access_t aux[SubElementCount] = bit_cast<access_t[SubElementCount],T>(value);
        for (uint64_t i=0; i<SubElementCount; i++)
            accessor.set(ix*ElementStride+i*SubElementStride,aux[i]);
    }
    
    template<typename T, typename S=BaseAccessor>
    enable_if_t<
        sizeof(T)==sizeof(access_t) && is_same_v<S,BaseAccessor> && is_same_v<has_method_atomicExchange<S,index_t,access_t>::return_type,access_t>,void
    > atomicExchange(const index_t ix, const T value, NBL_REF_ARG(T) orig)
    {
       orig = bit_cast<T,access_t>(accessor.atomicExchange(ix,bit_cast<access_t,T>(value)));
    }
    template<typename T, typename S=BaseAccessor>
    enable_if_t<
        sizeof(T)==sizeof(access_t) && is_same_v<S,BaseAccessor> && is_same_v<has_method_atomicCompSwap<S,index_t,access_t,access_t>::return_type,access_t>,void
    > atomicCompSwap(const index_t ix, const T value, const T comp, NBL_REF_ARG(T) orig)
    {
       orig = bit_cast<T,access_t>(accessor.atomicCompSwap(ix,bit_cast<access_t,T>(comp),bit_cast<access_t,T>(value)));
    }

    template<typename T, typename S=BaseAccessor>
    enable_if_t<
        sizeof(T)==sizeof(access_t) && is_same_v<S,BaseAccessor> && is_same_v<has_method_atomicAnd<S,index_t,access_t>::return_type,access_t>,void
    > atomicAnd(const index_t ix, const T value, NBL_REF_ARG(T) orig)
    {
       orig = bit_cast<T,access_t>(accessor.atomicAnd(ix,bit_cast<access_t,T>(value)));
    }
    template<typename T, typename S=BaseAccessor>
    enable_if_t<
        sizeof(T)==sizeof(access_t) && is_same_v<S,BaseAccessor> && is_same_v<has_method_atomicOr<S,index_t,access_t>::return_type,access_t>,void
    > atomicOr(const index_t ix, const T value, NBL_REF_ARG(T) orig)
    {
       orig = bit_cast<T,access_t>(accessor.atomicOr(ix,bit_cast<access_t,T>(value)));
    }
    template<typename T, typename S=BaseAccessor>
    enable_if_t<
        sizeof(T)==sizeof(access_t) && is_same_v<S,BaseAccessor> && is_same_v<has_method_atomicXor<S,index_t,access_t>::return_type,access_t>,void
    > atomicXor(const index_t ix, const T value, NBL_REF_ARG(T) orig)
    {
       orig = bit_cast<T,access_t>(accessor.atomicXor(ix,bit_cast<access_t,T>(value)));
    }

    // This has the upside of never calling a `(uint32_t)(uint32_t,uint32_t)` overload of `atomicAdd` because it checks the return type!
    // If someone makes a `(float)(uint32_t,uint32_t)` they will break this detection code, but oh well.
    template<typename T>
    enable_if_t<is_same_v<has_method_atomicAdd<BaseAccessor,index_t,T>::return_type,T>,void> atomicAdd(const index_t ix, const T value, NBL_REF_ARG(T) orig)
    {
       orig = accessor.atomicAdd(ix,value);
    }
    template<typename T>
    enable_if_t<is_same_v<has_method_atomicMin<BaseAccessor,index_t,T>::return_type,T>,void> atomicMin(const index_t ix, const T value, NBL_REF_ARG(T) orig)
    {
       orig = accessor.atomicMin(ix,value);
    }
    template<typename T>
    enable_if_t<is_same_v<has_method_atomicMax<BaseAccessor,index_t,T>::return_type,T>,void> atomicMax(const index_t ix, const T value, NBL_REF_ARG(T) orig)
    {
        orig = accessor.atomicMax(ix,value);
    }
    
    template<typename S=BaseAccessor>
    enable_if_t<
        is_same_v<S,BaseAccessor> && is_same_v<has_method_workgroupExecutionAndMemoryBarrier<S>::return_type,void>,void
    > workgroupExecutionAndMemoryBarrier()
    {
        accessor.workgroupExecutionAndMemoryBarrier();
    }
};

}
}

#endif