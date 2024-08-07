// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MEMORY_ACCESSOR_INCLUDED_
#define _NBL_BUILTIN_HLSL_MEMORY_ACCESSOR_INCLUDED_

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"

namespace nbl 
{
namespace hlsl
{

template<class BaseAccessor, uint32_t Stride = _NBL_HLSL_WORKGROUP_SIZE_>
struct MemoryAdaptor
{
    BaseAccessor accessor;

    // TODO: template atomic... then add static_asserts of `has_method<BaseAccessor,signature>::value`, do vectors and matrices in terms of each other
    uint get(const uint ix) 
    { 
        uint retVal;
        accessor.get(ix, retVal);
        return retVal; 
    }

    template<typename Scalar>
    enable_if_t<sizeof(Scalar) == sizeof(uint32_t), void> get(const uint ix, NBL_REF_ARG(Scalar) value) 
    { 
        uint32_t aux;
        accessor.get(ix, aux);
        value = bit_cast<Scalar, uint32_t>(aux);   
    }
    template<typename Scalar>
    enable_if_t<sizeof(Scalar) == sizeof(uint32_t), void> get(const uint ix, NBL_REF_ARG(vector <Scalar, 2>) value) 
    {
        uint32_t2 aux;
        accessor.get(ix, aux.x);
        accessor.get(ix + Stride, aux.y);
        value = bit_cast<vector<Scalar, 2>, uint32_t2>(aux);
    }
    template<typename Scalar>    
    enable_if_t<sizeof(Scalar) == sizeof(uint32_t), void> get(const uint ix, NBL_REF_ARG(vector <Scalar, 3>) value) 
    { 
        uint32_t3 aux;
        accessor.get(ix, aux.x);
        accessor.get(ix + Stride, aux.y);
        accessor.get(ix + 2 * Stride, aux.z);
        value = bit_cast<vector<Scalar, 3>, uint32_t3>(aux);
    }
    template<typename Scalar>   
    enable_if_t<sizeof(Scalar) == sizeof(uint32_t), void> get(const uint ix, NBL_REF_ARG(vector <Scalar, 4>) value) 
    { 
        uint32_t4 aux;
        accessor.get(ix, aux.x);
        accessor.get(ix + Stride, aux.y);
        accessor.get(ix + 2 * Stride, aux.z);
        accessor.get(ix + 3 * Stride, aux.w);
        value = bit_cast<vector<Scalar, 3>, uint32_t4>(aux);  
    }

    template<typename Scalar>
    enable_if_t<sizeof(Scalar) == sizeof(uint32_t), void> set(const uint ix, const Scalar value) {accessor.set(ix, asuint(value));}
    template<typename Scalar>    
    enable_if_t<sizeof(Scalar) == sizeof(uint32_t), void> set(const uint ix, const vector <Scalar, 2> value) {
        accessor.set(ix, asuint(value.x));
        accessor.set(ix + Stride, asuint(value.y));
    }
    template<typename Scalar>
    enable_if_t<sizeof(Scalar) == sizeof(uint32_t), void> set(const uint ix, const vector <Scalar, 3> value)  {
        accessor.set(ix, asuint(value.x));
        accessor.set(ix + Stride, asuint(value.y));
        accessor.set(ix + 2 * Stride, asuint(value.z));
    }
    template<typename Scalar>
    enable_if_t<sizeof(Scalar) == sizeof(uint32_t), void> set(const uint ix, const vector <Scalar, 4> value) {
        accessor.set(ix, asuint(value.x));
        accessor.set(ix + Stride, asuint(value.y));
        accessor.set(ix + 2 * Stride, asuint(value.z));
        accessor.set(ix + 3 * Stride, asuint(value.w));
    }
    
    void atomicAnd(const uint ix, const uint value, NBL_REF_ARG(uint) orig) {
       orig = accessor.atomicAnd(ix, value);
    }
    void atomicAnd(const uint ix, const int value, NBL_REF_ARG(int) orig) {
       orig = asint(accessor.atomicAnd(ix, asuint(value)));
    }
    void atomicAnd(const uint ix, const float value, NBL_REF_ARG(float) orig) {
       orig = asfloat(accessor.atomicAnd(ix, asuint(value)));
    }
    void atomicOr(const uint ix, const uint value, NBL_REF_ARG(uint) orig) {
       orig = accessor.atomicOr(ix, value);
    }
    void atomicOr(const uint ix, const int value, NBL_REF_ARG(int) orig) {
       orig = asint(accessor.atomicOr(ix, asuint(value)));
    }
    void atomicOr(const uint ix, const float value, NBL_REF_ARG(float) orig) {
       orig = asfloat(accessor.atomicOr(ix, asuint(value)));
    }
    void atomicXor(const uint ix, const uint value, NBL_REF_ARG(uint) orig) {
       orig = accessor.atomicXor(ix, value);
    }
    void atomicXor(const uint ix, const int value, NBL_REF_ARG(int) orig) {
       orig = asint(accessor.atomicXor(ix, asuint(value)));
    }
    void atomicXor(const uint ix, const float value, NBL_REF_ARG(float) orig) {
       orig = asfloat(accessor.atomicXor(ix, asuint(value)));
    }
    void atomicAdd(const uint ix, const uint value, NBL_REF_ARG(uint) orig) {
       orig = accessor.atomicAdd(ix, value);
    }
    void atomicMin(const uint ix, const uint value, NBL_REF_ARG(uint) orig) {
       orig = accessor.atomicMin(ix, value);
    }
    void atomicMax(const uint ix, const uint value, NBL_REF_ARG(uint) orig) {
       orig = accessor.atomicMax(ix, value);
    }
    void atomicExchange(const uint ix, const uint value, NBL_REF_ARG(uint) orig) {
       orig = accessor.atomicExchange(ix, value);
    }
    void atomicCompSwap(const uint ix, const uint value, const uint comp, NBL_REF_ARG(uint) orig) {
       orig = accessor.atomicCompSwap(ix, comp, value);
    }
    
    // TODO: figure out the `enable_if` syntax for this
    void workgroupExecutionAndMemoryBarrier() {
        accessor.workgroupExecutionAndMemoryBarrier();
    }
};

// Dynamic stride specialization
template<class BaseAccessor>
struct MemoryAdaptor<BaseAccessor, 0>
{
    BaseAccessor accessor;
    uint32_t stride;
   
    // TODO: template atomic... then add static_asserts of `has_method<BaseAccessor,signature>::value`, do vectors and matrices in terms of each other
    uint get(const uint ix) 
    { 
        uint retVal;
        accessor.get(ix, retVal);
        return retVal; 
    }

    template<typename Scalar>
    enable_if_t<sizeof(Scalar) == sizeof(uint32_t), void> get(const uint ix, NBL_REF_ARG(Scalar) value) 
    { 
        uint32_t aux;
        accessor.get(ix, aux);
        value = bit_cast<Scalar, uint32_t>(aux);   
    }
    template<typename Scalar>
    enable_if_t<sizeof(Scalar) == sizeof(uint32_t), void> get(const uint ix, NBL_REF_ARG(vector <Scalar, 2>) value) 
    {
        uint32_t2 aux;
        accessor.get(ix, aux.x);
        accessor.get(ix + stride, aux.y);
        value = bit_cast<vector<Scalar, 2>, uint32_t2>(aux);
    }
    template<typename Scalar>    
    enable_if_t<sizeof(Scalar) == sizeof(uint32_t), void> get(const uint ix, NBL_REF_ARG(vector <Scalar, 3>) value) 
    { 
        uint32_t3 aux;
        accessor.get(ix, aux.x);
        accessor.get(ix + stride, aux.y);
        accessor.get(ix + 2 * stride, aux.z);
        value = bit_cast<vector<Scalar, 3>, uint32_t3>(aux);
    }
    template<typename Scalar>   
    enable_if_t<sizeof(Scalar) == sizeof(uint32_t), void> get(const uint ix, NBL_REF_ARG(vector <Scalar, 4>) value) 
    { 
        uint32_t4 aux;
        accessor.get(ix, aux.x);
        accessor.get(ix + stride, aux.y);
        accessor.get(ix + 2 * stride, aux.z);
        accessor.get(ix + 3 * stride, aux.w);
        value = bit_cast<vector<Scalar, 3>, uint32_t4>(aux);  
    }

    template<typename Scalar>
    enable_if_t<sizeof(Scalar) == sizeof(uint32_t), void> set(const uint ix, const Scalar value) {accessor.set(ix, asuint(value));}
    template<typename Scalar>    
    enable_if_t<sizeof(Scalar) == sizeof(uint32_t), void> set(const uint ix, const vector <Scalar, 2> value) {
        accessor.set(ix, asuint(value.x));
        accessor.set(ix + stride, asuint(value.y));
    }
    template<typename Scalar>
    enable_if_t<sizeof(Scalar) == sizeof(uint32_t), void> set(const uint ix, const <Scalar, 3> value)  {
        accessor.set(ix, asuint(value.x));
        accessor.set(ix + stride, asuint(value.y));
        accessor.set(ix + 2 * stride, asuint(value.z));
    }
    template<typename Scalar>
    enable_if_t<sizeof(Scalar) == sizeof(uint32_t), void> set(const uint ix, const <Scalar, 4> value) {
        accessor.set(ix, asuint(value.x));
        accessor.set(ix + stride, asuint(value.y));
        accessor.set(ix + 2 * stride, asuint(value.z));
        accessor.set(ix + 3 * stride, asuint(value.w));
    }

    void atomicAnd(const uint ix, const uint value, NBL_REF_ARG(uint) orig) {
       orig = accessor.atomicAnd(ix, value);
    }
    void atomicAnd(const uint ix, const int value, NBL_REF_ARG(int) orig) {
       orig = asint(accessor.atomicAnd(ix, asuint(value)));
    }
    void atomicAnd(const uint ix, const float value, NBL_REF_ARG(float) orig) {
       orig = asfloat(accessor.atomicAnd(ix, asuint(value)));
    }
    void atomicOr(const uint ix, const uint value, NBL_REF_ARG(uint) orig) {
       orig = accessor.atomicOr(ix, value);
    }
    void atomicOr(const uint ix, const int value, NBL_REF_ARG(int) orig) {
       orig = asint(accessor.atomicOr(ix, asuint(value)));
    }
    void atomicOr(const uint ix, const float value, NBL_REF_ARG(float) orig) {
       orig = asfloat(accessor.atomicOr(ix, asuint(value)));
    }
    void atomicXor(const uint ix, const uint value, NBL_REF_ARG(uint) orig) {
       orig = accessor.atomicXor(ix, value);
    }
    void atomicXor(const uint ix, const int value, NBL_REF_ARG(int) orig) {
       orig = asint(accessor.atomicXor(ix, asuint(value)));
    }
    void atomicXor(const uint ix, const float value, NBL_REF_ARG(float) orig) {
       orig = asfloat(accessor.atomicXor(ix, asuint(value)));
    }
    void atomicAdd(const uint ix, const uint value, NBL_REF_ARG(uint) orig) {
       orig = accessor.atomicAdd(ix, value);
    }
    void atomicMin(const uint ix, const uint value, NBL_REF_ARG(uint) orig) {
       orig = accessor.atomicMin(ix, value);
    }
    void atomicMax(const uint ix, const uint value, NBL_REF_ARG(uint) orig) {
       orig = accessor.atomicMax(ix, value);
    }
    void atomicExchange(const uint ix, const uint value, NBL_REF_ARG(uint) orig) {
       orig = accessor.atomicExchange(ix, value);
    }
    void atomicCompSwap(const uint ix, const uint value, const uint comp, NBL_REF_ARG(uint) orig) {
       orig = accessor.atomicCompSwap(ix, comp, value);
    }
    
    // TODO: figure out the `enable_if` syntax for this
    void workgroupExecutionAndMemoryBarrier() {
        accessor.workgroupExecutionAndMemoryBarrier();
    }
};

// ---------------------------------------------- Offset Accessor ----------------------------------------------------

template<class BaseAccessor, uint32_t Offset>
struct OffsetAccessor
{
    BaseAccessor accessor;

    template <typename T>
    void set(uint32_t idx, T value) {accessor.set(idx + Offset, value);}

    template <typename T> 
    void get(uint32_t idx, NBL_REF_ARG(T) value) {accessor.get(idx + Offset, value);}

    // TODO: figure out the `enable_if` syntax for this
    void workgroupExecutionAndMemoryBarrier() {accessor.workgroupExecutionAndMemoryBarrier();}
};

// Dynamic offset version
template<class BaseAccessor>
struct DynamicOffsetAccessor
{
    BaseAccessor accessor;
    uint32_t offset;

    template <typename T>
    void set(uint32_t idx, T value) {accessor.set(idx + offset, value);}

    template <typename T> 
    void get(uint32_t idx, NBL_REF_ARG(T) value) {accessor.get(idx + offset, value);}

    // TODO: figure out the `enable_if` syntax for this
    void workgroupExecutionAndMemoryBarrier() {accessor.workgroupExecutionAndMemoryBarrier();}
};

}
}

#endif