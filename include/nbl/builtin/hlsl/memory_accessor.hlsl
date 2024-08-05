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

template<class BaseAccessor>
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
    void get(const uint ix, NBL_REF_ARG(Scalar) value) { accessor.get(ix, value);}
    template<typename Scalar>
    void get(const uint ix, NBL_REF_ARG(vector <Scalar, 2>) value) { accessor.get(ix, value.x), accessor.get(ix + _NBL_HLSL_WORKGROUP_SIZE_, value.y);}
    template<typename Scalar>    
    void get(const uint ix, NBL_REF_ARG(vector <Scalar, 3>) value) { accessor.get(ix, value.x), accessor.get(ix + _NBL_HLSL_WORKGROUP_SIZE_, value.y), accessor.get(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_, value.z);}
    template<typename Scalar>   
    void get(const uint ix, NBL_REF_ARG(vector <Scalar, 4>) value) { accessor.get(ix, value.x), accessor.get(ix + _NBL_HLSL_WORKGROUP_SIZE_, value.y), accessor.get(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_, value.z), accessor.get(ix + 3 * _NBL_HLSL_WORKGROUP_SIZE_, value.w);}

    template<typename Scalar>
    void set(const uint ix, const Scalar value) {accessor.set(ix, value);}
    template<typename Scalar>    
    void set(const uint ix, const vector <Scalar, 2> value) {
        accessor.set(ix, value.x);
        accessor.set(ix + _NBL_HLSL_WORKGROUP_SIZE_, value.y);
    }
    template<typename Scalar>
    void set(const uint ix, const <Scalar, 3> value)  {
        accessor.set(ix, value.x);
        accessor.set(ix + _NBL_HLSL_WORKGROUP_SIZE_, value.y);
        accessor.set(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_, value.z);
    }
    template<typename Scalar>
    void set(const uint ix, const <Scalar, 4> value) {
        accessor.set(ix, value.x);
        accessor.set(ix + _NBL_HLSL_WORKGROUP_SIZE_, value.y);
        accessor.set(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_, value.z);
        accessor.set(ix + 3 * _NBL_HLSL_WORKGROUP_SIZE_, value.w);
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

}
}

#endif