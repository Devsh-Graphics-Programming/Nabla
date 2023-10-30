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
template<class NumberMemoryAccessor>
struct MemoryAdaptor
{
    NumberMemoryAccessor accessor;
    
    uint get(const uint ix) { return accessor.get(ix); }
    void get(const uint ix, NBL_REF_ARG(uint) value) { value = accessor.get(ix);}
    void get(const uint ix, NBL_REF_ARG(uint2) value) { value = uint2(accessor.get(ix), accessor.get(ix + _NBL_HLSL_WORKGROUP_SIZE_));}
    void get(const uint ix, NBL_REF_ARG(uint3) value) { value = uint3(accessor.get(ix), accessor.get(ix + _NBL_HLSL_WORKGROUP_SIZE_), accessor.get(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_));}
    void get(const uint ix, NBL_REF_ARG(uint4) value) { value = uint4(accessor.get(ix), accessor.get(ix + _NBL_HLSL_WORKGROUP_SIZE_), accessor.get(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_), accessor.get(ix + 3 * _NBL_HLSL_WORKGROUP_SIZE_));}

    void get(const uint ix, NBL_REF_ARG(int) value) { value = asint(accessor.get(ix));}
    void get(const uint ix, NBL_REF_ARG(int2) value) { value = asint(uint2(accessor.get(ix), accessor.get(ix + _NBL_HLSL_WORKGROUP_SIZE_)));}
    void get(const uint ix, NBL_REF_ARG(int3) value) { value = asint(uint3(accessor.get(ix), accessor.get(ix + _NBL_HLSL_WORKGROUP_SIZE_), accessor.get(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_)));}
    void get(const uint ix, NBL_REF_ARG(int4) value) { value = asint(uint4(accessor.get(ix), accessor.get(ix + _NBL_HLSL_WORKGROUP_SIZE_), accessor.get(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_), accessor.get(ix + 3 * _NBL_HLSL_WORKGROUP_SIZE_)));}

    void get(const uint ix, NBL_REF_ARG(float) value) { value = asfloat(accessor.get(ix));}
    void get(const uint ix, NBL_REF_ARG(float2) value) { value = asfloat(uint2(accessor.get(ix), accessor.get(ix + _NBL_HLSL_WORKGROUP_SIZE_)));}
    void get(const uint ix, NBL_REF_ARG(float3) value) { value = asfloat(uint3(accessor.get(ix), accessor.get(ix + _NBL_HLSL_WORKGROUP_SIZE_), accessor.get(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_)));}
    void get(const uint ix, NBL_REF_ARG(float4) value) { value = asfloat(uint4(accessor.get(ix), accessor.get(ix + _NBL_HLSL_WORKGROUP_SIZE_), accessor.get(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_), accessor.get(ix + 3 * _NBL_HLSL_WORKGROUP_SIZE_)));}

    void set(const uint ix, const uint value) {accessor.set(ix, value);}
    void set(const uint ix, const uint2 value) {
        accessor.set(ix, value.x);
        accessor.set(ix + _NBL_HLSL_WORKGROUP_SIZE_, value.y);
    }
    void set(const uint ix, const uint3 value)  {
        accessor.set(ix, value.x);
        accessor.set(ix + _NBL_HLSL_WORKGROUP_SIZE_, value.y);
        accessor.set(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_, value.z);
    }
    void set(const uint ix, const uint4 value) {
        accessor.set(ix, value.x);
        accessor.set(ix + _NBL_HLSL_WORKGROUP_SIZE_, value.y);
        accessor.set(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_, value.z);
        accessor.set(ix + 3 * _NBL_HLSL_WORKGROUP_SIZE_, value.w);
    }

    void set(const uint ix, const int value) {accessor.set(ix, asuint(value));}
    void set(const uint ix, const int2 value) {
        accessor.set(ix, asuint(value.x));
        accessor.set(ix + _NBL_HLSL_WORKGROUP_SIZE_, asuint(value.y));
    }
    void set(const uint ix, const int3 value)  {
        accessor.set(ix, asuint(value.x));
        accessor.set(ix + _NBL_HLSL_WORKGROUP_SIZE_, asuint(value.y));
        accessor.set(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_, asuint(value.z));
    }
    void set(const uint ix, const int4 value) {
        accessor.set(ix, asuint(value.x));
        accessor.set(ix + _NBL_HLSL_WORKGROUP_SIZE_, asuint(value.y));
        accessor.set(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_, asuint(value.z));
        accessor.set(ix + 3 * _NBL_HLSL_WORKGROUP_SIZE_, asuint(value.w));
    }

    void set(const uint ix, const float value) {accessor.set(ix, asuint(value));}
    void set(const uint ix, const float2 value) {
        accessor.set(ix, asuint(value.x));
        accessor.set(ix + _NBL_HLSL_WORKGROUP_SIZE_, asuint(value.y));
    }
    void set(const uint ix, const float3 value)  {
        accessor.set(ix, asuint(value.x));
        accessor.set(ix + _NBL_HLSL_WORKGROUP_SIZE_, asuint(value.y));
        accessor.set(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_, asuint(value.z));
    }
    void set(const uint ix, const float4 value) {
        accessor.set(ix, asuint(value.x));
        accessor.set(ix + _NBL_HLSL_WORKGROUP_SIZE_, asuint(value.y));
        accessor.set(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_, asuint(value.z));
        accessor.set(ix + 3 * _NBL_HLSL_WORKGROUP_SIZE_, asuint(value.w));
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
    
    void workgroupExecutionAndMemoryBarrier() {
        accessor.workgroupExecutionAndMemoryBarrier();
    }
};

}
}

#endif