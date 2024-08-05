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
        accessor.set(ix + Stride, value.y);
    }
    template<typename Scalar>
    void set(const uint ix, const <Scalar, 3> value)  {
        accessor.set(ix, value.x);
        accessor.set(ix + Stride, value.y);
        accessor.set(ix + 2 * Stride, value.z);
    }
    template<typename Scalar>
    void set(const uint ix, const <Scalar, 4> value) {
        accessor.set(ix, value.x);
        accessor.set(ix + Stride, value.y);
        accessor.set(ix + 2 * Stride, value.z);
        accessor.set(ix + 3 * Stride, value.w);
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
    
    // TODO: template all get,set, atomic... then add static_asserts of `has_method<BaseAccessor,signature>::value`, do vectors and matrices in terms of each other
    uint get(const uint ix) { return accessor.get(ix); }
    void get(const uint ix, NBL_REF_ARG(uint) value) { value = accessor.get(ix);}
    void get(const uint ix, NBL_REF_ARG(uint2) value) { value = uint2(accessor.get(ix), accessor.get(ix + stride));}
    void get(const uint ix, NBL_REF_ARG(uint3) value) { value = uint3(accessor.get(ix), accessor.get(ix + stride), accessor.get(ix + 2 * stride));}
    void get(const uint ix, NBL_REF_ARG(uint4) value) { value = uint4(accessor.get(ix), accessor.get(ix + stride), accessor.get(ix + 2 * stride), accessor.get(ix + 3 * stride));}

    void get(const uint ix, NBL_REF_ARG(int) value) { value = asint(accessor.get(ix));}
    void get(const uint ix, NBL_REF_ARG(int2) value) { value = asint(uint2(accessor.get(ix), accessor.get(ix + stride)));}
    void get(const uint ix, NBL_REF_ARG(int3) value) { value = asint(uint3(accessor.get(ix), accessor.get(ix + stride), accessor.get(ix + 2 * stride)));}
    void get(const uint ix, NBL_REF_ARG(int4) value) { value = asint(uint4(accessor.get(ix), accessor.get(ix + stride), accessor.get(ix + 2 * stride), accessor.get(ix + 3 * stride)));}

    void get(const uint ix, NBL_REF_ARG(float) value) { value = asfloat(accessor.get(ix));}
    void get(const uint ix, NBL_REF_ARG(float2) value) { value = asfloat(uint2(accessor.get(ix), accessor.get(ix + stride)));}
    void get(const uint ix, NBL_REF_ARG(float3) value) { value = asfloat(uint3(accessor.get(ix), accessor.get(ix + stride), accessor.get(ix + 2 * stride)));}
    void get(const uint ix, NBL_REF_ARG(float4) value) { value = asfloat(uint4(accessor.get(ix), accessor.get(ix + stride), accessor.get(ix + 2 * stride), accessor.get(ix + 3 * stride)));}

    void set(const uint ix, const uint value) {accessor.set(ix, value);}
    void set(const uint ix, const uint2 value) {
        accessor.set(ix, value.x);
        accessor.set(ix + stride, value.y);
    }
    void set(const uint ix, const uint3 value)  {
        accessor.set(ix, value.x);
        accessor.set(ix + stride, value.y);
        accessor.set(ix + 2 * stride, value.z);
    }
    void set(const uint ix, const uint4 value) {
        accessor.set(ix, value.x);
        accessor.set(ix + stride, value.y);
        accessor.set(ix + 2 * stride, value.z);
        accessor.set(ix + 3 * stride, value.w);
    }

    void set(const uint ix, const int value) {accessor.set(ix, asuint(value));}
    void set(const uint ix, const int2 value) {
        accessor.set(ix, asuint(value.x));
        accessor.set(ix + stride, asuint(value.y));
    }
    void set(const uint ix, const int3 value)  {
        accessor.set(ix, asuint(value.x));
        accessor.set(ix + stride, asuint(value.y));
        accessor.set(ix + 2 * stride, asuint(value.z));
    }
    void set(const uint ix, const int4 value) {
        accessor.set(ix, asuint(value.x));
        accessor.set(ix + stride, asuint(value.y));
        accessor.set(ix + 2 * stride, asuint(value.z));
        accessor.set(ix + 3 * stride, asuint(value.w));
    }

    void set(const uint ix, const float value) {accessor.set(ix, asuint(value));}
    void set(const uint ix, const float2 value) {
        accessor.set(ix, asuint(value.x));
        accessor.set(ix + stride, asuint(value.y));
    }
    void set(const uint ix, const float3 value)  {
        accessor.set(ix, asuint(value.x));
        accessor.set(ix + stride, asuint(value.y));
        accessor.set(ix + 2 * stride, asuint(value.z));
    }
    void set(const uint ix, const float4 value) {
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

template<class BaseAccessor, class AccessorType, uint32_t Offset>
struct OffsetAccessor
{
    BaseAccessor accessor;

    void set(uint32_t idx, NBL_REF_ARG(AccessorType) x) {accessor.set(idx + Offset, x);}

    AccessorType get(uint32_t idx) {return accessor.get(idx + Offset);}

    // TODO: figure out the `enable_if` syntax for this
    void workgroupExecutionAndMemoryBarrier() {accessor.workgroupExecutionAndMemoryBarrier();}
};

// Dynamic offset version
template<class BaseAccessor, class AccessorType>
struct DynamicOffsetAccessor
{
    BaseAccessor accessor;
    uint32_t offset;

    void set(uint32_t idx, NBL_REF_ARG(AccessorType) x) {accessor.set(idx + offset, x);}

    AccessorType get(uint32_t idx) {return accessor.get(idx + offset);}

    // TODO: figure out the `enable_if` syntax for this
    void workgroupExecutionAndMemoryBarrier() {accessor.workgroupExecutionAndMemoryBarrier();}
};

}
}

#endif