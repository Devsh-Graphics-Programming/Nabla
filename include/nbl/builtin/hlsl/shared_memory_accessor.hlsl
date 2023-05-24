// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SHARED_MEMORY_ACCESSOR_INCLUDED_
#define _NBL_BUILTIN_HLSL_SHARED_MEMORY_ACCESSOR_INCLUDED_

#include "nbl/builtin/hlsl/atomics.hlsl"

// REVIEW: Where would _NBL_HLSL_WORKGROUP_SIZE_ be defined?

namespace nbl 
{
namespace hlsl
{

// REVIEW:  Bank conflict avoidance. It seems the offset for each index should be SUBGROUP_SIZE,
//          not WORKGROUP_SIZE (assuming banks == SUBGROUP_SIZE)

#ifdef SHARED_MEM
struct MemProxy
{
	uint get(uint ix)
	{
		return SHARED_MEM[ix];
	}

	void set(uint ix, uint value)
	{
		SHARED_MEM[ix] = value;
	}

	uint atomicAdd(uint ix, uint value)
	{
        return atomics::atomicAdd(SHARED_MEM[ix], value);
	}
	
	uint atomicAnd(uint ix, uint value)
	{
        return atomics::atomicAnd(SHARED_MEM[ix], value);
	}
	
	uint atomicOr(uint ix, uint value)
	{
        return atomics::atomicOr(SHARED_MEM[ix], value);
	}
	
	uint atomicXor(uint ix, uint value)
	{
        return atomics::atomicXor(SHARED_MEM[ix], value);
	}
	
	uint atomicMin(uint ix, uint value)
	{
        return atomics::atomicMin(SHARED_MEM[ix], value);
	}
	
	uint atomicMax(uint ix, uint value)
	{
        return atomics::atomicMax(SHARED_MEM[ix], value);
	}
	
	uint atomicExchange(uint ix, uint value)
	{
        return atomics::atomicExchange(SHARED_MEM[ix], value);
	}
	
	uint atomicCompSwap(uint ix, const uint comp, uint value)
	{
        return atomics::atomicCompSwap(SHARED_MEM[ix], comp, value);
	}
};
#else
#error "Must #define scratch memory array as SHARED_MEM"
#endif

template<class NumberSharedMemoryAccessor>
struct SharedMemoryAdaptor
{
	NumberSharedMemoryAccessor accessor;

    uint get(const uint ix) { return accessor.get(ix); }
	void get(const uint ix, out uint value) { value = accessor.get(ix);}
    void get(const uint ix, out uint2 value) { value = uint2(accessor.get(ix), accessor.get(ix + _NBL_HLSL_WORKGROUP_SIZE_));}
    void get(const uint ix, out uint3 value) { value = uint3(accessor.get(ix), accessor.get(ix + _NBL_HLSL_WORKGROUP_SIZE_), accessor.get(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_));}
    void get(const uint ix, out uint4 value) { value = uint4(accessor.get(ix), accessor.get(ix + _NBL_HLSL_WORKGROUP_SIZE_), accessor.get(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_), accessor.get(ix + 3 * _NBL_HLSL_WORKGROUP_SIZE_));}

    void get(const uint ix, out int value) { value = asint(accessor.get(ix));}
    void get(const uint ix, out int2 value) { value = asint(uint2(accessor.get(ix), accessor.get(ix + _NBL_HLSL_WORKGROUP_SIZE_)));}
    void get(const uint ix, out int3 value) { value = asint(uint3(accessor.get(ix), accessor.get(ix + _NBL_HLSL_WORKGROUP_SIZE_), accessor.get(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_)));}
    void get(const uint ix, out int4 value) { value = asint(uint4(accessor.get(ix), accessor.get(ix + _NBL_HLSL_WORKGROUP_SIZE_), accessor.get(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_), accessor.get(ix + 3 * _NBL_HLSL_WORKGROUP_SIZE_)));}

    void get(const uint ix, out float value) { value = asfloat(accessor.get(ix));}
    void get(const uint ix, out float2 value) { value = asfloat(uint2(accessor.get(ix), accessor.get(ix + _NBL_HLSL_WORKGROUP_SIZE_)));}
    void get(const uint ix, out float3 value) { value = asfloat(uint3(accessor.get(ix), accessor.get(ix + _NBL_HLSL_WORKGROUP_SIZE_), accessor.get(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_)));}
    void get(const uint ix, out float4 value) { value = asfloat(uint4(accessor.get(ix), accessor.get(ix + _NBL_HLSL_WORKGROUP_SIZE_), accessor.get(ix + 2 * _NBL_HLSL_WORKGROUP_SIZE_), accessor.get(ix + 3 * _NBL_HLSL_WORKGROUP_SIZE_)));}

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
    
    // TODO (PentaKon): Need to handle other types apart from uint
    
    // uint atomics
    void atomicAdd(const uint ix, const uint value, out uint orig) {
	   orig = accessor.atomicAdd(ix, value);
	}
	void atomicAnd(const uint ix, const uint value, out uint orig) {
	   orig = accessor.atomicAnd(ix, value);
	}
	void atomicOr(const uint ix, const uint value, out uint orig) {
	   orig = accessor.atomicOr(ix, value);
	}
	void atomicXor(const uint ix, const uint value, out uint orig) {
	   orig = accessor.atomicXor(ix, value);
	}
	void atomicMin(const uint ix, const uint value, out uint orig) {
	   orig = accessor.atomicMin(ix, value);
	}
	void atomicMax(const uint ix, const uint value, out uint orig) {
	   orig = accessor.atomicMax(ix, value);
	}
	void atomicExchange(const uint ix, const uint value, out uint orig) {
	   orig = accessor.atomicExchange(ix, value);
	}
	void atomicCompSwap(const uint ix, const uint value, const uint comp, out uint orig) {
	   orig = accessor.atomicCompSwap(ix, comp, value);
	}
};

}
}

#endif