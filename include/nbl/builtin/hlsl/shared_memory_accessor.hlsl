// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SHARED_MEMORY_ACCESSOR_INCLUDED_
#define _NBL_BUILTIN_HLSL_SHARED_MEMORY_ACCESSOR_INCLUDED_

#include "nbl/builtin/hlsl/atomics.hlsl"

namespace nbl 
{
namespace hlsl
{

#ifdef SHARED_MEM
struct MainScratchProxy
{
	uint get(uint ix)
	{
		return SHARED_MEM[ix];
	}

	void set(uint ix, uint value)
	{
		SHARED_MEM[ix] = value;
	}
};
#else
#error "Must #define scratch memory array as SHARED_MEM"
#endif

#ifdef BROADCAST_MEM
struct BroadcastScratchProxy
{
	uint get(uint ix)
	{
		return BROADCAST_MEM[ix];
	}

	void set(uint ix, uint value)
	{
		BROADCAST_MEM[ix] = value;
	}
};
#else
struct BroadcastScratchProxy {};
#endif

#ifdef SHUFFLE_MEM 
struct ShuffleScratchProxy
{
	uint get(uint ix)
	{
		return SHUFFLE_MEM[ix];
	}

	void set(uint ix, uint value)
	{
		SHUFFLE_MEM[ix] = value;
	}
};
#else
struct ShuffleScratchProxy
{};
#endif

// REVIEW:  Bank conflict avoidance. It seems the offset for each index should be SUBGROUP_SIZE,
//          not WORKGROUP_SIZE (assuming banks == SUBGROUP_SIZE)

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
    
    // REVIEW:  Depending on how we handle SHARED_MEM in nbl::hlsl::atomics 
	//			we should also update this part to handle float and int
    
    // uint atomics
    void atomicAdd(const uint ix, const uint value, out uint orig) {
	   orig = atomics::atomicAdd(ix, value);
	}
	void atomicAnd(const uint ix, const uint value, out uint orig) {
	   orig = atomics::atomicAnd(ix, value);
	}
	void atomicOr(const uint ix, const uint value, out uint orig) {
	   orig = atomics::atomicOr(ix, value);
	}
	void atomicXor(const uint ix, const uint value, out uint orig) {
	   orig = atomics::atomicXor(ix, value);
	}
	void atomicMin(const uint ix, const uint value, out uint orig) {
	   orig = atomics::atomicMin(ix, value);
	}
	void atomicMax(const uint ix, const uint value, out uint orig) {
	   orig = atomics::atomicMax(ix, value);
	}
	void atomicExchange(const uint ix, const uint value, out uint orig) {
	   orig = atomics::atomicExchange(ix, value);
	}
	void atomicCompSwap(const uint ix, const uint value, const uint comp, out uint orig) {
	   orig = atomics::atomicCompSwap(ix, comp, value);
	}
};

struct SharedMemory
{
	SharedMemoryAdaptor<MainScratchProxy> main;
	SharedMemoryAdaptor<BroadcastScratchProxy> broadcast;
	SharedMemoryAdaptor<ShuffleScratchProxy> shuffle;
};

}
}

#endif