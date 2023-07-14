// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_ATOMICS_INCLUDED_
#define _NBL_BUILTIN_HLSL_ATOMICS_INCLUDED_

#ifndef SHARED_MEM
#error "Atomics need SHARED_MEM defined to point to uint groupshared array"
#else
namespace nbl
{
namespace hlsl
{
namespace atomics
{
	// REVIEW:  How should we handle shared memory access in here? Right now we assume SHARED_MEM is defined.
	// 			Additionally, is there a point in having templated type or should it be uint and the called 
	//			uses `asuint()` on different types before calling atomics? Note that Interlocked* API only works 
	//			on uint and int
	template<typename T>
	T atomicAdd(in uint ix, T data)
	{
		T orig;
		InterlockedAdd(SHARED_MEM[ix], data, orig);
		return orig;
	}
	
	template<typename T>
	T atomicAnd(in uint ix, T data)
	{
		T orig;
		InterlockedAnd(SHARED_MEM[ix], data, orig);
		return orig;
	}
	
	template<typename T>
	T atomicOr(in uint ix, T data)
	{
		T orig;
		InterlockedOr(SHARED_MEM[ix], data, orig);
		return orig;
	}
	
	template<typename T>
	T atomicXor(in uint ix, T data)
	{
		T orig;
		InterlockedXor(SHARED_MEM[ix], data, orig);
		return orig;
	}
	
	template<typename T>
	T atomicMin(in uint ix, T data)
	{
		T orig;
		InterlockedMin(SHARED_MEM[ix], data, orig);
		return orig;
	}
	
	template<typename T>
	T atomicMax(in uint ix, T data)
	{
		T orig;
		InterlockedMax(SHARED_MEM[ix], data, orig);
		return orig;
	}
	
	template<typename T>
	T atomicExchange(in uint ix, T data)
	{
		T orig;
		InterlockedExchange(SHARED_MEM[ix], data, orig);
		return orig;
	}
	
	template<typename T>
	T atomicCompSwap(in uint ix, T compare, T data)
	{
		T orig;
		InterlockedCompareExchange(SHARED_MEM[ix], compare, data, orig);
		return orig;
	}
}
}
}
#endif
#endif