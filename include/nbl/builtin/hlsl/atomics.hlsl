// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_ATOMICS_INCLUDED_
#define _NBL_BUILTIN_HLSL_ATOMICS_INCLUDED_

namespace nbl
{
namespace hlsl
{
namespace atomics
{
	template<typename T>
	T atomicAdd(inout T mem, T data)
	{
		T orig;
		InterlockedAdd(mem, data, orig);
		return orig;
	}
	
	template<typename T>
	T atomicAnd(inout T mem, T data)
	{
		T orig;
		InterlockedAnd(mem, data, orig);
		return orig;
	}
	
	template<typename T>
	T atomicOr(inout T mem, T data)
	{
		T orig;
		InterlockedOr(mem, data, orig);
		return orig;
	}
	
	template<typename T>
	T atomicXor(inout T mem, T data)
	{
		T orig;
		InterlockedXor(mem, data, orig);
		return orig;
	}
	
	template<typename T>
	T atomicMin(inout T mem, T data)
	{
		T orig;
		InterlockedMin(mem, data, orig);
		return orig;
	}
	
	template<typename T>
	T atomicMax(inout T mem, T data)
	{
		T orig;
		InterlockedMax(mem, data, orig);
		return orig;
	}
	
	template<typename T>
	T atomicExchange(inout T mem, T data)
	{
		T orig;
		InterlockedExchange(mem, data, orig);
		return orig;
	}
	
	template<typename T>
	T atomicCompSwap(inout T mem, T compare, T data)
	{
		T orig;
		InterlockedCompareExchange(mem, compare, data, orig);
		return orig;
	}
}
}
}
#endif