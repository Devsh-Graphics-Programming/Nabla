// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_GLSL_COMPAT_ATOMICS_INCLUDED_
#define _NBL_BUILTIN_HLSL_GLSL_COMPAT_ATOMICS_INCLUDED_

#include "nbl/builtin/hlsl/spirv_intrinsics/atomics.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace glsl
{
namespace atomics
{

template<typename T>
T atomicAdd(inout T ptr, T value)
{
	return spirv::atomicAdd(ptr, 1, 0, value);
}
template<typename T>
T atomicAnd(inout T ptr, T value)
{
	return spirv::atomicAnd(ptr, 1, 0, value);
}
template<typename T>
T atomicOr(inout T ptr, T value)
{
	return spirv::atomicOr(ptr, 1, 0, value);
}
template<typename T>
T atomicXor(inout T ptr, T value)
{
	return spirv::atomicXor(ptr, 1, 0, value);
}
template<typename T>
T atomicMin(inout T ptr, T value)
{
	return spirv::atomicMin(ptr, 1, 0, value);
}
template<typename T>
T atomicMax(inout T ptr, T value)
{
	return spirv::atomicMax(ptr, 1, 0, value);
}
template<typename T>
T atomicExchange(inout T ptr, T value)
{
	return spirv::atomicExchange(ptr, 1, 0, value);
}
template<typename T>
T atomicCompSwap(inout T ptr, T comparator, T value)
{
	return spirv::atomicCompSwap(ptr, 1, 0, 0, value, comparator);
}

}
}
}
}

#endif