// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_ARITHMETIC_PORTABILITY_IMPL_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_ARITHMETIC_PORTABILITY_IMPL_INCLUDED_

#include "nbl/builtin/hlsl/glsl_compat/arithmetic.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/shuffle.hlsl"
#include "nbl/builtin/hlsl/binops.hlsl"
#include "nbl/builtin/hlsl/subgroup/ballot.hlsl"

namespace nbl
{
namespace hlsl
{
namespace subgroup
{

#ifdef NBL_GL_KHR_shader_subgroup_arithmetic
namespace native
{

template<typename T, class Binop>
struct reduction;
template<typename T, class Binop>
struct exclusive_scan;
template<typename T, class Binop>
struct inclusive_scan;

// *** AND ***
template<typename T>
struct reduction<T, binops::bitwise_and<T> >
{
    T operator()(const T x)
    {
        return glsl::arithmetic::subgroupAnd(x);
    }
};

template<typename T>
struct inclusive_scan<T, binops::bitwise_and<T> >
{
    T operator()(const T x)
    {
        return glsl::arithmetic::subgroupInclusiveAnd(x);
    }
};

template<typename T>
struct exclusive_scan<T, binops::bitwise_and<T> >
{
    T operator()(const T x)
    {
        return glsl::arithmetic::subgroupExclusiveAnd(x);
    }
};

// *** OR ***
template<typename T>
struct reduction<T, binops::bitwise_or<T> >
{
    T operator()(const T x)
    {
        return glsl::arithmetic::subgroupOr(x);
    }
};

template<typename T>
struct inclusive_scan<T, binops::bitwise_or<T> >
{
    T operator()(const T x)
    {
        return glsl::arithmetic::subgroupInclusiveOr(x);
    }
};

template<typename T>
struct exclusive_scan<T, binops::bitwise_or<T> >
{
    T operator()(const T x)
    {
        return glsl::arithmetic::subgroupExclusiveOr(x);
    }
};

// *** XOR ***
template<typename T>
struct reduction<T, binops::bitwise_xor<T> >
{
    T operator()(const T x)
    {
        return glsl::arithmetic::subgroupXor(x);
    }
};

template<typename T>
struct inclusive_scan<T, binops::bitwise_xor<T> >
{
    T operator()(const T x)
    {
        return glsl::arithmetic::subgroupInclusiveXor(x);
    }
};

template<typename T>
struct exclusive_scan<T, binops::bitwise_xor<T> >
{
    T operator()(const T x)
    {
        return glsl::arithmetic::subgroupExclusiveXor(x);
    }
};

// *** ADD ***
template<typename T>
struct reduction<T, binops::add<T> >
{
    T operator()(const T x)
    {
        return glsl::arithmetic::subgroupAdd(x);
    }
};
template<typename T>
struct inclusive_scan<T, binops::add<T> >
{
    T operator()(const T x)
    {
        return glsl::arithmetic::subgroupInclusiveAdd(x);
    }
};
template<typename T>
struct exclusive_scan<T, binops::add<T> >
{
    T operator()(const T x)
    {
        return glsl::arithmetic::subgroupExclusiveAdd(x);
    }
};

// *** MUL ***
template<typename T>
struct reduction<T, binops::mul<T> >
{
    T operator()(const T x)
    {
        return glsl::arithmetic::subgroupMul(x);
    }
};
template<typename T>
struct exclusive_scan<T, binops::mul<T> >
{
    T operator()(const T x)
    {
        return glsl::arithmetic::subgroupInclusiveMul(x);
    }
};
template<typename T>
struct inclusive_scan<T, binops::mul<T> >
{
    T operator()(const T x)
    {
        return glsl::arithmetic::subgroupExclusiveMul(x);
    }
};

// *** MIN ***
template<typename T>
struct reduction<T, binops::min<T> >
{
    T operator()(const T x)
    {
        return glsl::arithmetic::subgroupMin(x);
    }
};

template<>
struct inclusive_scan<int, binops::min<int> >
{
    int operator()(const int x)
    {
        return glsl::arithmetic::subgroupInclusiveMin(x);
    }
};

template<>
struct inclusive_scan<uint, binops::min<uint> >
{
    uint operator()(const uint x)
    {
        return glsl::arithmetic::subgroupInclusiveMin(x);
    }
};

template<>
struct inclusive_scan<uint, binops::min<float> >
{
    float operator()(const float x)
    {
        return glsl::arithmetic::subgroupInclusiveMin(x);
    }
};

template<>
struct exclusive_scan<int, binops::min<int> >
{
    int operator()(const int x)
    {
        return glsl::arithmetic::subgroupExclusiveMin(x);
    }
};

template<>
struct exclusive_scan<uint, binops::min<uint> >
{
    uint operator()(const uint x)
    {
        return glsl::arithmetic::subgroupExclusiveMin(x);
    }
};

template<>
struct exclusive_scan<uint, binops::min<float> >
{
    float operator()(const float x)
    {
        return glsl::arithmetic::subgroupExclusiveMin(x);
    }
};

// *** MAX ***
template<typename T>
struct reduction<T, binops::max<T> >
{
    T operator()(const T x)
    {
        return glsl::arithmetic::subgroupMax(x);
    }
};

template<>
struct inclusive_scan<int, binops::max<int> >
{
    int operator()(const int x)
    {
        return glsl::arithmetic::subgroupInclusiveMax(x);
    }
};

template<>
struct inclusive_scan<uint, binops::max<uint> >
{
    uint operator()(const uint x)
    {
        return glsl::arithmetic::subgroupInclusiveMax(x);
    }
};

template<>
struct inclusive_scan<uint, binops::max<float> >
{
    float operator()(const float x)
    {
        return glsl::arithmetic::subgroupInclusiveMax(x);
    }
};

template<>
struct exclusive_scan<int, binops::max<int> >
{
    int operator()(const int x)
    {
        return glsl::arithmetic::subgroupExclusiveMax(x);
    }
};

template<>
struct exclusive_scan<uint, binops::max<uint> >
{
    uint operator()(const uint x)
    {
        return glsl::arithmetic::subgroupExclusiveMax(x);
    }
};

template<>
struct exclusive_scan<uint, binops::max<float> >
{
    float operator()(const float x)
    {
        return glsl::arithmetic::subgroupExclusiveMax(x);
    }
};

}
#else
namespace portability
{
	
// WARNING
// THIS PORTABILITY IMPLEMENTATION USES SHUFFLE OPS
// Shuffles where you attempt to read an invactive lane, return garbage, 
// which means that our portability reductions and prefix sums will also return garbage/UB/UV
// Always use the native subgroup_arithmetic extensions if supported
	
template<typename T, class Binop>
struct inclusive_scan
{
    T operator()(T value)
    {
		Binop op;
		const uint subgroupInvocation = glsl::subgroup::gl_SubgroupInvocationID();
		const uint halfSubgroupSize = glsl::subgroup::gl_SubgroupSize() >> 1u;
		
		uint rhs = glsl::shuffle::subgroupShuffleUp(value, 1u); // all invocations must execute the shuffle, even if we don't apply the op() to all of them
		value = op(value, subgroupInvocation < 1u ? Binop::identity() : rhs);
		
		[[unroll(5)]]
		for (uint step=2u; step <= halfSubgroupSize; step <<= 1u)
		{
			rhs = glsl::shuffle::subgroupShuffleUp(value, step);
			value = op(value, subgroupInvocation < step ? Binop::identity() : rhs);
		}
		return value;
    }
};

template<typename T, class Binop>
struct exclusive_scan
{

    T operator()(T value)
    {
		const uint subgroupInvocation = glsl::subgroup::gl_SubgroupInvocationID();
		
		value = impl(value);
		// store value to smem so we can shuffle it
		uint left = glsl::shuffle::subgroupShuffleUp(value, 1);
		value = subgroupInvocation >= 1 ? left : Binop::identity(); // the first invocation doesn't have anything in its left so we set to the binop's identity value for exlusive scan
		// return it
		return value;
    }

// protected:
	inclusive_scan<T, Binop> impl;
};

template<typename T, class Binop>
struct reduction
{
    T operator()(T value)
    {
		value = impl(value);
		value = glsl::ballot::subgroupBroadcast(value, LastSubgroupInvocation()); // take the last subgroup invocation's value for the reduction
		return value;
    }
	
// protected:
    inclusive_scan<T, Binop> impl;
};
}
#endif
}
}
}

#endif