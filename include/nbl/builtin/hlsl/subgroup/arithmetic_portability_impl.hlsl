// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_ARITHMETIC_PORTABILITY_IMPL_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_ARITHMETIC_PORTABILITY_IMPL_INCLUDED_

#include "nbl/builtin/hlsl/glsl_compat/subgroup_arithmetic.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_shuffle.hlsl"
#include "nbl/builtin/hlsl/functional.hlsl"
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
struct reduction<T, bit_and<T> >
{
    T operator()(NBL_CONST_REF_ARG(T)x)
    {
        return glsl::subgroupAnd<T>(x);
    }
};

template<typename T>
struct inclusive_scan<T, bit_and<T> >
{
    T operator()(NBL_CONST_REF_ARG(T)x)
    {
        return glsl::subgroupInclusiveAnd<T>(x);
    }
};

template<typename T>
struct exclusive_scan<T, bit_and<T> >
{
    T operator()(NBL_CONST_REF_ARG(T)x)
    {
        return glsl::subgroupExclusiveAnd<T>(x);
    }
};

// *** OR ***
template<typename T>
struct reduction<T, bit_or<T> >
{
    T operator()(NBL_CONST_REF_ARG(T)x)
    {
        return glsl::subgroupOr<T>(x);
    }
};

template<typename T>
struct inclusive_scan<T, bit_or<T> >
{
    T operator()(NBL_CONST_REF_ARG(T)x)
    {
        return glsl::subgroupInclusiveOr<T>(x);
    }
};

template<typename T>
struct exclusive_scan<T, bit_or<T> >
{
    T operator()(NBL_CONST_REF_ARG(T)x)
    {
        return glsl::subgroupExclusiveOr<T>(x);
    }
};

// *** XOR ***
template<typename T>
struct reduction<T, bit_xor<T> >
{
    T operator()(NBL_CONST_REF_ARG(T)x)
    {
        return glsl::subgroupXor<T>(x);
    }
};

template<typename T>
struct inclusive_scan<T, bit_xor<T> >
{
    T operator()(NBL_CONST_REF_ARG(T)x)
    {
        return glsl::subgroupInclusiveXor<T>(x);
    }
};

template<typename T>
struct exclusive_scan<T, bit_xor<T> >
{
    T operator()(NBL_CONST_REF_ARG(T)x)
    {
        return glsl::subgroupExclusiveXor<T>(x);
    }
};

// *** ADD ***
template<typename T>
struct reduction<T, plus<T> >
{
    T operator()(NBL_CONST_REF_ARG(T)x)
    {
        return glsl::subgroupAdd<T>(x);
    }
};
template<typename T>
struct inclusive_scan<T, plus<T> >
{
    T operator()(NBL_CONST_REF_ARG(T)x)
    {
        return glsl::subgroupInclusiveAdd<T>(x);
    }
};
template<typename T>
struct exclusive_scan<T, plus<T> >
{
    T operator()(NBL_CONST_REF_ARG(T)x)
    {
        return glsl::subgroupExclusiveAdd<T>(x);
    }
};

// *** MUL ***
template<typename T>
struct reduction<T, multiplies<T> >
{
    T operator()(NBL_CONST_REF_ARG(T)x)
    {
        return glsl::subgroupMul<T>(x);
    }
};
template<typename T>
struct exclusive_scan<T, multiplies<T> >
{
    T operator()(NBL_CONST_REF_ARG(T)x)
    {
        return glsl::subgroupInclusiveMul<T>(x);
    }
};
template<typename T>
struct inclusive_scan<T, multiplies<T> >
{
    T operator()(NBL_CONST_REF_ARG(T)x)
    {
        return glsl::subgroupExclusiveMul<T>(x);
    }
};

// *** MIN ***
template<typename T>
struct reduction<T, mininum<T> >
{
    T operator()(NBL_CONST_REF_ARG(T)x)
    {
        return glsl::subgroupMin<T>(x);
    }
};

template<>
struct inclusive_scan<int, mininum<int> >
{
    int operator()(const int x)
    {
        return glsl::subgroupInclusiveMin<T>(x);
    }
};

template<>
struct inclusive_scan<uint, mininum<uint> >
{
    uint operator()(const uint x)
    {
        return glsl::subgroupInclusiveMin<T>(x);
    }
};

template<>
struct inclusive_scan<uint, mininum<float> >
{
    float operator()(const float x)
    {
        return glsl::subgroupInclusiveMin<T>(x);
    }
};

template<>
struct exclusive_scan<int, mininum<int> >
{
    int operator()(const int x)
    {
        return glsl::subgroupExclusiveMin<T>(x);
    }
};

template<>
struct exclusive_scan<uint, mininum<uint> >
{
    uint operator()(const uint x)
    {
        return glsl::subgroupExclusiveMin<T>(x);
    }
};

template<>
struct exclusive_scan<uint, mininum<float> >
{
    float operator()(const float x)
    {
        return glsl::subgroupExclusiveMin<T>(x);
    }
};

// *** MAX ***
template<typename T>
struct reduction<T, maximum<T> >
{
    T operator()(NBL_CONST_REF_ARG(T)x)
    {
        return glsl::subgroupMax<T>(x);
    }
};

template<>
struct inclusive_scan<int, maximum<int> >
{
    int operator()(const int x)
    {
        return glsl::subgroupInclusiveMax<T>(x);
    }
};

template<>
struct inclusive_scan<uint, maximum<uint> >
{
    uint operator()(const uint x)
    {
        return glsl::subgroupInclusiveMax<T>(x);
    }
};

template<>
struct inclusive_scan<uint, maximum<float> >
{
    float operator()(const float x)
    {
        return glsl::subgroupInclusiveMax<T>(x);
    }
};

template<>
struct exclusive_scan<int, maximum<int> >
{
    int operator()(const int x)
    {
        return glsl::subgroupExclusiveMax<T>(x);
    }
};

template<>
struct exclusive_scan<uint, maximum<uint> >
{
    uint operator()(const uint x)
    {
        return glsl::subgroupExclusiveMax<T>(x);
    }
};

template<>
struct exclusive_scan<uint, maximum<float> >
{
    float operator()(const float x)
    {
        return glsl::subgroupExclusiveMax<T>(x);
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
T inclusive_scan(NBL_CONST_REF_ARG(T) value)
{
    Binop op;
    const uint subgroupInvocation = glsl::gl_SubgroupInvocationID();
    const uint halfSubgroupSize = glsl::gl_SubgroupSize() >> 1u;
    
    uint rhs = glsl::subgroupShuffleUp<T>(value, 1u); // all invocations must execute the shuffle, even if we don't apply the op() to all of them
    value = op(value, subgroupInvocation < 1u ? Binop::identity() : rhs);
    
    [[unroll(5)]]
    for (uint step=2u; step <= halfSubgroupSize; step <<= 1u)
    {
        rhs = glsl::subgroupShuffleUp<T>(value, step);
        value = op(value, subgroupInvocation < step ? Binop::identity() : rhs);
    }
    return value;
}

template<typename T, class Binop>
T exclusive_scan(NBL_CONST_REF_ARG(T) value)
{
    const uint subgroupInvocation = glsl::gl_SubgroupInvocationID();
    value = inclusive_scan<T, Binop>(value);
    // store value to smem so we can shuffle it
    uint left = glsl::subgroupShuffleUp<T>(value, 1);
    value = subgroupInvocation >= 1 ? left : Binop::identity(); // the first invocation doesn't have anything in its left so we set to the binop's identity value for exlusive scan
    // return it
    return value;
}

template<typename T, class Binop>
T reduction(NBL_CONST_REF_ARG(T) value)
{
    value = inclusive_scan<T, Binop>(value);
    value = glsl::subgroupBroadcast<T>(value, LastSubgroupInvocation()); // take the last subgroup invocation's value for the reduction
    return value;
}
}
#endif
}
}
}

#endif