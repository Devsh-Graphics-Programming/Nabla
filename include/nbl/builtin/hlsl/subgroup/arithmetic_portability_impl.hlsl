// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_ARITHMETIC_PORTABILITY_IMPL_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_ARITHMETIC_PORTABILITY_IMPL_INCLUDED_


#include "nbl/builtin/hlsl/glsl_compat/subgroup_shuffle.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_arithmetic.hlsl"

#include "nbl/builtin/hlsl/subgroup/ballot.hlsl"

#include "nbl/builtin/hlsl/functional.hlsl"


// TODO: If you ever get errors from trying this stuff on a vector, its because `minimum` and `maximum` functionals are not implemented with `mix`
// TODO: split into two files
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

#define SPECIALIZE(NAME,BINOP,SUBGROUP_OP) template<typename T> struct NAME<T,BINOP<T> > { \
    T operator()(NBL_CONST_REF_ARG(T) v) {return glsl::subgroup##SUBGROUP_OP<T>(x);} \
}

#define SPECIALIZE_ALL(BINOP,SUBGROUP_OP) SPECIALIZE(reduction,BINOP,SUBGROUP_OP); \
    SPECIALIZE(inclusive_scan,BINOP,Inclusive##SUBGROUP_OP); \
    SPECIALIZE(exclusive_scan,BINOP,Exclusive##SUBGROUP_OP);

SPECIALIZE_ALL(bit_and,And);
SPECIALIZE_ALL(bit_or,Or);
SPECIALIZE_ALL(bit_xor,Xor);

SPECIALIZE_ALL(plus,Add);
SPECIALIZE_ALL(multiplies,Mul);

SPECIALIZE_ALL(minimum,Min);
SPECIALIZE_ALL(maximum,Max);

#undef SPECIALIZE_ALL
#undef SPECIALIZE

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
T inclusive_scan(T value)
{
    Binop op;
    const uint subgroupInvocation = glsl::gl_SubgroupInvocationID();
    const uint halfSubgroupSize = glsl::gl_SubgroupSize() >> 1u;
    
    uint rhs = glsl::subgroupShuffleUp<T>(value, 1u); // all invocations must execute the shuffle, even if we don't apply the op() to all of them
    value = op(value, subgroupInvocation<1u ? Binop::identity:rhs);
    
    [[unroll(MinSubgroupSizeLog2-1)]]
    for (uint step=2u; step<=halfSubgroupSize; step <<= 1u)
    {
        rhs = glsl::subgroupShuffleUp<T>(value, step);
        value = op(value, subgroupInvocation<step ? Binop::identity:rhs);
    }
    return value;
}

template<typename T, class Binop>
T exclusive_scan(T value)
{
    value = inclusive_scan<T, Binop>(value);
    // can't risk getting short-circuited, need to store to a var
    T left = glsl::subgroupShuffleUp<T>(value,1);
    // the first invocation doesn't have anything in its left so we set to the binop's identity value for exlusive scan
    return bool(glsl::gl_SubgroupInvocationID()) ? left:Binop::identity;
}

template<typename T, class Binop>
T reduction(NBL_CONST_REF_ARG(T) value)
{
    // take the last subgroup invocation's value for the reduction
    return BroadcastLast<T>(inclusive_scan<T,Binop>(value));
}
}
#endif
}
}
}

#endif