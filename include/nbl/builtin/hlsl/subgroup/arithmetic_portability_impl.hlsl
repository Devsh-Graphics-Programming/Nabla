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

namespace native
{

template<class Binop, typename T=typename Binop::type_t>
struct reduction;
template<class Binop, typename T=typename Binop::type_t>
struct inclusive_scan;
template<class Binop, typename T=typename Binop::type_t>
struct exclusive_scan;

#define SPECIALIZE(NAME,BINOP,SUBGROUP_OP) template<typename T> struct NAME<BINOP<T>,T> \
{ \
    using type_t = T; \
 \
    type_t operator()(NBL_CONST_REF_ARG(type_t) v) {return glsl::subgroup##SUBGROUP_OP<type_t>(v);} \
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

namespace portability
{
    
// WARNING
// THIS PORTABILITY IMPLEMENTATION USES SHUFFLE OPS
// Shuffles where you attempt to read an invactive lane, return garbage, 
// which means that our portability reductions and prefix sums will also return garbage/UB/UV
// Always use the native subgroup_arithmetic extensions if supported
    
template<class Binop>
struct inclusive_scan
{
    using type_t = typename Binop::type_t;

    type_t operator()(type_t value)
    {
        return __call(value);
    }

    static type_t __call(type_t value)
    {
        Binop op;
        const uint subgroupInvocation = glsl::gl_SubgroupInvocationID();
        const uint halfSubgroupSize = glsl::gl_SubgroupSize() >> 1u;
    
        uint rhs = glsl::subgroupShuffleUp<type_t>(value, 1u); // all invocations must execute the shuffle, even if we don't apply the op() to all of them
        value = op(value, subgroupInvocation<1u ? Binop::identity:rhs);
    
        [[unroll(MinSubgroupSizeLog2-1)]]
        for (uint step=2u; step<=halfSubgroupSize; step <<= 1u)
        {
            rhs = glsl::subgroupShuffleUp<type_t>(value, step);
            value = op(value, subgroupInvocation<step ? Binop::identity:rhs);
        }
        return value;
    }
};

template<class Binop>
struct exclusive_scan
{
    using type_t = typename Binop::type_t;

    type_t operator()(type_t value)
    {
        value = inclusive_scan<Binop>::__call(value);
        // can't risk getting short-circuited, need to store to a var
        type_t left = glsl::subgroupShuffleUp<type_t>(value,1);
        // the first invocation doesn't have anything in its left so we set to the binop's identity value for exlusive scan
        return bool(glsl::gl_SubgroupInvocationID()) ? left:Binop::identity;
    }
};

template<class Binop>
struct reduction
{
    using type_t = typename Binop::type_t;

    type_t operator()(NBL_CONST_REF_ARG(type_t) value)
    {
        // take the last subgroup invocation's value for the reduction
        return BroadcastLast<type_t>(inclusive_scan<Binop>::__call(value));
    }
};
}

}
}
}

#endif