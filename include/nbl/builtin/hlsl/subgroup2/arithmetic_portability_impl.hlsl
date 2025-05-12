// Copyright (C) 2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP2_ARITHMETIC_PORTABILITY_IMPL_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP2_ARITHMETIC_PORTABILITY_IMPL_INCLUDED_

#include "nbl/builtin/hlsl/glsl_compat/subgroup_shuffle.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_arithmetic.hlsl"

#include "nbl/builtin/hlsl/subgroup/ballot.hlsl"
#include "nbl/builtin/hlsl/subgroup2/ballot.hlsl"

#include "nbl/builtin/hlsl/functional.hlsl"
#include "nbl/builtin/hlsl/cpp_compat/intrinsics.hlsl"

namespace nbl
{
namespace hlsl
{
namespace subgroup2
{

namespace impl
{

// forward declarations
template<class Params, class BinOp, uint32_t ItemsPerInvocation, bool native>
struct inclusive_scan;

template<class Params, class BinOp, uint32_t ItemsPerInvocation, bool native>
struct exclusive_scan;

template<class Params, class BinOp, uint32_t ItemsPerInvocation, bool native>
struct reduction;


// BinOp needed to specialize native
template<class Params, class BinOp, uint32_t ItemsPerInvocation, bool native>
struct inclusive_scan
{
    using type_t = typename Params::type_t;
    using scalar_t = typename Params::scalar_t;
    using binop_t = typename Params::binop_t;
    // assert binop_t == BinOp
    using exclusive_scan_op_t = exclusive_scan<Params, binop_t, 1, native>;

    type_t operator()(NBL_CONST_REF_ARG(type_t) value)
    {
        binop_t binop;
        type_t retval;
        retval[0] = value[0];
        [unroll]
        for (uint32_t i = 1; i < ItemsPerInvocation; i++)
            retval[i] = binop(retval[i-1], value[i]);

        exclusive_scan_op_t op;
        scalar_t exclusive = op(retval[ItemsPerInvocation-1]);

        [unroll]
        for (uint32_t i = 0; i < ItemsPerInvocation; i++)
            retval[i] = binop(retval[i], exclusive);
        return retval;
    }
};

template<class Params, class BinOp, uint32_t ItemsPerInvocation, bool native>
struct exclusive_scan
{
    using type_t = typename Params::type_t;
    using scalar_t = typename Params::scalar_t;
    using binop_t = typename Params::binop_t;
    using exclusive_scan_op_t = exclusive_scan<Params, binop_t, 1, native>;

    type_t operator()(type_t value)
    {
        binop_t binop;
        type_t retval;
        retval[0] = value[0];
        [unroll]
        for (uint32_t i = 1; i < ItemsPerInvocation; i++)
            retval[i] = binop(retval[i-1], value[i]);

        exclusive_scan_op_t op;
        scalar_t exclusive = op(retval[ItemsPerInvocation-1]);

        [unroll]
        for (uint32_t i = 1; i < ItemsPerInvocation; i++)
            retval[ItemsPerInvocation-i] = binop(exclusive,retval[ItemsPerInvocation-i-1]);
        retval[0] = exclusive;
        return retval;
    }
};

template<class Params, class BinOp, uint32_t ItemsPerInvocation, bool native>
struct reduction
{
    using type_t = typename Params::type_t;
    using scalar_t = typename Params::scalar_t;
    using binop_t = typename Params::binop_t;
    using op_t = reduction<Params, binop_t, 1, native>;

    scalar_t operator()(NBL_CONST_REF_ARG(type_t) value)
    {
        binop_t binop;
        op_t op;
        scalar_t retval = value[0];
        [unroll]
        for (uint32_t i = 1; i < ItemsPerInvocation; i++)
            retval = binop(retval, value[i]);
        return op(retval);
    }
};


// specs for N=1 uses subgroup funcs
// specialize native
#define SPECIALIZE(NAME,BINOP,SUBGROUP_OP) template<class Params, typename T> struct NAME<Params,BINOP<T>,1,true> \
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

// specialize portability
template<class Params, class BinOp>
struct inclusive_scan<Params, BinOp, 1, false>
{
    using type_t = typename Params::type_t;
    using scalar_t = typename Params::scalar_t;
    using binop_t = typename Params::binop_t;
    // assert T == scalar type, binop::type == T
    using config_t = typename Params::config_t;

    // affected by https://github.com/microsoft/DirectXShaderCompiler/issues/7006
    // NBL_CONSTEXPR_STATIC_INLINE uint32_t SubgroupSizeLog2 = config_t::SizeLog2;

    scalar_t operator()(scalar_t value)
    {
        return __call(value);
    }

    static scalar_t __call(scalar_t value)
    {
        binop_t op;
        const uint32_t subgroupInvocation = glsl::gl_SubgroupInvocationID();

        scalar_t rhs = glsl::subgroupShuffleUp<scalar_t>(value, 1u); // all invocations must execute the shuffle, even if we don't apply the op() to all of them
        value = op(value, hlsl::mix(rhs, binop_t::identity, subgroupInvocation < 1u));

        const uint32_t SubgroupSizeLog2 = config_t::SizeLog2;
        [unroll]
        for (uint32_t i = 1; i < integral_constant<uint32_t,SubgroupSizeLog2>::value; i++)
        {
            const uint32_t step = 1u << i;
            rhs = glsl::subgroupShuffleUp<scalar_t>(value, step);
            value = op(value, hlsl::mix(rhs, binop_t::identity, subgroupInvocation < step));
        }
        return value;
    }
};

template<class Params, class BinOp>
struct exclusive_scan<Params, BinOp, 1, false>
{
    using type_t = typename Params::type_t;
    using scalar_t = typename Params::scalar_t;
    using binop_t = typename Params::binop_t;

    scalar_t operator()(scalar_t value)
    {
        value = inclusive_scan<Params, BinOp, 1, false>::__call(value);
        // can't risk getting short-circuited, need to store to a var
        scalar_t left = glsl::subgroupShuffleUp<scalar_t>(value,1);
        // the first invocation doesn't have anything in its left so we set to the binop's identity value for exlusive scan
        return hlsl::mix(binop_t::identity, left, bool(glsl::gl_SubgroupInvocationID()));
    }
};

template<class Params, class BinOp>
struct reduction<Params, BinOp, 1, false>
{
    using type_t = typename Params::type_t;
    using scalar_t = typename Params::scalar_t;
    using binop_t = typename Params::binop_t;
    using config_t = typename Params::config_t;

    // affected by https://github.com/microsoft/DirectXShaderCompiler/issues/7006
    // NBL_CONSTEXPR_STATIC_INLINE uint32_t SubgroupSizeLog2 = config_t::SizeLog2;

    scalar_t operator()(scalar_t value)
    {
        binop_t op;

        const uint32_t SubgroupSizeLog2 = config_t::SizeLog2;
        [unroll]
        for (uint32_t i = 0; i < integral_constant<uint32_t,SubgroupSizeLog2>::value; i++)
            value = op(glsl::subgroupShuffleXor<scalar_t>(value,0x1u<<i),value);

        return value;
    }
};

}

}
}
}

#endif
