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
        for (uint32_t i = ItemsPerInvocation-1; i > 0; i--)
            retval[i] = binop(exclusive,retval[i-1]);
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

template<class BinOp, uint16_t begin, uint16_t end>
struct inclusive_scan_impl
{
    using scalar_t = typename BinOp::type_t;

    static scalar_t __call(scalar_t value)
    {
        BinOp op;
        const uint32_t subgroupInvocation = glsl::gl_SubgroupInvocationID();
        const uint32_t step = 1u << begin;
        scalar_t rhs = glsl::subgroupShuffleUp<scalar_t>(value, step);
        scalar_t new_value = op(value, hlsl::mix(rhs, BinOp::identity, subgroupInvocation < step));
        return inclusive_scan_impl<BinOp,begin+1,end>::__call(new_value);
    }
};

template<class BinOp, uint16_t end>
struct inclusive_scan_impl<BinOp,end,end>
{
    using scalar_t = typename BinOp::type_t;

    static scalar_t __call(scalar_t value)
    {
        BinOp op;
        const uint32_t subgroupInvocation = glsl::gl_SubgroupInvocationID();
        const uint32_t step = 1u << end;
        scalar_t rhs = glsl::subgroupShuffleUp<scalar_t>(value, step);
        return op(value, hlsl::mix(rhs, BinOp::identity, subgroupInvocation < step));
    }
};

// specialize portability
template<class Params, class BinOp>
struct inclusive_scan<Params, BinOp, 1, false>
{
    using type_t = typename Params::type_t;
    using scalar_t = typename Params::scalar_t;
    using binop_t = typename Params::binop_t;
    using config_t = typename Params::config_t;

    scalar_t operator()(scalar_t value)
    {
        return __call(value);
    }

    static scalar_t __call(scalar_t value)
    {
        return inclusive_scan_impl<binop_t, 0, config_t::SizeLog2-1>::__call(value);
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
        // sync up each subgroup invocation so it runs in lockstep
        spirv::controlBarrier(spv::ScopeSubgroup, spv::ScopeSubgroup, spv::MemorySemanticsMaskNone);

        scalar_t left = hlsl::mix(binop_t::identity, glsl::subgroupShuffleUp<scalar_t>(value,1), bool(glsl::gl_SubgroupInvocationID()));
        spirv::controlBarrier(spv::ScopeSubgroup, spv::ScopeSubgroup, spv::MemorySemanticsMaskNone);
        return inclusive_scan<Params, BinOp, 1, false>::__call(left);
    }
};

template<class BinOp, uint16_t begin, uint16_t end>
struct reduction_impl
{
    using scalar_t = typename BinOp::type_t;

    static scalar_t __call(scalar_t value)
    {
        BinOp op;
        scalar_t new_value = op(glsl::subgroupShuffleXor<scalar_t>(value, 0x1u<<begin),value);
        return reduction_impl<BinOp,begin+1,end>::__call(new_value);
    }
};

template<class BinOp, uint16_t end>
struct reduction_impl<BinOp,end,end>
{
    using scalar_t = typename BinOp::type_t;

    static scalar_t __call(scalar_t value)
    {
        BinOp op;
        return op(glsl::subgroupShuffleXor<scalar_t>(value, 0x1u<<end),value);
    }
};

template<class Params, class BinOp>
struct reduction<Params, BinOp, 1, false>
{
    using type_t = typename Params::type_t;
    using scalar_t = typename Params::scalar_t;
    using binop_t = typename Params::binop_t;
    using config_t = typename Params::config_t;

    scalar_t operator()(scalar_t value)
    {
        return reduction_impl<binop_t, 0, config_t::SizeLog2-1>::__call(value);
    }
};

}

}
}
}

#endif
