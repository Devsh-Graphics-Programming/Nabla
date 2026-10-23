// Copyright (C) 2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP2_ARITHMETIC_PORTABILITY_IMPL_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP2_ARITHMETIC_PORTABILITY_IMPL_INCLUDED_

// TODO: `glsl::subgroupShuffleUp` and `glsl::subgroupShuffleXor` are only used for their default `spv::ScopeSubgroup`, drop this dependency on the `glsl` namespace once the subgroup utilities move to a better namespace
#include "nbl/builtin/hlsl/glsl_compat/subgroup_shuffle.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_arithmetic.hlsl"

#include "nbl/builtin/hlsl/subgroup/ballot.hlsl"
#include "nbl/builtin/hlsl/subgroup2/ballot.hlsl"

#include "nbl/builtin/hlsl/algorithm.hlsl"
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


// Primary templates, used when ItemsPerInvocation != 1: each invocation holds ItemsPerInvocation consecutive elements in a vector.
// They combine the invocation's own items serially, so a single `ItemsPerInvocation = 1` subgroup op (native or emulated,
// specialized below) is all that crosses lanes.
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
        // (1) inclusive scan of this invocation's own items
        retval[0] = value[0];
        [unroll]
        for (uint32_t i = 1; i < ItemsPerInvocation; i++)
            retval[i] = binop(retval[i-1], value[i]);

        // (2) one subgroup exclusive scan of each invocation's total: everything held by the lanes before this one
        exclusive_scan_op_t op;
        scalar_t exclusive = op(retval[ItemsPerInvocation-1]);

        // (3) combine that with every item
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
        // (1) inclusive scan of this invocation's own items
        retval[0] = value[0];
        [unroll]
        for (uint32_t i = 1; i < ItemsPerInvocation; i++)
            retval[i] = binop(retval[i-1], value[i]);

        // (2) one subgroup exclusive scan of each invocation's total: everything held by the lanes before this one
        exclusive_scan_op_t op;
        scalar_t exclusive = op(retval[ItemsPerInvocation-1]);

        // (3) shift the items up by one to make them exclusive, combining each with that; item 0 gets it as is
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
        // (1) reduce this invocation's own items
        scalar_t retval = value[0];
        [unroll]
        for (uint32_t i = 1; i < ItemsPerInvocation; i++)
            retval = binop(retval, value[i]);
        // (2) one subgroup reduction of each invocation's total
        return op(retval);
    }
};


// specs for N=1 uses subgroup funcs
// TODO: drop the weird `glsl::` dependency, call `spirv::group*` directly with `spv::ScopeSubgroup` and the `GroupOperation`, or a utility outside `glsl`.
// `plus`, `multiplies` and the bitwise ops map 1:1; `minimum`/`maximum` need signed/unsigned/float dispatch (`spirv::groupSMin/UMin/FMin`),
// either as 3 constrained specializations each (primaries need `NBL_STRUCT_CONSTRAINABLE`) or a shared dispatch helper also used by `glsl::subgroupMin/Max`.
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

template<class BinOp>
struct inclusive_scan_impl
{
    using scalar_t = typename BinOp::type_t;

    static inclusive_scan_impl<BinOp> create(scalar_t _value)
    {
        inclusive_scan_impl<BinOp> retval;
        retval.value = _value;
        retval.subgroupInvocation = spirv::SubgroupLocalInvocationId;
        return retval;
    }

    template<uint16_t StepLog2>
    void __call()
    {
        BinOp op;
        const uint32_t step = 1u << StepLog2;
        spirv::controlBarrier(spv::ScopeSubgroup, spv::ScopeSubgroup, spv::MemorySemanticsMaskNone);
        scalar_t rhs = glsl::subgroupShuffleUp<scalar_t>(value, step);
        value = op(value, hlsl::mix(rhs, BinOp::identity, subgroupInvocation < step));
    }

    scalar_t value;
    uint32_t subgroupInvocation;
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
        inclusive_scan_impl<binop_t> f_impl = inclusive_scan_impl<binop_t>::create(value);
        unrolled_for_range<0, config_t::SizeLog2>::template __call<inclusive_scan_impl<binop_t> >(f_impl);
        return f_impl.value;
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

        scalar_t left = hlsl::mix(binop_t::identity, glsl::subgroupShuffleUp<scalar_t>(value,1), bool(spirv::SubgroupLocalInvocationId));
        return inclusive_scan<Params, BinOp, 1, false>::__call(left);
    }
};

template<class BinOp>
struct reduction_impl
{
    using scalar_t = typename BinOp::type_t;

    static reduction_impl<BinOp> create(scalar_t _value)
    {
        reduction_impl<BinOp> retval;
        retval.value = _value;
        return retval;
    }

    template<uint16_t StepLog2>
    void __call()
    {
        BinOp op;
        spirv::controlBarrier(spv::ScopeSubgroup, spv::ScopeSubgroup, spv::MemorySemanticsMaskNone);
        value = op(glsl::subgroupShuffleXor<scalar_t>(value, 0x1u<<StepLog2),value);
    }

    scalar_t value;
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
        reduction_impl<binop_t> f_impl = reduction_impl<binop_t>::create(value);
        unrolled_for_range<0, config_t::SizeLog2>::template __call<reduction_impl<binop_t> >(f_impl);
        return f_impl.value;
    }
};

}

}
}
}

#endif
