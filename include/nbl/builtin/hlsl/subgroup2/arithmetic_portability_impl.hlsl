// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP2_ARITHMETIC_PORTABILITY_IMPL_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP2_ARITHMETIC_PORTABILITY_IMPL_INCLUDED_

#include "nbl/builtin/hlsl/subgroup/arithmetic_portability.hlsl"

namespace nbl
{
namespace hlsl
{
namespace subgroup2
{

namespace impl
{

template<template<class> class Binop, typename T, int32_t ItemsPerInvocation, bool native>
struct inclusive_scan
{
    using type_t = typename T;
    using par_type_t = conditional_t<ItemsPerInvocation < 2, type_t, vector<type_t, ItemsPerInvocation> >;
    using binop_t = Binop<type_t>;
    using binop_par_t = Binop<par_type_t>;
    using exclusive_scan_op_t = subgroup::impl::exclusive_scan<binop_t, native>;

    par_type_t operator()(NBL_CONST_REF_ARG(par_type_t) value)
    {
        binop_t binop;
        par_type_t retval;
        retval[0] = value[0];
        [unroll(ItemsPerInvocation-1)]
        for (uint32_t i = 1; i < ItemsPerInvocation; i++)
            retval[i] = binop(retval[i-1], value[i]);
        
        exclusive_scan_op_t op;
        type_t exclusive = op(retval[ItemsPerInvocation-1]);

        [unroll(ItemsPerInvocation)]
        for (uint32_t i = 0; i < ItemsPerInvocation; i++)
            retval[i] = binop(retval[i], exclusive);
        return retval;
    }
};

template<template<class> class Binop, typename T, int32_t ItemsPerInvocation, bool native>
struct exclusive_scan
{
    using type_t = typename T;
    using par_type_t = conditional_t<ItemsPerInvocation < 2, type_t, vector<type_t, ItemsPerInvocation> >;
    using binop_t = Binop<type_t>;
    using binop_par_t = Binop<par_type_t>;
    using inclusive_scan_op_t = subgroup2::impl::inclusive_scan<binop_par_t, native>;

    par_type_t operator()(NBL_CONST_REF_ARG(par_type_t) value)
    {
        inclusive_scan_op_t op;
        value = op(value);

        par_type_t left = glsl::subgroupShuffleUp<par_type_t>(value,1);

        par_type_t retval;
        [unroll(ItemsPerInvocation-1)]
        for (uint32_t i = 1; i < ItemsPerInvocation; i++)
            retval[ItemsPerInvocation-i] = retval[ItemsPerInvocation-i-1];
        retval[0] = bool(glsl::gl_SubgroupInvocationID()) ? left[ItemsPerInvocation-1] : binop_t::identity;
        return retval;
    }
};

template<template<class> class Binop, typename T, int32_t ItemsPerInvocation, bool native>
struct reduction
{
    using type_t = typename T;
    using par_type_t = conditional_t<ItemsPerInvocation < 2, type_t, vector<type_t, ItemsPerInvocation> >;
    using binop_t = Binop<type_t>;
    using binop_par_t = Binop<par_type_t>;
    using op_t = subgroup::impl::reduction<binop_par_t, native>;

    type_t operator()(NBL_CONST_REF_ARG(par_type_t) value)
    {
        binop_t binop;
        op_t op;
        par_type_t result = op(value);
        type_t retval;
        [unroll(ItemsPerInvocation-1)]
        for (uint32_t i = 0; i < ItemsPerInvocation; i++)
            retval += binop(retval, result[i]);
        return retval;
    }
};

}

}
}
}

#endif