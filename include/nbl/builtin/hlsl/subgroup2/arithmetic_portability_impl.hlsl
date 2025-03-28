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

template<class Binop, bool native>
struct inclusive_scan
{
    using type_t = typename Binop::type_t;
    using scalar_t = typename Binop::scalar_t;
    using binop_t = Binop<scalar_t>;
    using binop_par_t = Binop<type_t>;
    using exclusive_scan_op_t = subgroup::impl::exclusive_scan<binop_t, native>;

    type_t operator()(NBL_CONST_REF_ARG(type_t) value)
    {
        binop_t binop;
        type_t retval;
        retval[0] = value[0];
        [unroll(ItemsPerInvocation-1)]
        for (uint32_t i = 1; i < ItemsPerInvocation; i++)
            retval[i] = binop(retval[i-1], value[i]);
        
        exclusive_scan_op_t op;
        scalar_t exclusive = op(retval[ItemsPerInvocation-1]);

        [unroll(ItemsPerInvocation)]
        for (uint32_t i = 0; i < ItemsPerInvocation; i++)
            retval[i] = binop(retval[i], exclusive);
        return retval;
    }
};

template<class Binop, bool native>
struct exclusive_scan
{
    using type_t = typename Binop::type_t;
    using scalar_t = typename Binop::scalar_t;
    using binop_t = Binop<scalar_t>;
    using binop_par_t = Binop<type_t>;
    using inclusive_scan_op_t = subgroup2::impl::inclusive_scan<binop_par_t, native>;

    type_t operator()(NBL_CONST_REF_ARG(type_t) value)
    {
        inclusive_scan_op_t op;
        value = op(value);

        type_t left = glsl::subgroupShuffleUp<type_t>(value,1);

        type_t retval;
        retval[0] = bool(glsl::gl_SubgroupInvocationID()) ? left[ItemsPerInvocation-1] : binop_t::identity;
        [unroll(ItemsPerInvocation-1)]
        for (uint32_t i = 1; i < ItemsPerInvocation; i++)
            retval[i] = value[i-1];
        return retval;
    }
};

template<class Binop, bool native>
struct reduction
{
    using type_t = typename Binop::type_t;
    using scalar_t = typename Binop::scalar_t;
    using binop_t = Binop<scalar_t>;
    using binop_par_t = Binop<type_t>;
    using op_t = subgroup::impl::reduction<binop_par_t, native>;

    scalar_t operator()(NBL_CONST_REF_ARG(type_t) value)
    {
        binop_t binop;
        op_t op;
        type_t result = op(value);
        scalar_t retval;
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