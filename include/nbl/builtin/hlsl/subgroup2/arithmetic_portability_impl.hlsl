// Copyright (C) 2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP2_ARITHMETIC_PORTABILITY_IMPL_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP2_ARITHMETIC_PORTABILITY_IMPL_INCLUDED_

// #include "nbl/builtin/hlsl/glsl_compat/subgroup_shuffle.hlsl"
// #include "nbl/builtin/hlsl/glsl_compat/subgroup_arithmetic.hlsl"

// #include "nbl/builtin/hlsl/subgroup/ballot.hlsl"

// #include "nbl/builtin/hlsl/functional.hlsl"

#include "nbl/builtin/hlsl/subgroup/arithmetic_portability_impl.hlsl"

namespace nbl
{
namespace hlsl
{
namespace subgroup2
{

namespace impl
{

template<class Params, uint32_t ItemsPerInvocation, bool native>
struct inclusive_scan
{
    using type_t = typename Params::type_t;
    using scalar_t = typename Params::scalar_t;
    using binop_t = typename Params::binop_t;
    using exclusive_scan_op_t = subgroup::impl::exclusive_scan<binop_t, native>;

    // NBL_CONSTEXPR_STATIC_INLINE uint32_t ItemsPerInvocation = vector_traits<T>::Dimension;

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

template<class Params, uint32_t ItemsPerInvocation, bool native>
struct exclusive_scan
{
    using type_t = typename Params::type_t;
    using scalar_t = typename Params::scalar_t;
    using binop_t = typename Params::binop_t;
    using inclusive_scan_op_t = subgroup2::impl::inclusive_scan<Params, ItemsPerInvocation, native>;

    // NBL_CONSTEXPR_STATIC_INLINE uint32_t ItemsPerInvocation = vector_traits<T>::Dimension;

    type_t operator()(type_t value)
    {
        inclusive_scan_op_t op;
        value = op(value);

        type_t left = glsl::subgroupShuffleUp<type_t>(value,1);

        type_t retval;
        retval[0] = bool(glsl::gl_SubgroupInvocationID()) ? left[ItemsPerInvocation-1] : binop_t::identity;
        [unroll]
        for (uint32_t i = 1; i < ItemsPerInvocation; i++)
            retval[i] = value[i-1];
        return retval;
    }
};

template<class Params, uint32_t ItemsPerInvocation, bool native>
struct reduction
{
    using type_t = typename Params::type_t;
    using scalar_t = typename Params::scalar_t;
    using binop_t = typename Params::binop_t;
    using op_t = subgroup::impl::reduction<binop_t, native>;

    // NBL_CONSTEXPR_STATIC_INLINE uint32_t ItemsPerInvocation = vector_traits<T>::Dimension;

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
// #define SPECIALIZE(NAME,BINOP,SUBGROUP_OP) template<typename T> struct NAME<BINOP<T>,true> \
// { \
//     using type_t = T; \
//  \
//     type_t operator()(NBL_CONST_REF_ARG(type_t) v) {return glsl::subgroup##SUBGROUP_OP<type_t>(v);} \
// }

// #define SPECIALIZE_ALL(BINOP,SUBGROUP_OP) SPECIALIZE(reduction,BINOP,SUBGROUP_OP); \
//     SPECIALIZE(inclusive_scan,BINOP,Inclusive##SUBGROUP_OP); \
//     SPECIALIZE(exclusive_scan,BINOP,Exclusive##SUBGROUP_OP);

// SPECIALIZE_ALL(bit_and,And);
// SPECIALIZE_ALL(bit_or,Or);
// SPECIALIZE_ALL(bit_xor,Xor);

// SPECIALIZE_ALL(plus,Add);
// SPECIALIZE_ALL(multiplies,Mul);

// SPECIALIZE_ALL(minimum,Min);
// SPECIALIZE_ALL(maximum,Max);

// #undef SPECIALIZE_ALL
// #undef SPECIALIZE

// specialize portability
template<class Params, bool native>
struct inclusive_scan<Params, 1, native>
{
    using type_t = typename Params::type_t;
    using scalar_t = typename Params::scalar_t;
    using binop_t = typename Params::binop_t;
    using op_t = subgroup::impl::inclusive_scan<binop_t, native>;
    // assert T == scalar type, binop::type == T

    type_t operator()(NBL_CONST_REF_ARG(type_t) value)
    {
        op_t op;
        return op(value);
    }
};

template<class Params, bool native>
struct exclusive_scan<Params, 1, native>
{
    using type_t = typename Params::type_t;
    using scalar_t = typename Params::scalar_t;
    using binop_t = typename Params::binop_t;
    using op_t = subgroup::impl::exclusive_scan<binop_t, native>;

    type_t operator()(NBL_CONST_REF_ARG(type_t) value)
    {
        op_t op;
        return op(value);
    }
};

template<class Params, bool native>
struct reduction<Params, 1, native>
{
    using type_t = typename Params::type_t;
    using scalar_t = typename Params::scalar_t;
    using binop_t = typename Params::binop_t;
    using op_t = subgroup::impl::reduction<binop_t, native>;

    scalar_t operator()(NBL_CONST_REF_ARG(type_t) value)
    {
        op_t op;
        return op(value);
    }
};

}

}
}
}

#endif
