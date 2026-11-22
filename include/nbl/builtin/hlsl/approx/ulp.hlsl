// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_APPROX_ULP_INCLUDED_
#define _NBL_BUILTIN_HLSL_APPROX_ULP_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/concepts/core.hlsl>
#include <nbl/builtin/hlsl/ieee754.hlsl>
#include <nbl/builtin/hlsl/tgmath.hlsl>
#include <nbl/builtin/hlsl/utils/elementwise.hlsl>

namespace nbl
{
namespace hlsl
{
namespace approx
{

// Scalar predicate: `lhs` and `rhs` are equal if they are at most `maxULPs` representable values apart, see `ieee754::ulpDistance`.
template<typename FloatingPoint NBL_PRIMARY_REQUIRES(concepts::FloatingPointScalar<FloatingPoint>)
struct ULPEqualPred
{
    using this_t = ULPEqualPred<FloatingPoint>;
    using uint_type = typename unsigned_integer_of_size<sizeof(FloatingPoint)>::type;

    static this_t create(const uint_type _maxULPs)
    {
        this_t retval;
        retval.maxULPs = _maxULPs;
        return retval;
    }

    bool operator()(const FloatingPoint lhs, const FloatingPoint rhs) NBL_CONST_MEMBER_FUNC
    {
        // NaN payloads can be a single ULP apart when looked at as integers, so NaN has to be rejected explicitly
        if (hlsl::isnan<FloatingPoint>(lhs) || hlsl::isnan<FloatingPoint>(rhs))
            return false;
        // the largest finite value is a single ULP away from infinity, so infinities have to match exactly
        if (hlsl::isinf<FloatingPoint>(lhs) || hlsl::isinf<FloatingPoint>(rhs))
            return ieee754::impl::bitCastToUintType(lhs) == ieee754::impl::bitCastToUintType(rhs);
        return ieee754::ulpDistance<FloatingPoint>(lhs, rhs) <= maxULPs;
    }

    uint_type maxULPs;
};

// Works on floating point scalars, vectors and matrices, every element has to pass.
template<typename T>
bool ulpEqual(NBL_CONST_REF_ARG(T) lhs, NBL_CONST_REF_ARG(T) rhs, const typename ULPEqualPred<scalar_type_t<T> >::uint_type maxULPs)
{
    using pred_t = ULPEqualPred<scalar_type_t<T> >;
    return utils::elementwiseAll<T, pred_t>(lhs, rhs, pred_t::create(maxULPs));
}

}
}
}

#endif
