// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_APPROX_ABS_REL_INCLUDED_
#define _NBL_BUILTIN_HLSL_APPROX_ABS_REL_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/concepts/core.hlsl>
#include <nbl/builtin/hlsl/tgmath.hlsl>
#include <nbl/builtin/hlsl/utils/elementwise.hlsl>

namespace nbl
{
namespace hlsl
{
namespace approx
{

// Scalar predicate: `lhs` and `rhs` are equal if they differ by at most `maxAbsoluteDifference`
// OR by at most `maxRelativeDifference` times the larger magnitude.
// The absolute bound is what makes comparisons against 0 work, a purely relative bound degrades to exact equality there.
// IEEE semantics: NaN never compares equal, infinities only equal themselves.
template<typename FloatingPoint NBL_PRIMARY_REQUIRES(concepts::FloatingPointScalar<FloatingPoint>)
struct AbsRelEqualPred
{
    using this_t = AbsRelEqualPred<FloatingPoint>;

    static this_t create(const FloatingPoint _maxAbsoluteDifference, const FloatingPoint _maxRelativeDifference)
    {
        this_t retval;
        retval.maxAbsoluteDifference = _maxAbsoluteDifference;
        retval.maxRelativeDifference = _maxRelativeDifference;
        return retval;
    }

    bool operator()(const FloatingPoint lhs, const FloatingPoint rhs) NBL_CONST_MEMBER_FUNC
    {
        // exact matches, `+0 == -0` and same-signed infinities
        if (lhs == rhs)
            return true;
        // no special case for NaN, `diff` will be NaN and every comparison below false
        const FloatingPoint diff = hlsl::abs<FloatingPoint>(lhs - rhs);
        const FloatingPoint largest = hlsl::max<FloatingPoint>(hlsl::abs<FloatingPoint>(lhs), hlsl::abs<FloatingPoint>(rhs));
        // multiplying instead of dividing by `largest` is cheaper, but `inf <= maxRelativeDifference*inf` would hold,
        // hence the infinity guard which uses a bit pattern test so it survives `/fp:fast` and `-ffast-math`
        return diff <= maxAbsoluteDifference || (diff <= maxRelativeDifference * largest && !hlsl::isinf<FloatingPoint>(largest));
    }

    FloatingPoint maxAbsoluteDifference;
    FloatingPoint maxRelativeDifference;
};

// Works on floating point scalars, vectors and matrices, every element has to pass.
template<typename T>
bool absRelEqual(NBL_CONST_REF_ARG(T) lhs, NBL_CONST_REF_ARG(T) rhs, const scalar_type_t<T> maxAbsoluteDifference, const scalar_type_t<T> maxRelativeDifference)
{
    using pred_t = AbsRelEqualPred<scalar_type_t<T> >;
    return utils::elementwiseAll<T, pred_t>(lhs, rhs, pred_t::create(maxAbsoluteDifference, maxRelativeDifference));
}

}
}
}

#endif
