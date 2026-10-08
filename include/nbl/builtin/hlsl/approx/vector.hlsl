#ifndef _NBL_BUILTIN_HLSL_APPROX_VECTOR_INCLUDED_
#define _NBL_BUILTIN_HLSL_APPROX_VECTOR_INCLUDED_

#include <nbl/builtin/hlsl/approx/abs_rel.hlsl>

// TODO: move `impl::OrientationEqualHelper`/`orientationEqual` from `approx/orientation.hlsl` here and rename to `isParallel`

namespace nbl
{
namespace hlsl
{
namespace approx
{
namespace impl
{

template<typename FloatingPointVector NBL_PRIMARY_REQUIRES(concepts::FloatingPointVectorial<FloatingPointVector>)
struct IsPerpendicularHelper
{
    using scalar_t = typename vector_traits<FloatingPointVector>::scalar_type;

    // Vectors `a` and `b` are perpendicular when the angle `theta` between them is 90deg. From the dot product formula:
    //     dot(a,b) = |a| * |b| * cos(theta)
    //     cos(theta) = dot(a,b) / (|a| * |b|)
    // so perpendicular means cos(theta) == 0.
    //
    // We accept |cos(theta)| <= epsilon. For a small deviation `delta` from 90deg, cos(90deg - delta) = sin(delta) ~= delta (in radians),
    // so this allows a deviation of asin(epsilon) ~= epsilon radians, e.g. epsilon = 1e-5 is ~5.73e-4 degrees (~2 arcseconds).
    //
    // Only the absolute error is used: we compare against 0.0 and an error relative to 0 is meaningless
    // (a relative bound scaled by a magnitude of 0 degenerates into requiring an exact match).
    //
    // To avoid the division, `|a| * |b|` moves to the other side of the inequality:
    //     |dot(a,b)| / (|a| * |b|) <= epsilon
    //     |dot(a,b)| <= epsilon * |a| * |b| = epsilon * sqrt(dot(a,a) * dot(b,b))
    // `maxAbsoluteDifference` is that right hand side (a single `sqrt`, since dot(v,v) = |v|^2), and `absRelEqual` checks `dot(a,b)` against it.
    // The bound grows with the lengths, so uniformly scaled vectors give the same answer.
    //
    // A zero length vector counts as perpendicular to everything (the dot product and the bound are both 0).
    static bool __call(NBL_CONST_REF_ARG(FloatingPointVector) lhs, NBL_CONST_REF_ARG(FloatingPointVector) rhs, const scalar_t cosThetaEpsilon)
    {
        // NOTE: we considered getting rid of the `sqrt` by squaring both sides, `dot(a,b)^2 <= epsilon^2 * dot(a,a) * dot(b,b)`,
        // and decided against it: squaring the already squared lengths makes floating point overflow much more likely for larger values.
        const scalar_t maxAbsoluteDifference = cosThetaEpsilon * hlsl::sqrt(hlsl::dot(lhs, lhs) * hlsl::dot(rhs, rhs));
        return absRelEqual<scalar_t>(hlsl::dot(lhs, rhs), scalar_t(0.0), maxAbsoluteDifference, scalar_t(0.0));
    }
};

}

// compares the squared lengths of `lhs` and `rhs` with `absRelEqual`
template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorial<T>)
bool squaredLengthEqual(NBL_CONST_REF_ARG(T) lhs, NBL_CONST_REF_ARG(T) rhs, const typename vector_traits<T>::scalar_type maxAbsoluteDifference, const typename vector_traits<T>::scalar_type maxRelativeDifference)
{
    using scalar_t = typename vector_traits<T>::scalar_type;
    return absRelEqual<scalar_t>(hlsl::dot(lhs,lhs), hlsl::dot(rhs,rhs), maxAbsoluteDifference, maxRelativeDifference);
}

// true if |cos(theta)| <= `cosThetaEpsilon` for the angle `theta` between `lhs` and `rhs`, regardless of their lengths
// (about `cosThetaEpsilon` radians of deviation from 90deg), see `impl::IsPerpendicularHelper` for the math
template<typename T>
bool isPerpendicular(NBL_CONST_REF_ARG(T) lhs, NBL_CONST_REF_ARG(T) rhs, const typename vector_traits<T>::scalar_type cosThetaEpsilon)
{
    return impl::IsPerpendicularHelper<T>::__call(lhs, rhs, cosThetaEpsilon);
}

}
}
}

#endif
