// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_LINEAR_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_LINEAR_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/builtin/hlsl/tgmath.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{

template<typename T>
struct Linear
{
    using scalar_type = T;
    using vector2_type = vector<T, 2>;

    // BackwardTractableSampler concept types
    using domain_type = scalar_type;
    using codomain_type = scalar_type;
    using density_type = scalar_type;
    using weight_type = density_type;

    struct cache_type
    {
        scalar_type diffTimesX;
    };

    static Linear<T> create(const vector2_type linearCoeffs)   // start and end importance values (start, end), assumed to be at x=0 and x=1
    {
        Linear<T> retval;
        // add min to both coefficients so (0,0) input produces a valid uniform sampler
        // instead of inf normalization (2/0) leading to NaN; negligible for normal inputs
        const vector2_type safeCoeffs = linearCoeffs + vector2_type(hlsl::numeric_limits<scalar_type>::min, hlsl::numeric_limits<scalar_type>::min);
        // normalize coefficients so that the PDF is simply linearCoeffStart + linearCoeffDiff * x
        const scalar_type normFactor = scalar_type(2.0) / (safeCoeffs[0] + safeCoeffs[1]);
        const vector2_type normalized = safeCoeffs * normFactor;
        retval.linearCoeffStart = normalized[0];
        retval.linearCoeffEnd = normalized[1];
        // precompute for the stable quadratic in generate()
        retval.squaredCoeffStart = normalized[0] * normalized[0];
        retval.twoTimesDiff = scalar_type(2.0) * (normalized[1] - normalized[0]);
        return retval;
    }

    codomain_type generate(const domain_type u, NBL_REF_ARG(cache_type) cache) NBL_CONST_MEMBER_FUNC
    {
        // Inverse CDF via stable quadratic solver.
        // CDF(x) = start*x + 0.5*diff*x^2 = u, with normalization start + 0.5*diff = 1.
        // Quadratic (1-start)*x^2 + start*x - u = 0; since start >= 0 the stable root is
        // x = 2u / (start + sqrt(start^2 + 2*diff*u)), which never cancels.
        const scalar_type sqrtTerm = sqrt(squaredCoeffStart + twoTimesDiff * u);
        const scalar_type denom = linearCoeffStart + sqrtTerm;
        // NOTE: floating point can make x slightly > 1 when u~1 and diff < 0; callers needing
        // non-negative PDF at the boundary should clamp with min(x, 1).
        const codomain_type x = (u + u) / denom;
        // diff*x == sqrtTerm - start algebraically (conjugate identity), saves 1 mul
        cache.diffTimesX = sqrtTerm - linearCoeffStart;
        return x;
    }

    density_type forwardPdf(const domain_type u, const cache_type cache) NBL_CONST_MEMBER_FUNC
    {
        return linearCoeffStart + cache.diffTimesX;
    }

    weight_type forwardWeight(const domain_type u, const cache_type cache) NBL_CONST_MEMBER_FUNC
    {
        return forwardPdf(u, cache);
    }

    // Alternative forms (since start + 0.5*diff == 1 after normalization):
    //   start + (0.5 - start) * x
    //   1 + diff * (x - 0.5)
    // Not used because we already store start for generate().
    density_type backwardPdf(const codomain_type x) NBL_CONST_MEMBER_FUNC
    {
        assert(x >= scalar_type(0.0) && x <= scalar_type(1.0));
        return hlsl::mix(linearCoeffStart, linearCoeffEnd, x);
    }

    weight_type backwardWeight(const codomain_type x) NBL_CONST_MEMBER_FUNC
    {
        return backwardPdf(x);
    }

    scalar_type linearCoeffStart;
    scalar_type linearCoeffEnd;
    scalar_type squaredCoeffStart;
    scalar_type twoTimesDiff;
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
