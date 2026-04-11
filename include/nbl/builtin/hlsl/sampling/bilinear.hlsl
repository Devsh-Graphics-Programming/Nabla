// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_BILINEAR_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_BILINEAR_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/builtin/hlsl/tgmath.hlsl>
#include <nbl/builtin/hlsl/sampling/linear.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{

template<typename T>
struct Bilinear
{
    using scalar_type = T;
    using vector2_type = vector<T, 2>;
    using vector3_type = vector<T, 3>;
    using vector4_type = vector<T, 4>;

    // BijectiveSampler concept types
    using domain_type = vector2_type;
    using codomain_type = vector2_type;
    using density_type = scalar_type;
    using weight_type = density_type;

    struct cache_type
    {
        scalar_type normalizedStart;
        typename Linear<T>::cache_type linearXCache;
    };

    static Bilinear<T> create(const vector4_type bilinearCoeffs)
    {
        Bilinear<T> retval;
        retval.yStarts = bilinearCoeffs.xy;
        retval.yDiffs = vector2_type(bilinearCoeffs[2]-bilinearCoeffs[0], bilinearCoeffs[3]- bilinearCoeffs[1]);
        vector2_type twiceAreasUnderXCurve = vector2_type(bilinearCoeffs[0] + bilinearCoeffs[1], bilinearCoeffs[2] + bilinearCoeffs[3]);
        // Linear::create adds FLT_MIN internally, replicate here so both divisions share
        // the same denominator (sum + 2*min), enabling CSE to merge them into one division
        const scalar_type safeSum = twiceAreasUnderXCurve[0] + twiceAreasUnderXCurve[1] + scalar_type(2.0) * hlsl::numeric_limits<scalar_type>::min;
        const scalar_type yNormFactor = scalar_type(2.0) / safeSum;
        retval.lineary = Linear<scalar_type>::create(twiceAreasUnderXCurve);
        retval.normFactor = yNormFactor * scalar_type(2.0);
        return retval;
    }

    codomain_type generate(const domain_type u, NBL_REF_ARG(cache_type) cache) NBL_CONST_MEMBER_FUNC
    {
        typename Linear<scalar_type>::cache_type linearYCache;

        vector2_type p;
        p.y = lineary.generate(u.y, linearYCache);

        const vector2_type ySliceEndPoints = vector2_type(yStarts[0] + p.y * yDiffs[0], yStarts[1] + p.y * yDiffs[1]);
        Linear<scalar_type> linearx = Linear<scalar_type>::create(ySliceEndPoints);
        p.x = linearx.generate(u.x, cache.linearXCache);

        // bilinear PDF = marginal_y_pdf * conditional_x_pdf; reuse both linear caches
        const scalar_type yPdf = lineary.forwardPdf(u.y, linearYCache);
        cache.normalizedStart = yPdf * linearx.linearCoeffStart;
        cache.linearXCache.diffTimesX *= yPdf;
        return p;
    }

    density_type forwardPdf(const domain_type u, const cache_type cache) NBL_CONST_MEMBER_FUNC
    {
        return cache.normalizedStart + cache.linearXCache.diffTimesX;
    }

    weight_type forwardWeight(const domain_type u, const cache_type cache) NBL_CONST_MEMBER_FUNC
    {
        return forwardPdf(u, cache);
    }

    density_type backwardPdf(const codomain_type p) NBL_CONST_MEMBER_FUNC
    {
        const vector2_type ySliceEndPoints = vector2_type(yStarts[0] + p.y * yDiffs[0], yStarts[1] + p.y * yDiffs[1]);
        return nbl::hlsl::mix(ySliceEndPoints[0], ySliceEndPoints[1], p.x) * normFactor;
    }

    weight_type backwardWeight(const codomain_type p) NBL_CONST_MEMBER_FUNC
    {
        return backwardPdf(p);
    }

    // TODO: if backwardPdf is queried 3+ times, it pays to normalize yStarts/yDiffs by normFactor
    // upfront in create (4 MULs), saving 1 MUL in lineary construction and 1 MUL per backwardPdf call.
    // lineary could then be constructed directly (bypassing create) with the pre-normalized values.
    vector2_type yStarts;
    vector2_type yDiffs;
    scalar_type normFactor;
    Linear<scalar_type> lineary;
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
