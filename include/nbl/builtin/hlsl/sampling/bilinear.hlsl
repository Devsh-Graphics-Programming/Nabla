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
        retval.bilinearCoeffs = bilinearCoeffs;
        retval.bilinearCoeffDiffs = vector2_type(bilinearCoeffs[2]-bilinearCoeffs[0], bilinearCoeffs[3]-bilinearCoeffs[1]);
        vector2_type twiceAreasUnderXCurve = vector2_type(bilinearCoeffs[0] + bilinearCoeffs[1], bilinearCoeffs[2] + bilinearCoeffs[3]);
        retval.fourOverTwiceAreasUnderXCurveSum = scalar_type(4.0) / (twiceAreasUnderXCurve[0] + twiceAreasUnderXCurve[1]);
        retval.lineary = Linear<scalar_type>::create(twiceAreasUnderXCurve);
        return retval;
    }

    codomain_type generate(const domain_type u, NBL_REF_ARG(cache_type) cache) NBL_CONST_MEMBER_FUNC
    {
        typename Linear<scalar_type>::cache_type linearYCache;

        vector2_type p;
        p.y = lineary.generate(u.y, linearYCache);

        const vector2_type ySliceEndPoints = vector2_type(bilinearCoeffs[0] + p.y * bilinearCoeffDiffs[0], bilinearCoeffs[1] + p.y * bilinearCoeffDiffs[1]);
        Linear<scalar_type> linearx = Linear<scalar_type>::create(ySliceEndPoints);
        p.x = linearx.generate(u.x, cache.linearXCache);

        // pre-multiply by normalization so forwardPdf is just addition
        cache.normalizedStart = ySliceEndPoints[0] * fourOverTwiceAreasUnderXCurveSum;
        cache.linearXCache.diffTimesX *= fourOverTwiceAreasUnderXCurveSum;
        return p;
    }

    density_type forwardPdf(const cache_type cache) NBL_CONST_MEMBER_FUNC
    {
        return cache.normalizedStart + cache.linearXCache.diffTimesX;
    }

    weight_type forwardWeight(const cache_type cache) NBL_CONST_MEMBER_FUNC
    {
        return forwardPdf(cache);
    }

    density_type backwardPdf(const codomain_type p) NBL_CONST_MEMBER_FUNC
    {
        const vector2_type ySliceEndPoints = vector2_type(bilinearCoeffs[0] + p.y * bilinearCoeffDiffs[0], bilinearCoeffs[1] + p.y * bilinearCoeffDiffs[1]);
        return nbl::hlsl::mix(ySliceEndPoints[0], ySliceEndPoints[1], p.x) * fourOverTwiceAreasUnderXCurveSum;
    }

    weight_type backwardWeight(const codomain_type p) NBL_CONST_MEMBER_FUNC
    {
        return backwardPdf(p);
    }

    // unit square: x0y0    x1y0
    //              x0y1    x1y1
    vector4_type bilinearCoeffs;    // (x0y0, x0y1, x1y0, x1y1)
    vector2_type bilinearCoeffDiffs;
    scalar_type fourOverTwiceAreasUnderXCurveSum;
    Linear<scalar_type> lineary;
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
