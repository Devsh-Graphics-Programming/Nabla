// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_BILINEAR_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_BILINEAR_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/limits.hlsl>
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
        density_type pdf;
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

    codomain_type generate(const domain_type u, NBL_REF_ARG(cache_type) cache)
    {
        typename Linear<scalar_type>::cache_type linearCache;

        vector2_type p;
        p.y = lineary.generate(u.y, linearCache);

        const vector2_type ySliceEndPoints = vector2_type(bilinearCoeffs[0] + p.y * bilinearCoeffDiffs[0], bilinearCoeffs[1] + p.y * bilinearCoeffDiffs[1]);
        Linear<scalar_type> linearx = Linear<scalar_type>::create(ySliceEndPoints);
        p.x = linearx.generate(u.x, linearCache);

        cache.pdf = backwardPdf(p);
        return p;
    }

    domain_type generateInverse(const codomain_type p, NBL_REF_ARG(cache_type) cache)
    {
        typename Linear<scalar_type>::cache_type linearCache;

        vector2_type u;
        const vector2_type ySliceEndPoints = vector2_type(bilinearCoeffs[0] + p.y * bilinearCoeffDiffs[0], bilinearCoeffs[1] + p.y * bilinearCoeffDiffs[1]);
        Linear<scalar_type> linearx = Linear<scalar_type>::create(ySliceEndPoints);
        u.x = linearx.generateInverse(p.x, linearCache);
        u.y = lineary.generateInverse(p.y, linearCache);

        cache.pdf = backwardPdf(p);
        return u;
    }

    density_type forwardPdf(const cache_type cache)
    {
        return cache.pdf;
    }

    weight_type forwardWeight(const cache_type cache)
    {
        return forwardPdf(cache);
    }

    density_type backwardPdf(const codomain_type p)
    {
        const vector2_type ySliceEndPoints = vector2_type(bilinearCoeffs[0] + p.y * bilinearCoeffDiffs[0], bilinearCoeffs[1] + p.y * bilinearCoeffDiffs[1]);
        return nbl::hlsl::mix(ySliceEndPoints[0], ySliceEndPoints[1], p.x) * fourOverTwiceAreasUnderXCurveSum;
    }

    weight_type backwardWeight(const codomain_type p)
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
