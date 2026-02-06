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

    static Bilinear<T> create(const vector4_type bilinearCoeffs)
    {
        Bilinear<T> retval;
        retval.bilinearCoeffs = bilinearCoeffs;
        retval.bilinearCoeffDiffs = vector2_type(bilinearCoeffs[2]-bilinearCoeffs[0], bilinearCoeffs[3]-bilinearCoeffs[1]);
        vector2_type twiceAreasUnderXCurve = vector2_type(bilinearCoeffs[0] + bilinearCoeffs[1], bilinearCoeffs[2] + bilinearCoeffs[3]);
        retval.twiceAreasUnderXCurveSumOverFour = scalar_type(4.0) / (twiceAreasUnderXCurve[0] + twiceAreasUnderXCurve[1]);
        retval.lineary = Linear<scalar_type>::create(twiceAreasUnderXCurve);
        return retval;
    }

    vector2_type generate(const vector2_type _u)
    {
        vector2_type u;
        u.y = lineary.generate(_u.y);

        const vector2_type ySliceEndPoints = vector2_type(bilinearCoeffs[0] + u.y * bilinearCoeffDiffs[0], bilinearCoeffs[1] + u.y * bilinearCoeffDiffs[1]);
        Linear<scalar_type> linearx = Linear<scalar_type>::create(ySliceEndPoints);
        u.x = linearx.generate(_u.x);

        return u;
    }

    scalar_type backwardPdf(const vector2_type u)
    {
        const vector2_type ySliceEndPoints = vector2_type(bilinearCoeffs[0] + u.y * bilinearCoeffDiffs[0], bilinearCoeffs[1] + u.y * bilinearCoeffDiffs[1]);
        return nbl::hlsl::mix(ySliceEndPoints[0], ySliceEndPoints[1], u.x) * fourOverTwiceAreasUnderXCurveSum;
    }

    // unit square: x0y0    x1y0
    //              x0y1    x1y1
    vector4_type bilinearCoeffs;    // (x0y0, x0y1, x1y0, x1y1)
    vector2_type bilinearCoeffDiffs;
    vector2_type fourOverTwiceAreasUnderXCurveSum;
    Linear<scalar_type> lineary;
};

}
}
}

#endif
