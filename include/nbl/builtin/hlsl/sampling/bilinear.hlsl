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

    static Bilinear<T> create(NBL_CONST_REF_ARG(vector4_type) bilinearCoeffs)
    {
        Bilinear<T> retval;
        retval.bilinearCoeffs = bilinearCoeffs;
        return retval;
    }

    vector2_type generate(NBL_REF_ARG(scalar_type) rcpPdf, NBL_CONST_REF_ARG(vector2_type) u)
    {
        const vector2_type twiceAreasUnderXCurve = vector2_type(bilinearCoeffs[0] + bilinearCoeffs[1], bilinearCoeffs[2] + bilinearCoeffs[3]);
        Linear<scalar_type> lineary = Linear<scalar_type>::create(twiceAreasUnderXCurve);
        u.y = lineary.generate(u.y);

        const vector2_type ySliceEndPoints = vector2_type(nbl::hlsl::mix(bilinearCoeffs[0], bilinearCoeffs[2], u.y), nbl::hlsl::mix(bilinearCoeffs[1], bilinearCoeffs[3], u.y));
        Linear<scalar_type> linearx = Linear<scalar_type>::create(ySliceEndPoints);
        u.x = linearx.generate(u.x);

        rcpPdf = (twiceAreasUnderXCurve[0] + twiceAreasUnderXCurve[1]) / (4.0 * nbl::hlsl::mix(ySliceEndPoints[0], ySliceEndPoints[1], u.x));

        return u;
    }

    scalar_type pdf(NBL_CONST_REF_ARG(vector2_type) u)
    {
        return 4.0 * nbl::hlsl::mix(nbl::hlsl::mix(bilinearCoeffs[0], bilinearCoeffs[1], u.x), nbl::hlsl::mix(bilinearCoeffs[2], bilinearCoeffs[3], u.x), u.y) / (bilinearCoeffs[0] + bilinearCoeffs[1] + bilinearCoeffs[2] + bilinearCoeffs[3]);
    }

    vector4_type bilinearCoeffs;
};

}
}
}

#endif
