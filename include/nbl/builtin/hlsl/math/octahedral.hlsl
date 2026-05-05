// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_MATH_OCTAHEDRAL_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_OCTAHEDRAL_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"
#include "nbl/builtin/hlsl/math/functions.hlsl"

namespace nbl
{
namespace hlsl
{
namespace math
{

// Octahedral Transform, maps 3D direction vectors to 2D square and vice versa
template<typename T = float32_t>
struct OctahedralTransform
{
    using scalar_type   = T;
    using vector2_type  = vector<T, 2>;
    using vector3_type  = vector<T, 3>;

    // F : [-1, 1]^2 -> S^2
    static vector3_type ndcToDir(NBL_CONST_REF_ARG(vector2_type) ndc)
    {
        const vector2_type a = abs(ndc);
        vector3_type p = vector3_type(ndc, scalar_type(1) - a.x - a.y);

        if (p.z < scalar_type(0))
            p.xy = __foldToUpperHemisphere(ndc);

        return hlsl::normalize(p);
    }

    // F : [0, 1]^2 -> S^2 (UV with half-texel handling)
    static vector3_type uvToDir(NBL_CONST_REF_ARG(vector2_type) uv, NBL_CONST_REF_ARG(vector2_type) halfMinusHalfPixel)
    {
        const vector2_type ndc = (uv - vector2_type(scalar_type(0.5), scalar_type(0.5))) / halfMinusHalfPixel;
        return ndcToDir(ndc);
    }

    // F^-1 : S^2 -> [-1, 1]^2
    static vector2_type dirToNDC(NBL_CONST_REF_ARG(vector3_type) d)
    {
        const scalar_type sum = lpNorm<vector3_type, 1>(d);
        vector3_type s = d / sum;

        if (s.z < scalar_type(0))
            s.xy = __foldToUpperHemisphere(s.xy);

        return s.xy;
    }

    // transforms direction vector into UV with half-texel handling
    // dir in S^2, halfMinusHalfPixel in [0, 0.5)^2,
    // where halfMinusHalfPixel = 0.5-0.5/texSize
    static vector2_type dirToUV(NBL_CONST_REF_ARG(vector3_type) dir, NBL_CONST_REF_ARG(vector2_type) halfMinusHalfPixel)
    {
        return dirToNDC(dir) * halfMinusHalfPixel + scalar_type(0.5);
    }

    static vector2_type __foldToUpperHemisphere(NBL_CONST_REF_ARG(vector2_type) v)
    {
        // Use copySign instead of sign() to preserve -0 and avoid DXC corner cases.
        const vector2_type factor = vector2_type(
            ieee754::copySign(scalar_type(1), v.x),
            ieee754::copySign(scalar_type(1), v.y));
        const vector2_type swapped = vector2_type(v.y, v.x);
        return factor * (vector2_type(scalar_type(1), scalar_type(1)) - abs(swapped));
    }
};

}
}
}

#endif // _NBL_BUILTIN_HLSL_MATH_OCTAHEDRAL_INCLUDED_
