// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_MATH_OCTAHEDRAL_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_OCTAHEDRAL_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"

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

    // F : [0, 1]^2 -> S^2
    static vector3_type uvToDir(const vector2_type uv)
    {
        vector3_type p = vector3_type((uv * scalar_type(2) - scalar_type(1)), scalar_type(0));
		const scalar_type a_x = abs(p.x); const scalar_type a_y = abs(p.y);

        p.z = scalar_type(1) - a_x - a_y;

        if (p.z < scalar_type(0)) 
        {
            p.x = (p.x < scalar_type(0) ? scalar_type(-1) : scalar_type(1)) * (scalar_type(1) - a_y);
            p.y = (p.y < scalar_type(0) ? scalar_type(-1) : scalar_type(1)) * (scalar_type(1) - a_x);
        }

        return hlsl::normalize(p);
    }

    // F^-1 : S^2 -> [-1, 1]^2
    static vector2_type dirToNDC(vector3_type dir)
    {
        dir = hlsl::normalize(dir);
        const scalar_type sum = hlsl::dot(vector3_type(scalar_type(1), scalar_type(1), scalar_type(1)), abs(dir));
        vector3_type s = dir / sum;

        if (s.z < scalar_type(0))
        {
            s.x = (s.x < scalar_type(0) ? scalar_type(-1) : scalar_type(1)) * (scalar_type(1) - abs(s.y));
            s.y = (s.y < scalar_type(0) ? scalar_type(-1) : scalar_type(1)) * (scalar_type(1) - abs(s.x));
        }

        return s.xy;
    }

    // transforms direction vector into UV for corner sampling
    // dir in S^2, halfMinusHalfPixel in [0, 0.5)^2,
    // where halfMinusHalfPixel = 0.5-0.5/texSize
    // and texSize.x >= 1, texSize.y >= 1
    // NOTE/TODO: not best place to keep it here
    static vector2_type toCornerSampledUV(vector3_type dir, vector2_type halfMinusHalfPixel)
    {
        // note: cornerSampled(NDC*0.5+0.5) = NDC*0.5*(1-1/texSize)+0.5
        return dirToNDC(dir) * halfMinusHalfPixel + scalar_type(0.5);
    }
};

}
}
}

#endif // _NBL_BUILTIN_HLSL_MATH_OCTAHEDRAL_INCLUDED_