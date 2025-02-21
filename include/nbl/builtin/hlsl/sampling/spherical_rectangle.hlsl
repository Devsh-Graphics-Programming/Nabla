// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_SPHERICAL_RECTANGLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_SPHERICAL_RECTANGLE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>
#include <nbl/builtin/hlsl/shapes/triangle.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{

template<typename T>
struct SphericalRectangle
{
    using scalar_type = T;
    using vector2_type = vector<T, 2>;
    using vector3_type = vector<T, 3>;
    using vector4_type = vector<T, 4>;

    static SphericalRectangle<T> create(NBL_CONST_REF_ARG(shapes::SphericalRectangle<T>) rect)
    {
        SphericalRectangle<T> retval;
        retval.rect = rect;
        return retval;
    }

    vector2_type generate(NBL_CONST_REF_ARG(vector2_type) rectangleExtents, NBL_CONST_REF_ARG(vector2_type) uv, NBL_REF_ARG(scalar_type) S)
    {
        const vector4_type denorm_n_z = vector4_type(-rect.r0.y, rect.r0.x + rectangleExtents.x, rect.r0.y + rectangleExtents.y, -rect.r0.x);
        const vector4_type n_z = denorm_n_z / nbl::hlsl::sqrt(vector4_type(rect.r0.z * rect.r0.z) + denorm_n_z * denorm_n_z);
        const vector4_type cosGamma = vector4_type(
            -n_z[0] * n_z[1],
            -n_z[1] * n_z[2],
            -n_z[2] * n_z[3],
            -n_z[3] * n_z[0]
        );

        scalar_type p = math::getSumofArccosAB(cosGamma[0], cosGamma[1]);
        scalar_type q = math::getSumofArccosAB(cosGamma[2], cosGamma[3]);

        const scalar_type k = 2 * numbers::pi<scalar_type> - q;
        const scalar_type b0 = n_z[0];
        const scalar_type b1 = n_z[2];
        S = p + q - 2 * numbers::pi<scalar_type>;

        const scalar_type CLAMP_EPS = 1e-5f;

        // flip z axsis if rect.r0.z > 0
        const uint32_t zFlipMask = (asuint(rect.r0.z) ^ 0x80000000u) & 0x80000000u;
        rect.r0.z = asfloat(asuint(rect.r0.z) ^ zFlipMask);
        vector3_type r1 = rect.r0 + vector3_type(rectangleExtents.x, rectangleExtents.y, 0);

        const scalar_type au = uv.x * S + k;
        const scalar_type fu = (nbl::hlsl::cos(au) * b0 - b1) / nbl::hlsl::sin(au);
        const scalar_type cu_2 = nbl::hlsl::max(fu * fu + b0 * b0, 1.f); // forces `cu` to be in [-1,1]
        const scalar_type cu = asfloat(asuint(1.0 / nbl::hlsl::sqrt(cu_2)) ^ (asuint(fu) & 0x80000000u));

        scalar_type xu = -(cu * rect.r0.z) * 1.0 / nbl::hlsl::sqrt(1 - cu * cu);
        xu = nbl::hlsl::clamp(xu, rect.r0.x, r1.x); // avoid Infs
        const scalar_type d_2 = xu * xu + rect.r0.z * rect.r0.z;
        const scalar_type d = nbl::hlsl::sqrt(d_2);

        const scalar_type h0 = rect.r0.y / nbl::hlsl::sqrt(d_2 + rect.r0.y * rect.r0.y);
        const scalar_type h1 = r1.y / nbl::hlsl::sqrt(d_2 + r1.y * r1.y);
        const scalar_type hv = h0 + uv.y * (h1 - h0), hv2 = hv * hv;
        const scalar_type yv = (hv2 < 1 - CLAMP_EPS) ? (hv * d) / nbl::hlsl::sqrt(1 - hv2) : r1.y;

        return vector2_type((xu - rect.r0.x) / rectangleExtents.x, (yv - rect.r0.y) / rectangleExtents.y);
    }

    shapes::SphericalRectangle<T> rect;
};

}
}
}

#endif
