// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_SPHERICAL_RECTANGLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_SPHERICAL_RECTANGLE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>
#include <nbl/builtin/hlsl/shapes/spherical_triangle.hlsl>

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
        const vector4_type n_z = denorm_n_z / hlsl::sqrt<vector4_type>(hlsl::promote<vector4_type>(rect.r0.z * rect.r0.z) + denorm_n_z * denorm_n_z);
        const vector4_type cosGamma = vector4_type(
            -n_z[0] * n_z[1],
            -n_z[1] * n_z[2],
            -n_z[2] * n_z[3],
            -n_z[3] * n_z[0]
        );

        math::sincos_accumulator<scalar_type> angle_adder = math::sincos_accumulator<scalar_type>::create(cosGamma[0]);
        angle_adder.addCosine(cosGamma[1]);
        scalar_type p = angle_adder.getSumofArccos();
        angle_adder = math::sincos_accumulator<scalar_type>::create(cosGamma[2]);
        angle_adder.addCosine(cosGamma[3]);
        scalar_type q = angle_adder.getSumofArccos();

        const scalar_type k = scalar_type(2.0) * numbers::pi<scalar_type> - q;
        const scalar_type b0 = n_z[0];
        const scalar_type b1 = n_z[2];
        S = p + q - scalar_type(2.0) * numbers::pi<scalar_type>;

        const scalar_type CLAMP_EPS = 1e-5;

        // flip z axis if rect.r0.z > 0
        rect.r0.z = ieee754::flipSignIfRHSNegative<scalar_type>(rect.r0.z, -rect.r0.z);
        vector3_type r1 = rect.r0 + vector3_type(rectangleExtents.x, rectangleExtents.y, 0);

        const scalar_type au = uv.x * S + k;
        const scalar_type fu = (hlsl::cos<scalar_type>(au) * b0 - b1) / hlsl::sin<scalar_type>(au);
        const scalar_type cu_2 = hlsl::max<scalar_type>(fu * fu + b0 * b0, 1.f); // forces `cu` to be in [-1,1]
        const scalar_type cu = ieee754::flipSignIfRHSNegative<scalar_type>(scalar_type(1.0) / hlsl::sqrt<scalar_type>(cu_2), fu);

        scalar_type xu = -(cu * rect.r0.z) / hlsl::sqrt<scalar_type>(scalar_type(1.0) - cu * cu);
        xu = hlsl::clamp<scalar_type>(xu, rect.r0.x, r1.x); // avoid Infs
        const scalar_type d_2 = xu * xu + rect.r0.z * rect.r0.z;
        const scalar_type d = hlsl::sqrt<scalar_type>(d_2);

        const scalar_type h0 = rect.r0.y / hlsl::sqrt<scalar_type>(d_2 + rect.r0.y * rect.r0.y);
        const scalar_type h1 = r1.y / hlsl::sqrt<scalar_type>(d_2 + r1.y * r1.y);
        const scalar_type hv = h0 + uv.y * (h1 - h0);
        const scalar_type hv2 = hv * hv;
        const scalar_type yv = hlsl::mix(r1.y, (hv * d) / hlsl::sqrt<scalar_type>(scalar_type(1.0) - hv2), hv2 < scalar_type(1.0) - CLAMP_EPS);

        return vector2_type((xu - rect.r0.x) / rectangleExtents.x, (yv - rect.r0.y) / rectangleExtents.y);
    }

    shapes::SphericalRectangle<T> rect;
};

}
}
}

#endif
