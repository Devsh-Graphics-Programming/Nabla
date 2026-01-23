// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_SPHERICAL_TRIANGLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_SPHERICAL_TRIANGLE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>
#include <nbl/builtin/hlsl/math/quaternions.hlsl>
#include <nbl/builtin/hlsl/shapes/spherical_triangle.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{

template<typename T>
struct SphericalTriangle
{
    using scalar_type = T;
    using vector2_type = vector<T, 2>;
    using vector3_type = vector<T, 3>;

    static SphericalTriangle<T> create(NBL_CONST_REF_ARG(shapes::SphericalTriangle<T>) tri)
    {
        SphericalTriangle<T> retval;
        retval.tri = tri;
        vector3_type cos_vertices, sin_vertices;
        retval.solidAngle = tri.solidAngle(cos_vertices, sin_vertices);
        retval.cosA = cos_vertices[0];
        retval.sinA = sin_vertices[0];
        return retval;
    }

    vector3_type generate(scalar_type cos_c, scalar_type csc_b, const vector2_type u)
    {
        scalar_type negSinSubSolidAngle,negCosSubSolidAngle;
        math::sincos(solidAngle * u.x - numbers::pi<scalar_type>, negSinSubSolidAngle, negCosSubSolidAngle);

        const scalar_type p = negCosSubSolidAngle * sinA - negSinSubSolidAngle * cosA;
        const scalar_type q = -negSinSubSolidAngle * sinA - negCosSubSolidAngle * cosA;
        
        // TODO: we could optimize everything up and including to the first slerp, because precision here is just godawful
        scalar_type u_ = q - cosA;
        scalar_type v_ = p + sinA * cos_c;

        // the slerps could probably be optimized by sidestepping `normalize` calls and accumulating scaling factors
        vector3_type C_s = tri.vertices[0];
        if (csc_b < numeric_limits<scalar_type>::max)
        {
            const scalar_type cosAngleAlongAC = ((v_ * q - u_ * p) * cosA - v_) / ((v_ * p + u_ * q) * sinA);
            if (nbl::hlsl::abs(cosAngleAlongAC) < 1.f)
                C_s += math::quaternion<scalar_type>::slerp_delta(tri.vertices[0], tri.vertices[2] * csc_b, cosAngleAlongAC);
        }

        vector3_type retval = tri.vertices[1];
        const scalar_type cosBC_s = nbl::hlsl::dot(C_s, tri.vertices[1]);
        const scalar_type csc_b_s = 1.0 / nbl::hlsl::sqrt(1.0 - cosBC_s * cosBC_s);
        if (csc_b_s < numeric_limits<scalar_type>::max)
        {
            const scalar_type cosAngleAlongBC_s = nbl::hlsl::clamp(1.0 + cosBC_s * u.y - u.y, -1.f, 1.f);
            if (nbl::hlsl::abs(cosAngleAlongBC_s) < 1.f)
                retval += math::quaternion<scalar_type>::slerp_delta(tri.vertices[1], C_s * csc_b_s, cosAngleAlongBC_s);
        }
        return retval;
    }

    vector3_type generate(NBL_REF_ARG(scalar_type) rcpPdf, const vector2_type u)
    {
        const scalar_type cos_c = tri.cos_sides[2];
        const scalar_type csc_b = tri.csc_sides[1];

        rcpPdf = solidAngle;

        return generate(cos_c, csc_b, u);
    }

    vector2_type generateInverse(NBL_REF_ARG(scalar_type) pdf, scalar_type cos_c, scalar_type csc_c, const vector3_type L)
    {
        pdf = 1.0 / solidAngle;

        const scalar_type cosAngleAlongBC_s = nbl::hlsl::dot(L, tri.vertices[1]);
        const scalar_type csc_a_ = 1.0 / nbl::hlsl::sqrt(1.0 - cosAngleAlongBC_s * cosAngleAlongBC_s);
        const scalar_type cos_b_ = nbl::hlsl::dot(L, tri.vertices[0]);

        const scalar_type cosB_ = (cos_b_ - cosAngleAlongBC_s * cos_c) * csc_a_ * csc_c;
        const scalar_type sinB_ = nbl::hlsl::sqrt(1.0 - cosB_ * cosB_);

        const scalar_type cosC_ = sinA * sinB_* cos_c - cosA * cosB_;
        const scalar_type sinC_ = nbl::hlsl::sqrt(1.0 - cosC_ * cosC_);

        math::sincos_accumulator<scalar_type> angle_adder = math::sincos_accumulator<scalar_type>::create(cosA, sinA);
        angle_adder.addAngle(cosB_, sinB_);
        angle_adder.addAngle(cosC_, sinC_);
        const scalar_type subTriSolidAngleRatio = (angle_adder.getSumofArccos() - numbers::pi<scalar_type>) * pdf;
        const scalar_type u = subTriSolidAngleRatio > numeric_limits<scalar_type>::min ? subTriSolidAngleRatio : 0.0;

        const scalar_type cosBC_s = (cosA + cosB_ * cosC_) / (sinB_ * sinC_);
        const scalar_type v = (1.0 - cosAngleAlongBC_s) / (1.0 - (cosBC_s < bit_cast<float>(0x3f7fffff) ? cosBC_s : cos_c));

        return vector2_type(u,v);
    }

    vector2_type generateInverse(NBL_REF_ARG(scalar_type) pdf, const vector3_type L)
    {
        const scalar_type cos_c = tri.cos_sides[2];
        const scalar_type csc_c = tri.csc_sides[2];

        return generateInverse(pdf, cos_c, csc_c, L);
    }

    shapes::SphericalTriangle<T> tri;
    scalar_type solidAngle;
    scalar_type cosA;
    scalar_type sinA;
};

}
}
}

#endif
