// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SHAPES_SPHERICAL_TRIANGLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SHAPES_SPHERICAL_TRIANGLE_INCLUDED_

#include <nbl/builtin/hlsl/tgmath.hlsl>
#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>
#include <nbl/builtin/hlsl/math/angle_adding.hlsl>
#include <nbl/builtin/hlsl/numbers.hlsl>

namespace nbl
{
namespace hlsl
{
namespace shapes
{

template<typename T>
struct SphericalTriangle
{
    using scalar_type = T;
    using vector3_type = vector<T, 3>;

    static SphericalTriangle<T> create(const vector3_type vertices[3], const vector3_type origin)
    {
        SphericalTriangle<T> retval;
        retval.vertices[0] = nbl::hlsl::normalize(vertices[0] - origin);
        retval.vertices[1] = nbl::hlsl::normalize(vertices[1] - origin);
        retval.vertices[2] = nbl::hlsl::normalize(vertices[2] - origin);
        retval.cos_sides = vector3_type(hlsl::dot(retval.vertices[1], retval.vertices[2]), hlsl::dot(retval.vertices[2], retval.vertices[0]), hlsl::dot(retval.vertices[0], retval.vertices[1]));
        const vector3_type sin_sides2 = hlsl::promote<vector3_type>(1.0) - retval.cos_sides * retval.cos_sides;
        retval.csc_sides = hlsl::rsqrt<vector3_type>(sin_sides2);
        return retval;
    }

    // checks if any angles are small enough to disregard
    bool pyramidAngles()
    {
        return hlsl::any<vector<bool, 3> >(csc_sides >= hlsl::promote<vector3_type>(numeric_limits<scalar_type>::max));
    }

    scalar_type solidAngle(NBL_REF_ARG(vector3_type) cos_vertices, NBL_REF_ARG(vector3_type) sin_vertices)
    {
        if (pyramidAngles())
            return 0.f;

        // Both vertices and angles at the vertices are denoted by the same upper case letters A, B, and C. The angles A, B, C of the triangle are equal to the angles between the planes that intersect the surface of the sphere or,
        // equivalently, the angles between the tangent vectors of the great circle arcs where they meet at the vertices. Angles are in radians. The angles of proper spherical triangles are (by convention) less than PI
        cos_vertices = hlsl::clamp((cos_sides - cos_sides.yzx * cos_sides.zxy) * csc_sides.yzx * csc_sides.zxy, hlsl::promote<vector3_type>(-1.0), hlsl::promote<vector3_type>(1.0)); // using Spherical Law of Cosines (TODO: do we need to clamp anymore? since the pyramid angles method introduction?) 
        sin_vertices = hlsl::sqrt(hlsl::promote<vector3_type>(1.0) - cos_vertices * cos_vertices);

        math::sincos_accumulator<scalar_type> angle_adder = math::sincos_accumulator<scalar_type>::create(cos_vertices[0], sin_vertices[0]);
        angle_adder.addAngle(cos_vertices[1], sin_vertices[1]);
        angle_adder.addAngle(cos_vertices[2], sin_vertices[2]);
        return angle_adder.getSumofArccos() - numbers::pi<scalar_type>;
    }

    scalar_type solidAngle()
    {
        vector3_type dummy0,dummy1;
        return solidAngle(dummy0,dummy1);
    }

    scalar_type projectedSolidAngle(const vector3_type receiverNormal, NBL_REF_ARG(vector3_type) cos_vertices)
    {
        if (pyramidAngles())
            return 0.f;

        cos_vertices = hlsl::clamp((cos_sides - cos_sides.yzx * cos_sides.zxy) * csc_sides.yzx * csc_sides.zxy, hlsl::promote<vector3_type>(-1.0), hlsl::promote<vector3_type>(1.0));

        matrix<scalar_type, 3, 3> awayFromEdgePlane;
        awayFromEdgePlane[0] = hlsl::cross(vertices[1], vertices[2]) * csc_sides[0];
        awayFromEdgePlane[1] = hlsl::cross(vertices[2], vertices[0]) * csc_sides[1];
        awayFromEdgePlane[2] = hlsl::cross(vertices[0], vertices[1]) * csc_sides[2];
        // The ABS makes it so that the computation is correct for an `abs(cos(theta))` factor which is the projected solid angle used for a BSDF
        // Proof: Kelvin-Stokes theorem, if you split the set into two along the horizon with constant CCW winding, the `cross` along the shared edge goes in different directions and cancels out,
        // while `acos` of the clipped great arcs corresponding to polygon edges add up to the original sides again
        const vector3_type externalProducts = hlsl::abs(hlsl::mul(/* transposed already */awayFromEdgePlane, receiverNormal));

        // Far TODO: `cross(A,B)*acos(dot(A,B))/sin(1-dot^2)` can be done with `cross*acos_csc_approx(dot(A,B))`
        // We could skip the `csc_sides` factor, and computing `pyramidAngles` and replace them with this approximation weighting before the dot product with the receiver notmal
        // The curve fit "revealed in a dream" to me is `exp2(F(log2(x+1)))` where `F(u)` is a polynomial, so far I've calculated `F = (1-u)0.635+(1-u^2)0.0118` which gives <5% error until 165 degrees
        // I have a feeling that a polynomial of ((Au+B)u+C)u+D could be sufficient if it has following properties:
        // `F(0) = 0` and
        // `F(u) <= log2(\frac{\cos^{-1}\left(2^{x}-1\right)}{\sqrt{1-\left(2^{x}-1\right)^{2}}})` because you want to consistently under-estimate the Projected Solid Angle to avoid creating energy
        // See https://www.desmos.com/calculator/sdptomhbju
        // Furthermore we could clip the polynomial calc to `Cu+D or `(Bu+C)u+D` for small arguments
        const vector3_type pyramidAngles = hlsl::acos<vector3_type>(cos_sides);
        // So that riangle covering almost whole hemisphere sums to PI
        return hlsl::dot(pyramidAngles, externalProducts) * scalar_type(0.5);
    }

    vector3_type vertices[3];
    vector3_type cos_sides;
    vector3_type csc_sides;
};

}
}
}

#endif
