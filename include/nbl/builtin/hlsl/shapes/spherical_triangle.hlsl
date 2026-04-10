// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SHAPES_SPHERICAL_TRIANGLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SHAPES_SPHERICAL_TRIANGLE_INCLUDED_

#include <nbl/builtin/hlsl/tgmath.hlsl>
#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>
#include <nbl/builtin/hlsl/math/fast_acos.hlsl>
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
        const vector3_type normalizedVerts[3] = {
            nbl::hlsl::normalize(vertices[0] - origin),
            nbl::hlsl::normalize(vertices[1] - origin),
            nbl::hlsl::normalize(vertices[2] - origin)
        };
        return createFromUnitSphereVertices(normalizedVerts);
    }

    static SphericalTriangle<T> createFromUnitSphereVertices(const vector3_type normalizedVertices[3])
    {
        SphericalTriangle<T> retval;
        retval.vertices[0] = normalizedVertices[0];
        retval.vertices[1] = normalizedVertices[1];
        retval.vertices[2] = normalizedVertices[2];

        retval.cos_sides = hlsl::clamp(
            vector3_type(hlsl::dot(retval.vertices[1], retval.vertices[2]), hlsl::dot(retval.vertices[2], retval.vertices[0]), hlsl::dot(retval.vertices[0], retval.vertices[1])),
            hlsl::promote<vector3_type>(-1.0), hlsl::promote<vector3_type>(1.0));
        const vector3_type sin_sides2 = hlsl::promote<vector3_type>(1.0) - retval.cos_sides * retval.cos_sides;
        retval.csc_sides = hlsl::rsqrt<vector3_type>(sin_sides2); // only need side B for generate, we still have it for projectedSolidAngle()

        // Both vertices and angles at the vertices are denoted by the same upper case letters A, B, and C. The angles A, B, C of the triangle are equal to the angles between the planes that intersect the surface of the sphere or,
        // equivalently, the angles between the tangent vectors of the great circle arcs where they meet at the vertices. Angles are in radians. The angles of proper spherical triangles are (by convention) less than PI
        // degenerate triangle: any side has near-zero sin, so csc blows up
        if (hlsl::any<vector<bool, 3> >(retval.csc_sides >= hlsl::promote<vector3_type>(numeric_limits<scalar_type>::max)))
        {
            retval.cos_vertices = hlsl::promote<vector3_type>(0.0);
            retval.sin_vertices = hlsl::promote<vector3_type>(0.0);
            retval.solid_angle = 0;
            return retval;
        }

        //  cos_a - cos_b * cos_c - (1/sin_b  *  1/sin_c)
        retval.cos_vertices = hlsl::clamp((retval.cos_sides - retval.cos_sides.yzx * retval.cos_sides.zxy) * retval.csc_sides.yzx * retval.csc_sides.zxy, hlsl::promote<vector3_type>(-1.0), hlsl::promote<vector3_type>(1.0)); // using Spherical Law of Cosines
        retval.sin_vertices = hlsl::sqrt(hlsl::promote<vector3_type>(1.0) - retval.cos_vertices * retval.cos_vertices);

        // Fast acos overshoot makes the solid angle slightly too large, which causes
        // generate() to place samples outside the triangle. poly3 (~6.9e-5 error) fails
        // the 1e-6 generatedInside tolerance; poly4 (~8.6e-6) and poly5 (~1.1e-6) are tighter.
        // Standard acos avoids this entirely at the cost of one transcendental call.
        // Benchmarks show fast acos is no faster here -- likely because the surrounding
        // code already saturates FMA throughput, so the SFU acos runs in parallel for free.
        math::sincos_accumulator<scalar_type> angle_adder = math::sincos_accumulator<scalar_type>::create(retval.cos_vertices[0], retval.sin_vertices[0]);
        angle_adder.addAngle(retval.cos_vertices[1], retval.sin_vertices[1]);
        angle_adder.addAngle(retval.cos_vertices[2], retval.sin_vertices[2]);
        // Use the clamped variant because addAngle() sum-of-products can push
        // the accumulated cosine slightly outside [-1,1] on GPU, making acos
        // return NaN. GPU max(NaN,0)=0 then silently zeroes the solid angle.
        retval.solid_angle = hlsl::max(angle_adder.getClampedSumOfArccosMinusPi(), scalar_type(0.0));

        return retval;
    }

    scalar_type projectedSolidAngle(const vector3_type receiverNormal) NBL_CONST_MEMBER_FUNC
    {
        if (solid_angle <= numeric_limits<scalar_type>::epsilon)
            return 0;

        matrix<scalar_type, 3, 3> awayFromEdgePlane;
        awayFromEdgePlane[0] = hlsl::cross(vertices[1], vertices[2]) * csc_sides[0];
        awayFromEdgePlane[1] = hlsl::cross(vertices[2], vertices[0]) * csc_sides[1];
        awayFromEdgePlane[2] = hlsl::cross(vertices[0], vertices[1]) * csc_sides[2];
        // The ABS makes it so that the computation is correct for an `abs(cos(theta))` factor which is the projected solid angle used for a BSDF.
        // Proof: Kelvin-Stokes theorem, if you split the set into two along the horizon with constant CCW winding, the `cross` along the shared edge
        // goes in different directions and cancels out, while `acos` of the clipped great arcs corresponding to polygon edges add up to the original sides again.
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
        // So that triangle covering almost whole hemisphere sums to PI
        return hlsl::dot(pyramidAngles, externalProducts) * scalar_type(0.5);
    }

    vector3_type vertices[3];
    // angles of vertices with origin, so the sides are INSIDE the sphere
    vector3_type cos_sides;
    vector3_type csc_sides;
    // angles between arcs on the sphere, so angles in the TANGENT plane at each vertex
    vector3_type cos_vertices;
    vector3_type sin_vertices;
    scalar_type solid_angle;
};

}
}
}

#endif
