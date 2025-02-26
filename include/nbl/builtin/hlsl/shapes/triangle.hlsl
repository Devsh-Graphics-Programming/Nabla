// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SHAPES_TRIANGLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SHAPES_TRIANGLE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>

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

    static SphericalTriangle<T> create(NBL_CONST_REF_ARG(vector3_type) vertex0, NBL_CONST_REF_ARG(vector3_type) vertex1, NBL_CONST_REF_ARG(vector3_type) vertex2, NBL_CONST_REF_ARG(vector3_type) origin)
    {
        SphericalTriangle<T> retval;
        retval.vertex0 = nbl::hlsl::normalize(vertex0 - origin);
        retval.vertex1 = nbl::hlsl::normalize(vertex1 - origin);
        retval.vertex2 = nbl::hlsl::normalize(vertex2 - origin);
        return retval;
    }

    bool pyramidAngles(NBL_REF_ARG(vector3_type) cos_sides, NBL_REF_ARG(vector3_type) csc_sides)
    {
        cos_sides = vector3_type(nbl::hlsl::dot(vertex1, vertex2), nbl::hlsl::dot(vertex2, vertex0), nbl::hlsl::dot(vertex0, vertex1));
        csc_sides = 1.0 / nbl::hlsl::sqrt((vector3_type)(1.f) - cos_sides * cos_sides);
        return nbl::hlsl::any(csc_sides >= (vector3_type)(numeric_limits<scalar_type>::max));
    }

    scalar_type solidAngleOfTriangle(NBL_REF_ARG(vector3_type) cos_vertices, NBL_REF_ARG(vector3_type) sin_vertices, NBL_REF_ARG(scalar_type) cos_a, NBL_REF_ARG(scalar_type) cos_c, NBL_REF_ARG(scalar_type) csc_b, NBL_REF_ARG(scalar_type) csc_c)
    {
        vector3_type cos_sides,csc_sides;
        if (pyramidAngles(cos_sides, csc_sides))
            return 0.f;

        // these variables might eventually get optimized out
        cos_a = cos_sides[0];
        cos_c = cos_sides[2];
        csc_b = csc_sides[1];
        csc_c = csc_sides[2];

        // Both vertices and angles at the vertices are denoted by the same upper case letters A, B, and C. The angles A, B, C of the triangle are equal to the angles between the planes that intersect the surface of the sphere or, equivalently, the angles between the tangent vectors of the great circle arcs where they meet at the vertices. Angles are in radians. The angles of proper spherical triangles are (by convention) less than PI
        cos_vertices = clamp((cos_sides - cos_sides.yzx * cos_sides.zxy) * csc_sides.yzx * csc_sides.zxy, (vector3_type)(-1.f), (vector3_type)1.f); // using Spherical Law of Cosines (TODO: do we need to clamp anymore? since the pyramid angles method introduction?) 
        sin_vertices = sqrt((vector3_type)1.f - cos_vertices * cos_vertices);

        return math::getArccosSumofABC_minus_PI(cos_vertices[0], cos_vertices[1], cos_vertices[2], sin_vertices[0], sin_vertices[1], sin_vertices[2]);
    }

    scalar_type solidAngleOfTriangle()
    {
        vector3_type dummy0,dummy1;
        scalar_type dummy2,dummy3,dummy4,dummy5;
        return solidAngleOfTriangle(dummy0,dummy1,dummy2,dummy3,dummy4,dummy5);
    }

    scalar_type projectedSolidAngleOfTriangle(NBL_CONST_REF_ARG(vector3_type) receiverNormal, NBL_REF_ARG(vector3_type) cos_sides, NBL_REF_ARG(vector3_type) csc_sides, NBL_REF_ARG(vector3_type) cos_vertices)
    {
        if (pyramidAngles(cos_sides, csc_sides))
            return 0.f;

        vector3_type awayFromEdgePlane0 = nbl::hlsl::cross(vertex1, vertex2) * csc_sides[0];
        vector3_type awayFromEdgePlane1 = nbl::hlsl::cross(vertex2, vertex0) * csc_sides[1];
        vector3_type awayFromEdgePlane2 = nbl::hlsl::cross(vertex0, vertex1) * csc_sides[2];

        // useless here but could be useful somewhere else
        cos_vertices[0] = nbl::hlsl::dot(awayFromEdgePlane1, awayFromEdgePlane2);
        cos_vertices[1] = nbl::hlsl::dot(awayFromEdgePlane2, awayFromEdgePlane0);
        cos_vertices[2] = nbl::hlsl::dot(awayFromEdgePlane0, awayFromEdgePlane1);
        // TODO: above dot products are in the wrong order, either work out which is which, or try all 6 permutations till it works
        cos_vertices = nbl::hlsl::clamp((cos_sides - cos_sides.yzx * cos_sides.zxy) * csc_sides.yzx * csc_sides.zxy, (vector3_type)(-1.f), (vector3_type)1.f);

        matrix<scalar_type, 3, 3> awayFromEdgePlane = matrix<scalar_type, 3, 3>(awayFromEdgePlane0, awayFromEdgePlane1, awayFromEdgePlane2);
        const vector3_type externalProducts = nbl::hlsl::abs(nbl::hlsl::mul(/* transposed already */awayFromEdgePlane, receiverNormal));

        const vector3_type pyramidAngles = acos(cos_sides);
        return nbl::hlsl::dot(pyramidAngles, externalProducts) / (2.f * numbers::pi<float>);
    }

    vector3_type vertex0;
    vector3_type vertex1;
    vector3_type vertex2;
};

}
}
}

#endif
