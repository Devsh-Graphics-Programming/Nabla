
#ifndef _NBL_BUILTIN_HLSL_SHAPES_TRIANGLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SHAPES_TRIANGLE_INCLUDED_

#include <nbl/builtin/hlsl/math/functions.hlsl>

namespace nbl
{
namespace hlsl
{

bool3 greaterThanEqual(float3 x, float3 y)
{
	return select(x>=y, bool3(true, true, true), bool3(false, false, false));
}

namespace shapes
{

//
float3x3 getSphericalTriangle(in float3x3 vertices, in float3 origin)
{
	float3x3 tp_vertices = transpose(vertices);

    // the `normalize` cannot be optimized out
    return float3x3(normalize(tp_vertices[0]-origin), normalize(tp_vertices[1]-origin), normalize(tp_vertices[2]-origin));
}

// returns true if pyramid degenerated into a line
bool SphericalTrianglePyramidAngles(in float3x3 sphericalVertices, out float3 cos_sides, out float3 csc_sides)
{
	float3x3 tp_sphericalVertices = transpose(sphericalVertices);
    // The sides are denoted by lower-case letters a, b, and c.
    // On the unit sphere their lengths are numerically equal to the radian measure of the angles that the great circle arcs subtend at the centre.
    // The sides of proper spherical triangles are (by convention) less than PI
    cos_sides = float3(dot(tp_sphericalVertices[1],tp_sphericalVertices[2]),
    				   dot(tp_sphericalVertices[2],tp_sphericalVertices[0]),
    				   dot(tp_sphericalVertices[0],tp_sphericalVertices[1]));
    csc_sides = rsqrt(float3(1.f,1.f,1.f)-cos_sides*cos_sides);
    return any(greaterThanEqual(csc_sides, float3(FLT_MAX,FLT_MAX,FLT_MAX)));
}

// returns solid angle of a spherical triangle, this function is beyond optimized.
float SolidAngleOfTriangle(in float3x3 sphericalVertices, out float3 cos_vertices, out float3 sin_vertices, out float cos_a, out float cos_c, out float csc_b, out float csc_c)
{   
    float3 cos_sides,csc_sides;
    if (SphericalTrianglePyramidAngles(sphericalVertices,cos_sides,csc_sides))
        return 0.f;

    // these variables might eventually get optimized out
    cos_a = cos_sides[0];
    cos_c = cos_sides[2];
    csc_b = csc_sides[1];
    csc_c = csc_sides[2];
    
    // Both vertices and angles at the vertices are denoted by the same upper case letters A, B, and C. The angles A, B, C of the triangle are equal to the angles between the planes that intersect the surface of the sphere or, equivalently, the angles between the tangent vectors of the great circle arcs where they meet at the vertices. Angles are in radians. The angles of proper spherical triangles are (by convention) less than PI
    cos_vertices = clamp((cos_sides-cos_sides.yzx*cos_sides.zxy)*csc_sides.yzx*csc_sides.zxy,float3(-1.f,-1.f,-1.f),float3(1.f,1.f,1.f));
    // using Spherical Law of Cosines (TODO: do we need to clamp anymore? since the pyramid angles method introduction?) 
    sin_vertices = sqrt(float3(1.f,1.f,1.f)-cos_vertices*cos_vertices);
    
    // the solid angle of a triangle is the sum of its planar vertices' angles minus PI
    return getArccosSumofABC_minus_PI(cos_vertices[0],cos_vertices[1],cos_vertices[2],sin_vertices[0],sin_vertices[1],sin_vertices[2]);
}
float SolidAngleOfTriangle(in float3x3 sphericalVertices)
{
    float3 dummy0,dummy1;
    float dummy2,dummy3,dummy4,dummy5;
    return SolidAngleOfTriangle(sphericalVertices,dummy0,dummy1,dummy2,dummy3,dummy4,dummy5);
}
// returns solid angle of a triangle given by its world-space vertices and world-space viewing position
float SolidAngleOfTriangle(in float3x3 vertices, in float3 origin)
{
    return SolidAngleOfTriangle(getSphericalTriangle(vertices,origin));
}


// return projected solid angle of a spherical triangle
float ProjectedSolidAngleOfTriangle(in float3x3 sphericalVertices, in float3 receiverNormal, out float3 cos_sides, out float3 csc_sides, out float3 cos_vertices)
{
    if (SphericalTrianglePyramidAngles(sphericalVertices,cos_sides,csc_sides))
        return 0.f;

    float3x3 tp_sphericalVertices = transpose(sphericalVertices);

    const float3x3 awayFromEdgePlane = float3x3(
        cross(tp_sphericalVertices[1],tp_sphericalVertices[2])*csc_sides[0],
        cross(tp_sphericalVertices[2],tp_sphericalVertices[0])*csc_sides[1],
        cross(tp_sphericalVertices[0],tp_sphericalVertices[1])*csc_sides[2]
    );

    float3x3 tp_awayFromEdgePlane = transpose(awayFromEdgePlane);

    // useless here but could be useful somewhere else
    cos_vertices[0] = dot(tp_awayFromEdgePlane[1],tp_awayFromEdgePlane[2]);
    cos_vertices[1] = dot(tp_awayFromEdgePlane[2],tp_awayFromEdgePlane[0]);
    cos_vertices[2] = dot(tp_awayFromEdgePlane[0],tp_awayFromEdgePlane[1]);
    // TODO: above dot products are in the wrong order, either work out which is which, or try all 6 permutations till it works
    cos_vertices = clamp((cos_sides-cos_sides.yzx*cos_sides.zxy)*csc_sides.yzx*csc_sides.zxy,float3(-1.f,-1.f,-1.f),float3(1.f,1.f,1.f));

    const float3 externalProducts = abs(mul(tp_awayFromEdgePlane, receiverNormal));

    const float3 pyramidAngles = acos(cos_sides);
    return dot(pyramidAngles,externalProducts)/(2.f*PI);
}
float ProjectedSolidAngleOfTriangle(in float3x3 sphericalVertices, in float3 receiverNormal, out float3 cos_sides, out float3 csc_sides)
{
    float3 cos_vertices;
    return ProjectedSolidAngleOfTriangle(sphericalVertices,receiverNormal,cos_sides,csc_sides,cos_vertices);
}
float ProjectedSolidAngleOfTriangle(in float3x3 sphericalVertices, in float3 receiverNormal)
{
    float3 cos_sides,csc_sides;
    return ProjectedSolidAngleOfTriangle(sphericalVertices,receiverNormal,cos_sides,csc_sides);
}

}
}
}
#endif