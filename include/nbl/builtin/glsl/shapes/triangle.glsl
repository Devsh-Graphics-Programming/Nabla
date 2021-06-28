#ifndef _NBL_BUILTIN_GLSL_SHAPES_TRIANGLE_INCLUDED_
#define _NBL_BUILTIN_GLSL_SHAPES_TRIANGLE_INCLUDED_

#include <nbl/builtin/glsl/math/functions.glsl>

//
mat3 nbl_glsl_shapes_getSphericalTriangle(in mat3 vertices, in vec3 origin)
{
    // the `normalize` cannot be optimized out
    return mat3(normalize(vertices[0]-origin),normalize(vertices[1]-origin),normalize(vertices[2]-origin));
}

// returns solid angle of a spherical triangle
// WARNING: can and will return NAN if one or three of the triangle edges are near zero length
// this function is beyond optimized.
float nbl_glsl_shapes_SolidAngleOfTriangle(in mat3 sphericalVertices, out vec3 cos_vertices, out vec3 sin_vertices, out float cos_a, out float cos_c, out float csc_b, out float csc_c)
{    
    // The sides are denoted by lower-case letters a, b, and c. On the unit sphere their lengths are numerically equal to the radian measure of the angles that the great circle arcs subtend at the centre. The sides of proper spherical triangles are (by convention) less than PI
    const vec3 cos_sides = vec3(dot(sphericalVertices[1],sphericalVertices[2]),dot(sphericalVertices[2],sphericalVertices[0]),dot(sphericalVertices[0],sphericalVertices[1]));
    const vec3 csc_sides = inversesqrt(vec3(1.0)-cos_sides*cos_sides);

    // these variables might eventually get optimized out
    cos_a = cos_sides[0];
    cos_c = cos_sides[2];
    csc_b = csc_sides[1];
    csc_c = csc_sides[2];
    
    // Both vertices and angles at the vertices are denoted by the same upper case letters A, B, and C. The angles A, B, C of the triangle are equal to the angles between the planes that intersect the surface of the sphere or, equivalently, the angles between the tangent vectors of the great circle arcs where they meet at the vertices. Angles are in radians. The angles of proper spherical triangles are (by convention) less than PI
    cos_vertices = (cos_sides-cos_sides.yzx*cos_sides.zxy)*csc_sides.yzx*csc_sides.zxy; // using Spherical Law of Cosines
    sin_vertices = sqrt(max(vec3(0.f),vec3(1.f)-cos_vertices*cos_vertices));
    
    // the solid angle of a triangle is the sum of its planar vertices' angles minus PI
    return nbl_glsl_getArccosSumofABC_minus_PI(cos_vertices[0],cos_vertices[1],cos_vertices[2],sin_vertices[0],sin_vertices[1],sin_vertices[2]);
}
float nbl_glsl_shapes_SolidAngleOfTriangle(in mat3 sphericalVertices)
{
    vec3 dummy0,dummy1;
    float dummy2,dummy3,dummy4,dummy5;
    return nbl_glsl_shapes_SolidAngleOfTriangle(sphericalVertices,dummy0,dummy1,dummy2,dummy3,dummy4,dummy5);
}
// returns solid angle of a triangle given by its world-space vertices and world-space viewing position
float nbl_glsl_shapes_SolidAngleOfTriangle(in mat3 vertices, in vec3 origin)
{
    return nbl_glsl_shapes_SolidAngleOfTriangle(nbl_glsl_shapes_getSphericalTriangle(vertices,origin));
}

#endif
