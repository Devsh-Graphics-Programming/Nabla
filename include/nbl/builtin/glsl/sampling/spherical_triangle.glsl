#ifndef _NBL_BUILTIN_GLSL_SAMPLING_SPHERICAL_TRIANGLE_INCLUDED_
#define _NBL_BUILTIN_GLSL_SAMPLING_SPHERICAL_TRIANGLE_INCLUDED_

#include <nbl/builtin/glsl/math/quaternions.glsl>

#include <nbl/builtin/glsl/shapes/triangle.glsl>

// WARNING: can and will return NAN if one or three of the triangle edges are near zero length
// this function could use some more optimizing
vec3 nbl_glsl_sampling_generateSphericalTriangleSample(in float solidAngle, in vec3 cos_vertices, in vec3 sin_vertices, in float cos_a, in float cos_c, in float csc_b, in float csc_c, in mat3 sphericalVertices, in vec2 u)
{
    // this part literally cannot be optimized further
    float negSinSubSolidAngle,negCosSubSolidAngle;
    nbl_glsl_sincos(solidAngle*u.x-nbl_glsl_PI,negSinSubSolidAngle,negCosSubSolidAngle);

	const float p = negCosSubSolidAngle*sin_vertices[0]-negSinSubSolidAngle*cos_vertices[0];
	const float q = -negSinSubSolidAngle*sin_vertices[0]-negCosSubSolidAngle*cos_vertices[0];
    
    // TODO: we could optimize everything up and including to the first slerp, because precision here is just godawful
	float u_ = q - cos_vertices[0];
	float v_ = p + sin_vertices[0]*cos_c;

    // the slerps could probably be optimized by sidestepping `normalize` calls and accumulating scaling factors
    vec3 C_s = sphericalVertices[0];
    if (csc_b<FLT_MAX)
    {
        const float cosAngleAlongAC = ((v_ * q - u_ * p) * cos_vertices[0] - v_) / ((v_ * p + u_ * q) * sin_vertices[0]);
        if (abs(cosAngleAlongAC)<1.f)
            C_s += nbl_glsl_slerp_delta_impl(sphericalVertices[0],sphericalVertices[2]*csc_b,cosAngleAlongAC);
    }

    vec3 retval = sphericalVertices[1];
    const float cosBC_s = dot(C_s,sphericalVertices[1]);
    const float csc_b_s = inversesqrt(1.0-cosBC_s*cosBC_s);
    if (csc_b_s<FLT_MAX)
    {
        const float cosAngleAlongBC_s = clamp(1.0+cosBC_s*u.y-u.y,-1.f,1.f);
        if (abs(cosAngleAlongBC_s)<1.f)
            retval += nbl_glsl_slerp_delta_impl(sphericalVertices[1],C_s*csc_b_s,cosAngleAlongBC_s);
    }
    return retval;
}
vec3 nbl_glsl_sampling_generateSphericalTriangleSample(out float rcpPdf, in mat3 sphericalVertices, in vec2 u)
{
    // for angles between view-to-vertex vectors
    float cos_a,cos_c,csc_b,csc_c;
    // Both vertices and angles at the vertices are denoted by the same upper case letters A, B, and C. The angles A, B, C of the triangle are equal to the angles between the planes that intersect the surface of the sphere or, equivalently, the angles between the tangent vectors of the great circle arcs where they meet at the vertices. Angles are in radians. The angles of proper spherical triangles are (by convention) less than PI
    vec3 cos_vertices,sin_vertices;
    // get solid angle, which is also the reciprocal of the probability
    rcpPdf = nbl_glsl_shapes_SolidAngleOfTriangle(sphericalVertices,cos_vertices,sin_vertices,cos_a,cos_c,csc_b,csc_c);

    return nbl_glsl_sampling_generateSphericalTriangleSample(rcpPdf,cos_vertices,sin_vertices,cos_a,cos_c,csc_b,csc_c,sphericalVertices,u);
}
vec3 nbl_glsl_sampling_generateSphericalTriangleSample(out float rcpPdf, in mat3 vertices, in vec3 origin, in vec2 u)
{
    return nbl_glsl_sampling_generateSphericalTriangleSample(rcpPdf,nbl_glsl_shapes_getSphericalTriangle(vertices,origin),u);
}


//
vec2 nbl_glsl_sampling_generateSphericalTriangleSampleInverse(out float pdf, in mat3 sphericalVertices, in vec3 L)
{
    // for angles between view-to-vertex vectors
    float cos_a,cos_c,csc_b,csc_c;
    // Both vertices and angles at the vertices are denoted by the same upper case letters A, B, and C. The angles A, B, C of the triangle are equal to the angles between the planes that intersect the surface of the sphere or, equivalently, the angles between the tangent vectors of the great circle arcs where they meet at the vertices. Angles are in radians. The angles of proper spherical triangles are (by convention) less than PI
    vec3 cos_vertices,sin_vertices;
    // get solid angle, which is also the reciprocal of the probability
    pdf = 1.0/nbl_glsl_shapes_SolidAngleOfTriangle(sphericalVertices,cos_vertices,sin_vertices,cos_a,cos_c,csc_b,csc_c);

    // get the modified B angle of the first subtriangle by getting it from the triangle formed by vertices A,B and the light sample L
    const float cosAngleAlongBC_s = dot(L,sphericalVertices[1]);
    const float csc_a_ = inversesqrt(1.0-cosAngleAlongBC_s*cosAngleAlongBC_s); // only NaN if L is close to B which implies v=0
    const float cos_b_ = dot(L,sphericalVertices[0]);

    const float cosB_ = (cos_b_-cosAngleAlongBC_s*cos_c)*csc_a_*csc_c; // only NaN if `csc_a_` (L close to B) is NaN OR if `csc_c` is NaN (which would mean zero solid angle triangle to begin with, so uv can be whatever)
    const float sinB_ = sqrt(1.0-cosB_*cosB_);

    // now all that remains is to obtain the modified C angle, which is the angle at the unknown vertex `C_s`
    const float cosC_ = sin_vertices[0]*sinB_*cos_c-cos_vertices[0]*cosB_; // if cosB_ is NaN then cosC_ doesn't matter because the subtriangle has zero Solid Angle (we could pretend its `-cos_vertices[0]`)
    const float sinC_ = sqrt(1.0-cosC_*cosC_);

    const float subTriSolidAngleRatio = nbl_glsl_getArccosSumofABC_minus_PI(cos_vertices[0],cosB_,cosC_,sin_vertices[0],sinB_,sinC_)*pdf; // will only be NaN if either the original triangle has zero solid angle or the subtriangle has zero solid angle (all can be satisfied with u=0) 
    const float u = subTriSolidAngleRatio>FLT_MIN ? subTriSolidAngleRatio:0.0; // tiny overruns of u>1.0 will not affect the PDF much because a bilinear warp is used and the gradient has a bound (won't be true if LTC will get used)

    // INF if any angle is 0 degrees, which implies L lays along BA arc, if the angle at A is PI minus the angle at either B_ or C_ while the other of C_ or B_ has a zero angle, we get a NaN (which is also a zero solid angle subtriangle, implying L along AB arc)
    const float cosBC_s = (cos_vertices[0]+cosB_*cosC_)/(sinB_*sinC_);
    // if cosBC_s is really large then we have numerical issues (should be 1.0 which means the arc is really short), if its NaN then either the original or sub-triangle has zero solid angle, in both cases we can consider that the BC_s arc is actually the BA arc and substitute
    const float v = (1.0-cosAngleAlongBC_s)/(1.0-(cosBC_s<uintBitsToFloat(0x3f7fffff) ? cosBC_s:cos_c));

    return vec2(u,v);
}

#endif
