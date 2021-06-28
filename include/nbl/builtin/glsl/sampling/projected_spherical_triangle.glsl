#ifndef _NBL_BUILTIN_GLSL_SAMPLING_PROJECTED_SPHERICAL_TRIANGLE_INCLUDED_
#define _NBL_BUILTIN_GLSL_SAMPLING_PROJECTED_SPHERICAL_TRIANGLE_INCLUDED_

#include <nbl/builtin/glsl/sampling/bilinear.glsl>
#include <nbl/builtin/glsl/sampling/spherical_triangle.glsl>

vec4 nbl_glsl_sampling_computeBilinearPatchForProjSphericalTriangle(in mat3 sphericalVertices, in vec3 receiverNormal, in bool isBSDF)
{
    // a positive would prevent us from a scenario where `nbl_glsl_sampling_rcpProbBilinearSample` will return NAN
    const float minimumProjSolidAngle = 0.0;
    
    // take abs of the value if we have a BSDF, clamp to 0 otherwise
    const vec3 bxdfPdfAtVertex = nbl_glsl_conditionalAbsOrMax(isBSDF,transpose(sphericalVertices)*receiverNormal,vec3(minimumProjSolidAngle));

    // the swizzle needs to match the mapping of the [0,1]^2 square to the triangle vertices
    return bxdfPdfAtVertex.yyxz;
}

// There are two different modes of sampling, one for BSDF and one for BRDF (depending if we throw away bottom hemisphere or not)
vec3 nbl_glsl_sampling_generateProjectedSphericalTriangleSample(out float rcpPdf, in float solidAngle, in vec3 cos_vertices, in vec3 sin_vertices, in float cos_a, in float cos_c, in float csc_b, in float csc_c, in mat3 sphericalVertices, in vec3 receiverNormal, in bool isBSDF, vec2 u)
{
    // pre-warp according to proj solid angle approximation
    u = nbl_glsl_sampling_generateBilinearSample(rcpPdf,nbl_glsl_sampling_computeBilinearPatchForProjSphericalTriangle(sphericalVertices,receiverNormal,isBSDF),u);

    // now warp the points onto a spherical triangle
    const vec3 L = nbl_glsl_sampling_generateSphericalTriangleSample(solidAngle,cos_vertices,sin_vertices,cos_a,cos_c,csc_b,csc_c,sphericalVertices,u);
    rcpPdf *= solidAngle;

    return L;
}
vec3 nbl_glsl_sampling_generateProjectedSphericalTriangleSample(out float rcpPdf, in vec3 cos_sides, in vec3 csc_sides, in vec3 cos_vertices, in mat3 sphericalVertices, in vec3 receiverNormal, in bool isBSDF, vec2 u)
{
    const vec3 sin_vertices = sqrt(vec3(1.f)-cos_vertices*cos_vertices);
    const float cos_a = cos_sides[0];
    const float cos_c = cos_sides[2];
    const float csc_b = csc_sides[1];
    const float csc_c = csc_sides[2];
    const float solidAngle = nbl_glsl_getArccosSumofABC_minus_PI(cos_vertices[0],cos_vertices[1],cos_vertices[2],sin_vertices[0],sin_vertices[1],sin_vertices[2]);
    return nbl_glsl_sampling_generateProjectedSphericalTriangleSample(rcpPdf,solidAngle,cos_vertices,sin_vertices,cos_a,cos_c,csc_b,csc_c,sphericalVertices,receiverNormal,isBSDF,u);
}
vec3 nbl_glsl_sampling_generateProjectedSphericalTriangleSample(out float rcpPdf, in mat3 sphericalVertices, in vec3 receiverNormal, in bool isBSDF, vec2 u)
{
    float cos_a,cos_c,csc_b,csc_c;
    vec3 cos_vertices,sin_vertices;
    const float solidAngle = nbl_glsl_shapes_SolidAngleOfTriangle(sphericalVertices,cos_vertices,sin_vertices,cos_a,cos_c,csc_b,csc_c);
    return nbl_glsl_sampling_generateProjectedSphericalTriangleSample(rcpPdf,solidAngle,cos_vertices,sin_vertices,cos_a,cos_c,csc_b,csc_c,sphericalVertices,receiverNormal,isBSDF,u);
}
vec3 nbl_glsl_sampling_generateProjectedSphericalTriangleSample(out float rcpPdf, in mat3 vertices, in vec3 origin, in vec3 receiverNormal, in bool isBSDF, in vec2 u)
{
    return nbl_glsl_sampling_generateProjectedSphericalTriangleSample(rcpPdf,nbl_glsl_shapes_getSphericalTriangle(vertices,origin),receiverNormal,isBSDF,u);
}

//
float nbl_glsl_sampling_probProjectedSphericalTriangleSample(in mat3 sphericalVertices, in vec3 receiverNormal, in bool receiverWasBSDF, in vec3 L)
{
    float pdf;
    const vec2 u = nbl_glsl_sampling_generateSphericalTriangleSampleInverse(pdf,sphericalVertices,L);

    return pdf*nbl_glsl_sampling_probBilinearSample(nbl_glsl_sampling_computeBilinearPatchForProjSphericalTriangle(sphericalVertices,receiverNormal,receiverWasBSDF),u);
}

#endif
