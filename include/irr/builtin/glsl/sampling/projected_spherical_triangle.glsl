#ifndef _NBL_BUILTIN_GLSL_SAMPLING_PROJECTED_SPHERICAL_TRIANGLE_INCLUDED_
#define _NBL_BUILTIN_GLSL_SAMPLING_PROJECTED_SPHERICAL_TRIANGLE_INCLUDED_

#include <irr/builtin/glsl/sampling/bilinear.glsl>
#include <irr/builtin/glsl/sampling/spherical_triangle.glsl>

vec4 irr_glsl_sampling_computeBilinearPatchForProjSphericalTriangle(in mat3 sphericalVertices, in vec3 receiverNormal, in bool isBSDF)
{
    // a positive would prevent us from a scenario where `irr_glsl_sampling_rcpProbBilinearSample` will return NAN
    const float minimumProjSolidAngle = 0.0;
    
    // take abs of the value if we have a BSDF, clamp to 0 otherwise
    const vec3 bxdfPdfAtVertex = irr_glsl_conditionalAbsOrMax(isBSDF,transpose(sphericalVertices)*receiverNormal,vec3(minimumProjSolidAngle));

    // the swizzle needs to match the mapping of the [0,1]^2 square to the triangle vertices
    return bxdfPdfAtVertex.yyxz;
}

// There are two different modes of sampling, one for BSDF and one for BRDF (depending if we throw away bottom hemisphere or not)
vec3 irr_glsl_sampling_generateProjectedSphericalTriangleSample(out float rcpPdf, in mat3 sphericalVertices, in vec3 receiverNormal, in bool isBSDF, vec2 u)
{
    // pre-warp according to proj solid angle approximation
    u = irr_glsl_sampling_generateBilinearSample(rcpPdf,irr_glsl_sampling_computeBilinearPatchForProjSphericalTriangle(sphericalVertices,receiverNormal,isBSDF),u);

    // now warp the points onto a spherical triangle
    float solidAngle;
    const vec3 L = irr_glsl_sampling_generateSphericalTriangleSample(solidAngle,sphericalVertices,u);
    rcpPdf *= solidAngle;

    return L;
}
vec3 irr_glsl_sampling_generateProjectedSphericalTriangleSample(out float rcpPdf, in mat3 vertices, in vec3 origin, in vec3 receiverNormal, in bool isBSDF, in vec2 u)
{
    return irr_glsl_sampling_generateProjectedSphericalTriangleSample(rcpPdf,irr_glsl_shapes_getSphericalTriangle(vertices,origin),receiverNormal,isBSDF,u);
}

//
float irr_glsl_sampling_probProjectedSphericalTriangleSample(in mat3 sphericalVertices, in vec3 receiverNormal, in bool receiverWasBSDF, in vec3 L)
{
    float pdf;
    const vec2 u = irr_glsl_sampling_generateSphericalTriangleSampleInverse(pdf,sphericalVertices,L);

    return pdf*irr_glsl_sampling_probBilinearSample(irr_glsl_sampling_computeBilinearPatchForProjSphericalTriangle(sphericalVertices,receiverNormal,receiverWasBSDF),u);
}

#endif
