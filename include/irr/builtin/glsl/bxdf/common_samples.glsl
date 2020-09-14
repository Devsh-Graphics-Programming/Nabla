#ifndef _IRR_BUILTIN_GLSL_BXDF_COMMON_SAMPLES_INCLUDED_
#define _IRR_BUILTIN_GLSL_BXDF_COMMON_SAMPLES_INCLUDED_

#include <irr/builtin/glsl/math/functions.glsl>

// do not use this struct in SSBO or UBO, its wasteful on memory
struct irr_glsl_BxDFSample
{
    vec3 L;  // incoming direction, normalized
    float TdotL; 
    float BdotL;
    float NdotL;

    float TdotH;
    float BdotH;
    float NdotH;
    float VdotH;
    // you don't need LdotH, you think you do, but you don't!
};

// require H and V already be normalized
// reflection from microfacet
irr_glsl_BxDFSample irr_glsl_createBRDFSample(in vec3 H, in vec3 V, in float VdotH, in mat3 m)
{
    irr_glsl_BxDFSample s;

    vec3 L = irr_glsl_reflect(V, H, VdotH);
    s.L = m * L; // m must be an orthonormal matrix
    s.TdotL = L.x;
    s.BdotL = L.y;
    s.NdotL = L.z;
    s.TdotH = H.x;
    s.BdotH = H.y;
    s.NdotH = H.z;
    s.VdotH = VdotH;

    return s;
}

// refraction or reflection from microfacet
irr_glsl_BxDFSample irr_glsl_createBSDFSample(in bool _refract, in vec3 H, in vec3 V, in float VdotH, in float VdotH2, in mat3 m, in float eta)
{
    irr_glsl_BxDFSample s;

    vec3 L = irr_glsl_reflect_refract(_refract, V, H, VdotH, VdotH2, eta);
    s.L = m * L; // m must be an orthonormal matrix
    s.TdotL = L.x;
    s.BdotL = L.y;
    s.NdotL = L.z;
    s.TdotH = H.x;
    s.BdotH = H.y;
    s.NdotH = H.z;
    s.VdotH = VdotH;

    return s;
}

#if 0 // TODO: for pseudo-spectral rendering
vec3 irr_glsl_sampleWavelengthContributionForRefraction(out float pdf, out float electedEta, in vec3 eta, in float u, in vec3 luminosityContributionHint)
{
    return;
}
#endif


#include <irr/builtin/glsl/bxdf/common.glsl>

void irr_glsl_updateBxDFParams(out irr_glsl_BSDFIsotropicParams p, in irr_glsl_BxDFSample s, in irr_glsl_IsotropicViewSurfaceInteraction inter)
{
    p.NdotL = s.NdotL;
    p.NdotL_squared = p.NdotL * p.NdotL;
    p.NdotH = s.NdotH;
    p.VdotH = s.VdotH;
    p.L = s.L;
    p.VdotL = dot(p.L, inter.V.dir);

    // @Crisspl wouldn't NaN be better here (or just nothing, you might be giving compiler more credit that its due)?
    p.LplusV_rcpLen = irr_glsl_FLT_INF;
    p.invlenL2 = irr_glsl_FLT_INF;
}
void irr_glsl_updateBxDFParams(out irr_glsl_BSDFAnisotropicParams p, in irr_glsl_BxDFSample s, in irr_glsl_AnisotropicViewSurfaceInteraction inter)
{
    irr_glsl_updateBxDFParams(p.isotropic, s, inter.isotropic);

    p.TdotL = s.TdotL;
    p.TdotH = s.TdotH;
    p.BdotL = s.BdotL;
    p.BdotH = s.BdotH;
}



irr_glsl_BxDFSample irr_glsl_transmission_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction)
{
    irr_glsl_BxDFSample smpl;
    smpl.L = -interaction.isotropic.V.dir;
    
    return smpl;
}

float irr_glsl_transmission_cos_remainder_and_pdf(out float pdf, in irr_glsl_BxDFSample s)
{
	pdf = 1.0/0.0;
	return 1.0;
}

irr_glsl_BxDFSample irr_glsl_reflection_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction)
{
    irr_glsl_BxDFSample smpl;
    smpl.L = irr_glsl_reflect(interaction.isotropic.V.dir,interaction.isotropic.N,interaction.isotropic.NdotV);
    smpl.NdotH = 1.0; 

    return smpl;
}

float irr_glsl_reflection_cos_remainder_and_pdf(out float pdf, in irr_glsl_BxDFSample s)
{
	pdf = 1.0/0.0;
	return 1.0;
}

#endif
