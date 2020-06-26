#ifndef _IRR_BSDF_COMMON_SAMPLES_INCLUDED_
#define _IRR_BSDF_COMMON_SAMPLES_INCLUDED_

// do not use this struct in SSBO or UBO, its wasteful on memory
struct irr_glsl_BSDFSample
{
    vec3 L;  // incoming direction, normalized
    float LdotT;
    float LdotB;
    float LdotN;

    float TdotH;
    float BdotH;
    float NdotH;
    float VdotH;//equal to LdotH
};

// require H and V already be normalized
irr_glsl_BSDFSample irr_glsl_createBSDFSample(in vec3 H, in vec3 V, in float VdotH, in mat3 m)
{
    irr_glsl_BSDFSample s;

    vec3 L = irr_glsl_reflect(V, H, VdotH);
    s.L = m * L; // m must be an orthonormal matrix
    s.LdotT = L.x;
    s.LdotB = L.y;
    s.LdotN = L.z;
    s.TdotH = H.x;
    s.BdotH = H.y;
    s.NdotH = H.z;
    s.VdotH = VdotH;

    return s;
}


#include <irr/builtin/glsl/bxdf/common.glsl>


irr_glsl_BSDFSample irr_glsl_transmission_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction)
{
    irr_glsl_BSDFSample smpl;
    smpl.L = -interaction.isotropic.V.dir;
    
    return smpl;
}

vec3 irr_glsl_transmission_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFSample s, in irr_glsl_BSDFAnisotropicParams params, in irr_glsl_AnisotropicViewSurfaceInteraction interaction)
{
	pdf = 1.0/0.0;
	return vec3(1.0);
}

irr_glsl_BSDFSample irr_glsl_reflection_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction)
{
    irr_glsl_BSDFSample smpl;
    smpl.L = irr_glsl_reflect(interaction.isotropic.V.dir,interaction.isotropic.N,interaction.isotropic.NdotV);

    return smpl;
}

vec3 irr_glsl_reflection_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFSample s, in irr_glsl_BSDFAnisotropicParams params, in irr_glsl_AnisotropicViewSurfaceInteraction interaction)
{
	pdf = 1.0/0.0;
	return vec3(1.0);
}

// TODO Move to different header
#if 0
//this is probably wrong so not touching it
// usually  `luminosityContributionHint` would be the Rec.709 luma coefficients (the Y row of the RGB to CIE XYZ matrix)
//assert(1.0==luminosityContributionHint.r+luminosityContributionHint.g+luminosityContributionHint.b);
irr_glsl_BSDFSample irr_glsl_smooth_dielectric_cos_sample(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec2 u, in vec3 eta, in vec3 luminosityContributionHint)
{
    vec3 Fr = irr_glsl_fresnel_dielectric(eta, interaction.isotropic.NdotV);
    float reflectionProb = dot(Fr, luminosityContributionHint);//why dont we just use fresnel as reflection probability? i know its a vec3 but all its components should be equal in case of dielectric
    
    irr_glsl_BSDFSample smpl;
    if (reflectionProb==1.0 || u.x<reflectionProb)
    {
        smpl.L = interaction.isotropic.N*2.0*NdotV - interaction.isotropic.V.dir;
        //smpl.probability = reflectionProb;
    }
    else
    {
        //no idea whats going on here and whats `k` a few lines below
        float fittedMonochromeEta = dot(eta, luminosityContributionHint); //????
        //refract
        float NdotL2 = fittedMonochromeEta*fittedMonochromeEta - (1.0-NdotV2);
        /*if (k < 0.0)
            smpl.L = vec3(0.0);
        else*/
            smpl.L = ((NdotV /*+ sqrt(k)*/) * interaction.isotropic.N - V) / fittedMonochromeEta;
        //smpl.probability = 1.0-reflectionProb;
    }

    return smpl;
}
irr_glsl_BSDFSample irr_glsl_smooth_dielectric_cos_sample(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in uvec2 _u, in vec3 eta, in vec3 luminosityContributionHint)
{
    vec2 u = vec2(_u)/float(UINT_MAX);
    return irr_glsl_smooth_dielectric_cos_sample(interaction, u, eta, luminosityContributionHint);
}
#endif
/*
irr_glsl_BSDFSample irr_glsl_reflection_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec2 u, in vec3 eta, in vec3 luminosityContributionHint)
{
    irr_glsl_BSDFSample smpl;
    smpl.L = irr_glsl_reflect(interaction.isotropic.V.dir,interaction.isotropic.N,interaction.isotropic.NdotV);

    return smpl;
}

vec3 irr_glsl_smooth_dielectric_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFSample s, in irr_glsl_BSDFAnisotropicParams params, in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec3 eta, in vec3 luminosityContributionHint)
{
    pdf = 1.0/0.0; // its inf anyway
    return vec3(1.0);
}
*/


#endif
