#ifndef _IRR_BSDF_COMMON_SAMPLES_INCLUDED_
#define _IRR_BSDF_COMMON_SAMPLES_INCLUDED_

#include <irr/builtin/glsl/bxdf/brdf/cos_weighted_sample.glsl>

irr_glsl_BSDFSample irr_glsl_transmission_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction)
{
    irr_glsl_BSDFSample smpl;
    smpl.L = -interaction.isotropic.V.dir;
    //smpl.probability = 1.0;
    
    return smpl;
}

irr_glsl_transmission_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFAnisotropicParams params, in irr_glsl_AnisotropicViewSurfaceInteraction interaction)
{
	pdf = 1.0;
	return 0.0;
}

irr_glsl_BSDFSample irr_glsl_delta_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction)
{
    irr_glsl_BSDFSample smpl;
    smpl.L = interaction.isotropic.N*2.0*interaction.isotropic.NdotV - interaction.isotropic.V.dir;
    //smpl.probability = 1.0;

    return smpl;
}

irr_glsl_delta_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFAnisotropicParams params, in irr_glsl_AnisotropicViewSurfaceInteraction interaction)
{
	pdf = 1.0;
	return 0.0;
}

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
