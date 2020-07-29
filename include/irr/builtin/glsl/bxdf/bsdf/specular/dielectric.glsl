#ifndef _IRR_BSDF_DIELECTRIC_INCLUDED_
#define _IRR_BSDF_DIELECTRIC_INCLUDED_

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
#endif
irr_glsl_BSDFSample irr_glsl_smooth_dielectric_cos_sample(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, vec3 u, in vec3 eta, out float throuhgput)
{
    float NdotV = interaction.isotropic.NdotV;
    vec3 Fr = irr_glsl_fresnel_dielectric(eta,NdotV);
    float reflectionProb = Fr.r;//dot(Fr, luminosityContributionHint);
    
    const bool reflected = irr_glsl_partitionRandVariable(reflectionProb,u.z,throuhgput);

    irr_glsl_BSDFSample smpl;
    vec3 V = interaction.isotropic.V.dir;
    vec3 N = interaction.isotropic.N;
    if (reflected)
        smpl.L = irr_glsl_reflect(V,-N,-NdotV);
    else
        smpl.L = irr_glsl_refract(V,N,NdotV,eta.r);

    return smpl;
}
irr_glsl_BSDFSample irr_glsl_smooth_dielectric_cos_sample(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec3 u, in vec3 eta)
{
    float dummy;
    return irr_glsl_smooth_dielectric_cos_sample(interaction,u,eta,dummy);
}

vec3 irr_glsl_smooth_dielectric_cos_remainder(out float pdf, in irr_glsl_BSDFSample _sample, in irr_glsl_IsotropicViewSurfaceInteraction interaction, in vec3 eta)
{
    pdf = 1.0/0.0; // should be reciprocal probability of the fresnel choice divided by 0.0, but still an INF.
    return vec3(1.0);
}


#endif
