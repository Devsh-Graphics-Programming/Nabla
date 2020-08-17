#ifndef _IRR_BSDF_DIELECTRIC_INCLUDED_
#define _IRR_BSDF_DIELECTRIC_INCLUDED_

#include <irr/builtin/glsl/math/functions.glsl>
#include <irr/builtin/glsl/bxdf/common_samples.glsl>

irr_glsl_BSDFSample irr_glsl_thin_smooth_dielectric_cos_sample(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, vec3 u, in vec3 eta, out float throuhgput)
{
    float NdotV = interaction.isotropic.NdotV;
    vec3 Fr = irr_glsl_fresnel_dielectric_frontface_only(eta, abs(NdotV));
    float reflectionProb = Fr.r;//dot(Fr, luminosityContributionHint);

    //TODO: compute multiscatter, we need this!
    //Fr *= irr_glsl_fresnel_dielectric(eta, NdotV);

    const bool transmitted = irr_glsl_partitionRandVariable(reflectionProb, u.z, throuhgput);

    irr_glsl_BSDFSample smpl;
    smpl.L = (transmitted ? vec3(0.0):interaction.isotropic.N*2.0*NdotV)-interaction.isotropic.V.dir;
    return smpl;
}
irr_glsl_BSDFSample irr_glsl_thin_smooth_dielectric_cos_sample(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec3 u, in vec3 eta)
{
    float dummy;
    return irr_glsl_thin_smooth_dielectric_cos_sample(interaction, u, eta, dummy);
}

vec3 irr_glsl_thin_smooth_dielectric_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFSample _sample, in irr_glsl_IsotropicViewSurfaceInteraction interaction, in vec3 eta)
{
    pdf = 1.0 / 0.0; // should be reciprocal probability of the fresnel choice divided by 0.0, but still an INF.
    return vec3(1.0);
}
vec3 irr_glsl_thin_smooth_dielectric_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFSample _sample, in irr_glsl_IsotropicViewSurfaceInteraction interaction, in vec3 eta, in float throughput)
{
    return irr_glsl_thin_smooth_dielectric_cos_remainder_and_pdf(pdf,_sample,interaction,eta);
}

#if 0
//this is probably wrong so not touching it
// usually  `luminosityContributionHint` would be the Rec.709 luma coefficients (the Y row of the RGB to CIE XYZ matrix)
//assert(1.0==luminosityContributionHint.r+luminosityContributionHint.g+luminosityContributionHint.b);
#endif
vec3 wavelen;
irr_glsl_BSDFSample irr_glsl_smooth_dielectric_cos_sample(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, vec3 u, in vec3 eta, out float throuhgput)
{
    float NdotV = interaction.isotropic.NdotV;
    vec3 Fr = irr_glsl_fresnel_dielectric(eta, NdotV);
    float reflectionProb = dot(vec3(1.0/3.0),Fr);//dot(Fr, luminosityContributionHint);

    const bool refracted = irr_glsl_partitionRandVariable(reflectionProb, u.z, throuhgput);

    float _eta = u.z < 0.5 ? mix(eta.r, eta.g, u.z * 2.0) : mix(eta.g, eta.b, u.z * 2.0 - 1.0);
    wavelen = u.z<0.5 ? mix(vec3(4.0,0.0,0.0),vec3(0.0,2.0,0.0),u.z*2.0):mix(vec3(0.0,2.0,0.0),vec3(0.0,0.0,4.0),u.z*2.0-1.0);

    irr_glsl_BSDFSample smpl;
    smpl.L = irr_glsl_reflect_refract(refracted, interaction.isotropic.V.dir, interaction.isotropic.N, NdotV, NdotV*NdotV, _eta);
    return smpl;
}
irr_glsl_BSDFSample irr_glsl_smooth_dielectric_cos_sample(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec3 u, in vec3 eta)
{
    float dummy;
    return irr_glsl_smooth_dielectric_cos_sample(interaction,u,eta,dummy);
}

vec3 irr_glsl_smooth_dielectric_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFSample _sample, in irr_glsl_IsotropicViewSurfaceInteraction interaction, in vec3 eta)
{
    pdf = 1.0/0.0; // should be reciprocal probability of the fresnel choice divided by 0.0, but still an INF.
    vec3 retval = vec3(1.0);
    //retval[wavelen] = 3.0;
    return wavelen;
}
vec3 irr_glsl_smooth_dielectric_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFSample _sample, in irr_glsl_IsotropicViewSurfaceInteraction interaction, in vec3 eta, in float throughput)
{
    return irr_glsl_smooth_dielectric_cos_remainder_and_pdf(pdf,_sample,interaction,eta);
}


#endif
