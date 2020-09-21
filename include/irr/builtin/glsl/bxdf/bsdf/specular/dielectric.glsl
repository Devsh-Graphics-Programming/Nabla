#ifndef _IRR_BSDF_DIELECTRIC_INCLUDED_
#define _IRR_BSDF_DIELECTRIC_INCLUDED_

#include <irr/builtin/glsl/math/functions.glsl>
#include <irr/builtin/glsl/bxdf/common.glsl>
#include <irr/builtin/glsl/bxdf/common_samples.glsl>
#include <irr/builtin/glsl/bxdf/fresnel.glsl>

// usually `luminosityContributionHint` would be the Rec.709 luma coefficients (the Y row of the RGB to CIE XYZ matrix)
// its basically a set of weights that determine 
// assert(1.0==luminosityContributionHint.r+luminosityContributionHint.g+luminosityContributionHint.b);
// `remainderMetadata` is a variable in which the generator function returns byproducts of sample generation that would otherwise have to be redundantly calculated in `remainder_and_pdf`
irr_glsl_LightSample irr_glsl_thin_smooth_dielectric_cos_generate_wo_clamps(in vec3 V, in vec3 T, in vec3 B, in vec3 N, in float NdotV, in float absNdotV, vec3 u, in vec3 eta2, in vec3 luminosityContributionHint, out vec3 remainderMetadata)
{
    // we will only ever intersect from the outside
    const vec3 reflectance = irr_glsl_thindielectric_infinite_scatter(irr_glsl_fresnel_dielectric_common(eta2,absNdotV));

    // we are only allowed one choice for the entire ray, so make the probability a weighted sum
    const float reflectionProb = dot(reflectance, luminosityContributionHint);

    float rcpChoiceProb;
    const bool transmitted = irr_glsl_partitionRandVariable(reflectionProb, u.z, rcpChoiceProb);
    remainderMetadata = (transmitted ? (vec3(1.0)-reflectance):reflectance)*rcpChoiceProb;
    
    const vec3 L = (transmitted ? vec3(0.0):N*2.0*NdotV)-V;
    return irr_glsl_createLightSample(L,dot(V,L),T,B,N);
}



irr_glsl_LightSample irr_glsl_thin_smooth_dielectric_cos_generate_wo_clamps(in vec3 V, in vec3 T, in vec3 B, in vec3 N, in float NdotV, in float absNdotV, vec3 u, in vec3 eta2, in vec3 luminosityContributionHint)
{
    vec3 dummy;
    return irr_glsl_thin_smooth_dielectric_cos_generate_wo_clamps(V,T,B,N,NdotV,absNdotV,u,eta2,luminosityContributionHint,dummy);
}

irr_glsl_LightSample irr_glsl_thin_smooth_dielectric_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, vec3 u, in vec3 eta2, in vec3 luminosityContributionHint)
{
    return irr_glsl_thin_smooth_dielectric_cos_generate_wo_clamps(interaction.isotropic.V.dir,interaction.T,interaction.B,interaction.isotropic.N,interaction.isotropic.NdotV,abs(interaction.isotropic.NdotV),u,eta2,luminosityContributionHint);
}



vec3 irr_glsl_thin_smooth_dielectric_cos_remainder_and_pdf_wo_clamps(out float pdf, in vec3 remainderMetadata)
{
    pdf = 1.0 / 0.0; // should be reciprocal probability of the fresnel choice divided by 0.0, but would still be an INF.
    return remainderMetadata;
}

vec3 irr_glsl_thin_smooth_dielectric_cos_remainder_and_pdf_wo_clamps(out float pdf, in bool transmitted, in float absNdotV, in vec3 eta2, in vec3 luminosityContributionHint)
{
    const vec3 reflectance = irr_glsl_thindielectric_infinite_scatter(irr_glsl_fresnel_dielectric_common(eta2,absNdotV));
    const vec3 sampleValue = transmitted ? (vec3(1.0)-reflectance):reflectance;

    const float sampleProb = dot(sampleValue,luminosityContributionHint);

    pdf = 1.0 / 0.0;
    return irr_glsl_thin_smooth_dielectric_cos_remainder_and_pdf_wo_clamps(pdf, sampleValue / sampleProb);
}

vec3 irr_glsl_thin_smooth_dielectric_cos_remainder_and_pdf(out float pdf, in irr_glsl_LightSample _sample, in irr_glsl_IsotropicViewSurfaceInteraction interaction, in vec3 eta2, in vec3 luminosityContributionHint)
{
    const bool transmitted = irr_glsl_isTransmissionPath(interaction.NdotV,_sample.NdotL);
    return irr_glsl_thin_smooth_dielectric_cos_remainder_and_pdf_wo_clamps(pdf,transmitted,abs(interaction.NdotV),eta2,luminosityContributionHint);
}




irr_glsl_LightSample irr_glsl_smooth_dielectric_cos_generate_wo_clamps(in vec3 V, in vec3 T, in vec3 B, in vec3 N, in bool backside, in float NdotV, in float absNdotV, in float NdotV2, vec3 u, in float rcpOrientedEta, in float orientedEta2, in float rcpOrientedEta2, out bool transmitted)
{
    const float reflectance = irr_glsl_fresnel_dielectric_common(orientedEta2,absNdotV);

    float rcpChoiceProb;
    transmitted = irr_glsl_partitionRandVariable(reflectance, u.z, rcpChoiceProb);

    const vec3 L = irr_glsl_reflect_refract(transmitted, V, N, backside, NdotV, NdotV2, rcpOrientedEta, rcpOrientedEta2);
    return irr_glsl_createLightSample(L,dot(V,L),T,B,N);
}

irr_glsl_LightSample irr_glsl_smooth_dielectric_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, vec3 u, in float eta)
{
    float orientedEta, rcpOrientedEta;
    const bool backside = irr_glsl_getOrientedEtas(orientedEta, rcpOrientedEta, interaction.isotropic.NdotV, eta);
    
    bool dummy;
    return irr_glsl_smooth_dielectric_cos_generate_wo_clamps(
        interaction.isotropic.V.dir,
        interaction.T,interaction.B,interaction.isotropic.N,
        backside,
        interaction.isotropic.NdotV,
        abs(interaction.isotropic.NdotV),
        interaction.isotropic.NdotV*interaction.isotropic.NdotV,
        u,
        rcpOrientedEta, orientedEta*orientedEta, rcpOrientedEta*rcpOrientedEta,
        dummy
    );
}


float irr_glsl_smooth_dielectric_cos_remainder_and_pdf(out float pdf, in bool transmitted, in float rcpOrientedEta2)
{
    pdf = 1.0 / 0.0; // should be reciprocal probability of the fresnel choice divided by 0.0, but would still be an INF.
    return transmitted ? rcpOrientedEta2:1.0;
}

float irr_glsl_smooth_dielectric_cos_remainder_and_pdf(out float pdf, in irr_glsl_LightSample _sample, in irr_glsl_IsotropicViewSurfaceInteraction interaction, in float eta)
{
    const bool transmitted = irr_glsl_isTransmissionPath(interaction.NdotV,_sample.NdotL);
    
    float dummy, rcpOrientedEta;
    const bool backside = irr_glsl_getOrientedEtas(dummy, rcpOrientedEta, interaction.NdotV, eta);

    return irr_glsl_smooth_dielectric_cos_remainder_and_pdf(pdf,transmitted,rcpOrientedEta);
}

#endif
