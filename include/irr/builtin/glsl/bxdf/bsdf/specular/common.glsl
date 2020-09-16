#ifndef _IRR_BUILTIN_GLSL_BXDF_BSDF_SPECULAR_COMMON_INCLUDED_
#define _IRR_BUILTIN_GLSL_BXDF_BSDF_SPECULAR_COMMON_INCLUDED_

#include <irr/builtin/glsl/math/functions.glsl>

#include <irr/builtin/glsl/bxdf/common_samples.glsl>
#include <irr/builtin/glsl/bxdf/fresnel.glsl>

// assert(VdotHLdotH<0.0)
float irr_glsl_microfacet_transmission_relative_to_reflection_differential_factor(in float VdotH, in float LdotH, in float VdotHLdotH, in float orientedEta)
{
    const float den_sqrt = VdotH+orientedEta*LdotH;
    return -4.0*VdotHLdotH/(den_sqrt*den_sqrt);
}

float irr_glsl_microfacet_transmission_relative_to_reflection_differential_factor(in float VdotH, in float LdotH, in float orientedEta)
{
    return irr_glsl_microfacet_transmission_relative_to_reflection_differential_factor(VdotH,LdotH,VdotH*LdotH,orientedEta);
}


// assuming VNDF sampling followed by transmission selection according to the fresnel has been used we can compute the remainder for every subsurface model
float irr_glsl_VNDF_fresnel_sampled_BSDF_cos_remainder(in bool transmitted, in float G2_over_G1, in float transmission_relative_to_reflection_differential_factor)
{
    return G2_over_G1*(transmitted ? transmission_relative_to_reflection_differential_factor:1.0);
}

float irr_glsl_VNDF_fresnel_sampled_BRDF_pdf_to_BSDF_pdf(in bool transmitted, in float reflectance, in float vndf_sampling_pdf)
{
    return (transmitted ? (1.0-reflectance):reflectance)*vndf_sampling_pdf;
}

#endif