#ifndef _IRR_BUILTIN_GLSL_BXDF_NDF_COMMON_INCLUDED_
#define _IRR_BUILTIN_GLSL_BXDF_NDF_COMMON_INCLUDED_

// general path
float irr_glsl_microfacet_to_light_measure_transform(in float NDFcos, in float absNdotV, in bool transmitted, in float VdotH, in float LdotH, in float VdotHLdotH, in float orientedEta)
{
    float denominator = absNdotV;
    if (transmitted)
    {
        const float VdotH_etaLdotH = (VdotH+orientedEta*LdotH);
        // VdotHLdotH is negative under transmission, so thats denominator is negative
        denominator *= -VdotH_etaLdotH*VdotH_etaLdotH;
    }
    return NDFcos*(transmitted ? VdotHLdotH:0.25)/denominator;
}
float irr_glsl_microfacet_to_light_measure_transform(in float NDFcos, in float maxNdotV)
{
    return 0.25*NDFcos/maxNdotV;
}

// specialized factorizations for GGX
float irr_glsl_ggx_microfacet_to_light_measure_transform(in float NDFcos_already_in_reflective_dL_measure, in float absNdotL, in bool transmitted, in float VdotH, in float LdotH, in float VdotHLdotH, in float orientedEta)
{
    float factor = absNdotL;
    if (transmitted)
    {
        const float VdotH_etaLdotH = (VdotH+orientedEta*LdotH);
        // VdotHLdotH is negative under transmission, so thats denominator is negative
        factor *= -4.0*VdotHLdotH/(VdotH_etaLdotH*VdotH_etaLdotH);
    }
    return NDFcos_already_in_reflective_dL_measure*factor;
}
float irr_glsl_microfacet_to_light_measure_transform(in float ndf, in float maxNdotL)
{
    return NDFcos_already_in_reflective_dL_measure*maxNdotL;
}

#endif