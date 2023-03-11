
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_BXDF_NDF_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_NDF_COMMON_INCLUDED_


namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace ndf
{

// general path
float microfacet_to_light_measure_transform(in float NDFcos, in float absNdotV, in bool transmitted, in float VdotH, in float LdotH, in float VdotHLdotH, in float orientedEta)
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
float microfacet_to_light_measure_transform(in float NDFcos, in float maxNdotV)
{
    return 0.25*NDFcos/maxNdotV;
}


namespace ggx
{

// specialized factorizations for GGX
float microfacet_to_light_measure_transform(in float NDFcos_already_in_reflective_dL_measure, in float absNdotL, in bool transmitted, in float VdotH, in float LdotH, in float VdotHLdotH, in float orientedEta)
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
float microfacet_to_light_measure_transform(in float NDFcos_already_in_reflective_dL_measure, in float maxNdotL)
{
    return NDFcos_already_in_reflective_dL_measure*maxNdotL;
}

}

}
}
}
}


#endif