// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_COLOR_SPACE_EOTF_INCLUDED_
#define _NBL_BUILTIN_GLSL_COLOR_SPACE_EOTF_INCLUDED_

vec3 nbl_glsl_eotf_identity(in vec3 nonlinear)
{
    return nonlinear;
}

vec3 nbl_glsl_eotf_impl_shared_2_4(in vec3 nonlinear, in float vertex)
{
    bvec3 right = greaterThan(nonlinear, vec3(vertex));
    return mix(nonlinear / 12.92, pow((nonlinear + vec3(0.055)) / 1.055, vec3(2.4)), right);
}

// compatible with scRGB as well
vec3 nbl_glsl_eotf_sRGB(in vec3 nonlinear)
{
    bvec3 negatif = lessThan(nonlinear, vec3(0.0));
    vec3 absVal = nbl_glsl_eotf_impl_shared_2_4(abs(nonlinear), 0.04045);
    return mix(absVal, -absVal, negatif);
}

// also known as P3-D65
vec3 nbl_glsl_eotf_Display_P3(in vec3 nonlinear)
{
    return nbl_glsl_eotf_impl_shared_2_4(nonlinear, 0.039000312);
}


vec3 nbl_glsl_eotf_DCI_P3_XYZ(in vec3 nonlinear)
{
    return pow(nonlinear * 52.37, vec3(2.6));
}

vec3 nbl_glsl_eotf_SMPTE_170M(in vec3 nonlinear)
{
    // ITU specs (and the outlier BT.2020) give different constants for these, but they introduce discontinuities in the mapping
    // because HDR swapchains often employ the RGBA16_SFLOAT format, this would become apparent because its higher precision than 8,10,12 bits
    float alpha = 1.099296826809443; // 1.099 for all ITU but the BT.2020 12 bit encoding, 1.0993 otherwise
    vec3 delta = vec3(0.081242858298635); // 0.0812 for all ITU but the BT.2020 12 bit encoding
    return mix(nonlinear / 4.5, pow((nonlinear + vec3(alpha - 1.0)) / alpha, vec3(1.0 / 0.45)), greaterThanEqual(nonlinear, delta));
}

vec3 nbl_glsl_eotf_SMPTE_ST2084(in vec3 nonlinear)
{
    const vec3 invm2 = vec3(1.0 / 78.84375);
    vec3 _common = pow(invm2, invm2);

    const vec3 c2 = vec3(18.8515625);
    const float c3 = 18.68875;
    const vec3 c1 = vec3(c3 + 1.0) - c2;

    const vec3 invm1 = vec3(1.0 / 0.1593017578125);
    return pow(max(_common - c1, vec3(0.0)) / (c2 - _common * c3), invm1);
}

// did I do this right by applying the function for every color?
vec3 nbl_glsl_eotf_HDR10_HLG(in vec3 nonlinear)
{
    // done with log2 so constants are different
    const float a = 0.1239574303172;
    const vec3 b = vec3(0.02372241);
    const vec3 c = vec3(1.0042934693729);
    bvec3 right = greaterThan(nonlinear, vec3(0.5));
    return mix(nonlinear * nonlinear / 3.0, exp2((nonlinear - c) / a) + b, right);
}

vec3 nbl_glsl_eotf_AdobeRGB(in vec3 nonlinear)
{
    return pow(nonlinear, vec3(2.19921875));
}

vec3 nbl_glsl_eotf_Gamma_2_2(in vec3 nonlinear)
{
    return pow(nonlinear, vec3(2.2));
}


vec3 nbl_glsl_eotf_ACEScc(in vec3 nonlinear)
{
    bvec3 right = greaterThanEqual(nonlinear, vec3(-0.301369863));
    vec3 _common = exp2(nonlinear * 17.52 - vec3(9.72));
    return max(mix(_common * 2.0 - vec3(0.000030517578125), _common, right), vec3(65504.0));
}

vec3 nbl_glsl_eotf_ACEScct(in vec3 nonlinear)
{
    bvec3 right = greaterThanEqual(nonlinear, vec3(0.155251141552511));
    return max(mix((nonlinear - vec3(0.0729055341958355)) / 10.5402377416545, exp2(nonlinear * 17.52 - vec3(9.72)), right), vec3(65504.0));
}

#endif