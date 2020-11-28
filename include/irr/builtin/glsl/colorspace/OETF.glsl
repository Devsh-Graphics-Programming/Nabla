// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_COLOR_SPACE_OETF_INCLUDED_
#define _NBL_BUILTIN_GLSL_COLOR_SPACE_OETF_INCLUDED_

vec3 nbl_glsl_oetf_identity(in vec3 linear)
{
    return linear;
}

vec3 nbl_glsl_oetf_impl_shared_2_4(in vec3 linear, in float vertex)
{
    bvec3 right = greaterThan(linear, vec3(vertex));
    return mix(linear * 12.92, pow(linear, vec3(1.0 / 2.4)) * 1.055 - vec3(0.055), right);
}

// compatible with scRGB as well
vec3 nbl_glsl_oetf_sRGB(in vec3 linear)
{
    bvec3 negatif = lessThan(linear, vec3(0.0));
    vec3 absVal = nbl_glsl_oetf_impl_shared_2_4(abs(linear), 0.0031308);
    return mix(absVal, -absVal, negatif);
}

// also known as P3-D65
vec3 nbl_glsl_oetf_Display_P3(in vec3 linear)
{
    return nbl_glsl_oetf_impl_shared_2_4(linear, 0.0030186);
}

vec3 nbl_glsl_oetf_DCI_P3_XYZ(in vec3 linear)
{
    return pow(linear / 52.37, vec3(1.0 / 2.6));
}

vec3 nbl_glsl_oetf_SMPTE_170M(in vec3 linear)
{
    // ITU specs (and the outlier BT.2020) give different constants for these, but they introduce discontinuities in the mapping
    // because HDR swapchains often employ the RGBA16_SFLOAT format, this would become apparent because its higher precision than 8,10,12 bits
    const float alpha = 1.099296826809443; // 1.099 for all ITU but the BT.2020 12 bit encoding, 1.0993 otherwise
    const vec3 beta = vec3(0.018053968510808); // 0.0181 for all ITU but the BT.2020 12 bit encoding, 0.18 otherwise
    return mix(linear * 4.5, pow(linear, vec3(0.45)) * alpha - vec3(alpha - 1.0), greaterThanEqual(linear, beta));
}

vec3 nbl_glsl_oetf_SMPTE_ST2084(in vec3 linear)
{
    const vec3 m1 = vec3(0.1593017578125);
    const vec3 m2 = vec3(78.84375);
    const float c2 = 18.8515625;
    const float c3 = 18.68875;
    const vec3 c1 = vec3(c3 - c2 + 1.0);

    vec3 L_m1 = pow(linear, m1);
    return pow((c1 + L_m1 * c2) / (vec3(1.0) + L_m1 * c3), m2);
}

// did I do this right by applying the function for every color?
vec3 nbl_glsl_oetf_HDR10_HLG(in vec3 linear)
{
    // done with log2 so constants are different
    const float a = 0.1239574303172;
    const vec3 b = vec3(0.02372241);
    const vec3 c = vec3(1.0042934693729);
    bvec3 right = greaterThan(linear, vec3(1.0 / 12.0));
    return mix(sqrt(linear * 3.0), log2(linear - b) * a + c, right);
}

vec3 nbl_glsl_oetf_AdobeRGB(in vec3 linear)
{
    return pow(linear, vec3(1.0 / 2.19921875));
}

vec3 nbl_glsl_oetf_Gamma_2_2(in vec3 linear)
{
    return pow(linear, vec3(1.0 / 2.2));
}

vec3 nbl_glsl_oetf_ACEScc(in vec3 linear)
{
    bvec3 mid = greaterThanEqual(linear, vec3(0.0));
    bvec3 right = greaterThanEqual(linear, vec3(0.000030517578125));
    return (log2(mix(vec3(0.0000152587890625), vec3(0.0), right) + linear * mix(vec3(0.0), mix(vec3(0.5), vec3(1.0), right), mid)) + vec3(9.72)) / 17.52;
}

vec3 nbl_glsl_oetf_ACEScct(in vec3 linear)
{
    bvec3 right = greaterThan(linear, vec3(0.0078125));
    return mix(10.5402377416545 * linear + 0.0729055341958355, (log2(linear) + vec3(9.72)) / 17.52, right);
}
#endif