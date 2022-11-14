
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_COLOR_SPACE_OETF_INCLUDED_
#define _NBL_BUILTIN_HLSL_COLOR_SPACE_OETF_INCLUDED_

#include <nbl/builtin/hlsl/common.hlsl>

namespace nbl
{
namespace hlsl
{
namespace colorspace
{
namespace oetf
{

float3 identity(in float3 _linear)
{
    return _linear;
}

float3 impl_shared_2_4(in float3 _linear, in float vertex)
{
    bool3 right = greaterThan(_linear, vertex.xxx);
    return lerp(_linear * 12.92, pow(_linear, (1.0 / 2.4).xxx) * 1.055 - (0.055).xxx, right);
}

// compatible with scRGB as well
float3 sRGB(in float3 _linear)
{
    bool3 negatif = lessThan(_linear, (0.0).xxx);
    float3 absVal = impl_shared_2_4(abs(_linear), 0.0031308);
    return lerp(absVal, -absVal, negatif);
}

// also known as P3-D65
float3 Display_P3(in float3 _linear)
{
    return impl_shared_2_4(_linear, 0.0030186);
}

float3 DCI_P3_XYZ(in float3 _linear)
{
    return pow(_linear / 52.37, (1.0 / 2.6).xxx);
}

float3 SMPTE_170M(in float3 _linear)
{
    // ITU specs (and the outlier BT.2020) give different constants for these, but they introduce discontinuities in the mapping
    // because HDR swapchains often employ the RGBA16_SFLOAT format, this would become apparent because its higher precision than 8,10,12 bits
    const float alpha = 1.099296826809443; // 1.099 for all ITU but the BT.2020 12 bit encoding, 1.0993 otherwise
    const float3 beta = (0.018053968510808).xxx; // 0.0181 for all ITU but the BT.2020 12 bit encoding, 0.18 otherwise
    return lerp(_linear * 4.5, pow(_linear, (0.45).xxx) * alpha - (alpha - 1.0).xxx, greaterThanEqual(_linear, beta));
}

float3 SMPTE_ST2084(in float3 _linear)
{
    const float3 m1 = (0.1593017578125).xxx;
    const float3 m2 = (78.84375).xxx;
    const float c2 = 18.8515625;
    const float c3 = 18.68875;
    const float3 c1 = (c3 - c2 + 1.0).xxx;

    float3 L_m1 = pow(_linear, m1);
    return pow((c1 + L_m1 * c2) / ((1.0).xxx + L_m1 * c3), m2);
}

// did I do this right by applying the function for every color?
float3 HDR10_HLG(in float3 _linear)
{
    // done with log2 so constants are different
    const float a = 0.1239574303172;
    const float3 b = (0.02372241).xxx;
    const float3 c = (1.0042934693729).xxx;
    bool3 right = greaterThan(_linear, (1.0 / 12.0).xxx);
    return lerp(sqrt(_linear * 3.0), log2(_linear - b) * a + c, right);
}

float3 AdobeRGB(in float3 _linear)
{
    return pow(_linear, (1.0 / 2.19921875).xxx);
}

float3 Gamma_2_2(in float3 _linear)
{
    return pow(_linear, (1.0 / 2.2).xxx);
}

float3 ACEScc(in float3 _linear)
{
    bool3 mid = greaterThanEqual(_linear, (0.0).xxx);
    bool3 right = greaterThanEqual(_linear, (0.000030517578125).xxx);
    return (log2(lerp((0.0000152587890625).xxx, (0.0).xxx, right) + _linear * lerp((0.0).xxx, lerp((0.5).xxx, (1.0).xxx, right), mid)) + (9.72).xxx) / 17.52;
}

float3 ACEScct(in float3 _linear)
{
    bool3 right = greaterThan(_linear, (0.0078125).xxx);
    return lerp(10.5402377416545 * _linear + 0.0729055341958355, (log2(_linear) + (9.72).xxx) / 17.52, right);
}

}
}
}
}

#endif