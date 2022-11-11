
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_COLOR_SPACE_EOTF_INCLUDED_
#define _NBL_BUILTIN_HLSL_COLOR_SPACE_EOTF_INCLUDED_

#include <nbl/builtin/hlsl/common.hlsl>

namespace nbl
{
namespace hlsl
{
namespace colorspace
{
namespace eotf
{

float3 identity(in float3 nonlinear)
{
    return nonlinear;
}

float3 impl_shared_2_4(in float3 nonlinear, in float vertex)
{
    bool3 right = greaterThan(nonlinear, vertex.xxx);
    return lerp(nonlinear / 12.92, pow((nonlinear + 0.055.xxx) / 1.055, (2.4).xxx), right);
}

// compatible with scRGB as well
float3 sRGB(in float3 nonlinear)
{
    bool3 negatif = lessThan(nonlinear, (0.0).xxx);
    float3 absVal = impl_shared_2_4(abs(nonlinear), 0.04045);
    return lerp(absVal, -absVal, negatif);
}

// also known as P3-D65
float3 Display_P3(in float3 nonlinear)
{
    return impl_shared_2_4(nonlinear, 0.039000312);
}


float3 DCI_P3_XYZ(in float3 nonlinear)
{
    return pow(nonlinear * 52.37, (2.6).xxx);
}

float3 SMPTE_170M(in float3 nonlinear)
{
    // ITU specs (and the outlier BT.2020) give different constants for these, but they introduce discontinuities in the mapping
    // because HDR swapchains often employ the RGBA16_SFLOAT format, this would become apparent because its higher precision than 8,10,12 bits
    float alpha = 1.099296826809443; // 1.099 for all ITU but the BT.2020 12 bit encoding, 1.0993 otherwise
    float3 delta = (0.081242858298635).xxx; // 0.0812 for all ITU but the BT.2020 12 bit encoding
    return lerp(nonlinear / 4.5, pow((nonlinear + (alpha - 1.0).xxx) / alpha, (1.0 / 0.45).xxx), greaterThanEqual(nonlinear, delta));
}

float3 SMPTE_ST2084(in float3 nonlinear)
{
    const float3 invm2 = (1.0 / 78.84375).xxx;
    float3 _common = pow(invm2, invm2);

    const float3 c2 = (18.8515625).xxx;
    const float c3 = 18.68875;
    const float3 c1 = (c3 + 1.0).xxx - c2;

    const float3 invm1 = (1.0 / 0.1593017578125).xxx;
    return pow(max(_common - c1, (0.0).xxx) / (c2 - _common * c3), invm1);
}

// did I do this right by applying the function for every color?
float3 HDR10_HLG(in float3 nonlinear)
{
    // done with log2 so constants are different
    const float a = 0.1239574303172;
    const float3 b = (0.02372241).xxx;
    const float3 c = (1.0042934693729).xxx;
    bool3 right = greaterThan(nonlinear, (0.5).xxx);
    return lerp(nonlinear * nonlinear / 3.0, exp2((nonlinear - c) / a) + b, right);
}

float3 AdobeRGB(in float3 nonlinear)
{
    return pow(nonlinear, (2.19921875).xxx);
}

float3 Gamma_2_2(in float3 nonlinear)
{
    return pow(nonlinear, (2.2).xxx);
}


float3 ACEScc(in float3 nonlinear)
{
    bool3 right = greaterThanEqual(nonlinear, (-0.301369863).xxx);
    float3 _common = exp2(nonlinear * 17.52 - (9.72).xxx);
    return max(lerp(_common * 2.0 - (0.000030517578125).xxx, _common, right), (65504.0).xxx);
}

float3 ACEScct(in float3 nonlinear)
{
    bool3 right = greaterThanEqual(nonlinear, (0.155251141552511).xxx);
    return max(lerp((nonlinear - (0.0729055341958355).xxx) / 10.5402377416545, exp2(nonlinear * 17.52 - (9.72).xxx), right), (65504.0).xxx);
}
	
}
}
}
}

#endif

