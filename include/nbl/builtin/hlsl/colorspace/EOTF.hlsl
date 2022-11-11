
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
namespace colorspace_eotf
{

float3 identity(in float3 nonlinear)
{
    return nonlinear;
}

float3 impl_shared_2_4(in float3 nonlinear, in float vertex)
{
    bool3 right = greaterThan(nonlinear, float3(vertex, vertex, vertex));
    return lerp(nonlinear / 12.92, pow((nonlinear + float3(0.055,0.055,0.055)) / 1.055, float3(2.4,2.4,2.4)), right);
}

// compatible with scRGB as well
float3 sRGB(in float3 nonlinear)
{
    bool3 negatif = lessThan(nonlinear, float3(0.0,0.0,0.0));
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
    return pow(nonlinear * 52.37, float3(2.6,2.6,2.6));
}

float3 SMPTE_170M(in float3 nonlinear)
{
    // ITU specs (and the outlier BT.2020) give different constants for these, but they introduce discontinuities in the mapping
    // because HDR swapchains often employ the RGBA16_SFLOAT format, this would become apparent because its higher precision than 8,10,12 bits
    float alpha = 1.099296826809443; // 1.099 for all ITU but the BT.2020 12 bit encoding, 1.0993 otherwise
    float3 delta = float3(0.081242858298635,
                          0.081242858298635,
                          0.081242858298635); // 0.0812 for all ITU but the BT.2020 12 bit encoding
    return lerp(nonlinear / 4.5, pow((nonlinear + float3(alpha-1.0,alpha-1.0,alpha-1.0)) / alpha, float3(1.0/0.45,1.0/0.45,1.0/0.45)), greaterThanEqual(nonlinear, delta));
}

float3 SMPTE_ST2084(in float3 nonlinear)
{
    const float3 invm2 = float3(1.0/78.84375,1.0/78.84375,1.0/78.84375);
    float3 _common = pow(invm2, invm2);

    const float3 c2 = float3(18.8515625,18.8515625,18.8515625);
    const float c3 = 18.68875;
    const float3 c1 = float3(c3+1.0,c3+1.0,c3+1.0) - c2;

    const float3 invm1 = float3(1.0/0.1593017578125,
                                1.0/0.1593017578125,
                                1.0/0.1593017578125);
    return pow(max(_common - c1, float3(0.0,0.0,0.0)) / (c2 - _common * c3), invm1);
}

// did I do this right by applying the function for every color?
float3 HDR10_HLG(in float3 nonlinear)
{
    // done with log2 so constants are different
    const float a = 0.1239574303172;
    const float3 b = float3(0.02372241,0.02372241,0.02372241);
    const float3 c = float3(1.0042934693729,1.0042934693729,1.0042934693729);
    bool3 right = greaterThan(nonlinear, float3(0.5,0.5,0.5));
    return lerp(nonlinear * nonlinear / 3.0, exp2((nonlinear - c) / a) + b, right);
}

float3 AdobeRGB(in float3 nonlinear)
{
    return pow(nonlinear, float3(2.19921875,2.19921875,2.19921875));
}

float3 Gamma_2_2(in float3 nonlinear)
{
    return pow(nonlinear, float3(2.2,2.2,2.2));
}


float3 ACEScc(in float3 nonlinear)
{
    bool3 right = greaterThanEqual(nonlinear, float3(-0.301369863,-0.301369863,-0.301369863));
    float3 _common = exp2(nonlinear * 17.52 - float3(9.72,9.72,9.72));
    return max(lerp(_common * 2.0 - float3(0.000030517578125,0.000030517578125,0.000030517578125), _common, right), float3(65504.0,65504.0,65504.0));
}

float3 ACEScct(in float3 nonlinear)
{
    bool3 right = greaterThanEqual(nonlinear, float3(0.155251141552511,0.155251141552511,0.155251141552511));
    return max(lerp((nonlinear - float3(0.0729055341958355,0.0729055341958355,0.0729055341958355)) / 10.5402377416545, exp2(nonlinear * 17.52 - float3(9.72,9.72,9.72)), right), float3(65504.0,65504.0,65504.0));
}
	
}
}
}

#endif

