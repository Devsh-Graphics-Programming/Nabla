
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_COLOR_SPACE_DECODE_CIE_XYZ_INCLUDED_
#define _NBL_BUILTIN_HLSL_COLOR_SPACE_DECODE_CIE_XYZ_INCLUDED_

namespace nbl
{
namespace hlsl
{
namespace colorspace
{
namespace decode
{

const float3x3 XYZtoscRGB = float3x3(   float3( 3.2404542, -1.5371385, -0.4985314),
										float3(-0.9692660,  1.8760108,  0.0415560),
										float3( 0.0556434, -0.2040259,  1.0572252));


const float3x3 XYZtosRGB = XYZtoscRGB;

const float3x3 XYZtoBT709 = XYZtoscRGB;

  
const float3x3 XYZtoDisplay_P3 = float3x3(  float3( 2.4934969, -0.9313836, -0.4027108),
                                            float3(-0.8294890,  1.7626641,  0.0236247),
                                            float3( 0.0358458, -0.0761724,  0.9568845));


const float3x3 XYZtoDCI_P3 = float3x3(float3(1.0,0.0,0.0),float3(0.0,1.0,0.0),float3(0.0,0.0,1.0));

 
const float3x3 XYZtoBT2020 = float3x3(  float3( 1.7166512, -0.3556708, -0.2533663),
                                        float3(-0.6666844,  1.6164812,  0.0157685),
                                        float3( 0.0176399, -0.0427706,  0.9421031));
 
const float3x3 XYZtoHDR10_ST2084 = XYZtoBT2020;

const float3x3 XYZtoDOLBYIVISION = XYZtoBT2020;

const float3x3 XYZtoHDR10_HLG = XYZtoBT2020;


const float3x3 XYZtoAdobeRGB = float3x3(    float3( 2.04159, -0.56501, -0.34473),
                                            float3(-0.96924,  1.87597, -0.04156),
                                            float3( 0.01344, -0.11836,  1.01517));


const float3x3 XYZtoACES2065_1 = float3x3(  float3( 1.0498110175, -0.4959030231,  0.0000000000),
                                            float3( 0.0000000000,  1.3733130458,  0.0000000000),
                                            float3(-0.0000974845,  0.0982400361,  0.9912520182));


const float3x3 XYZtoACEScc = float3x3(  float3( 1.6410234, -0.3248033, -0.2364247),
                                        float3(-0.6636629,  1.6153316,  0.0167563),
                                        float3( 0.0117219, -0.0082844,  0.9883949));

const float3x3 XYZtoACEScct = XYZtoACEScc;

}
}
}
}


#endif