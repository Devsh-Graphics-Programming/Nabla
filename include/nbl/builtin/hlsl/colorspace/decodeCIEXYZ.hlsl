
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_COLOR_SPACE_DECODE_CIE_XYZ_INCLUDED_
#define _NBL_BUILTIN_HLSL_COLOR_SPACE_DECODE_CIE_XYZ_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/cpp_compat.hlsl>

namespace nbl
{
namespace hlsl
{
namespace colorspace
{
namespace decode
{

NBL_CONSTEXPR float3x3 XYZtoscRGB = float3x3(
    float3( 3.240970, -1.537383, -0.498611),
    float3(-0.969244,  1.875968,  0.041555),
    float3( 0.055630, -0.203977,  1.056972)
);


NBL_CONSTEXPR float3x3 XYZtosRGB = XYZtoscRGB;

NBL_CONSTEXPR float3x3 XYZtoBT709 = XYZtoscRGB;

  
NBL_CONSTEXPR float3x3 XYZtoDisplay_P3 = float3x3(
    float3( 2.4934969119, -0.9313836179, -0.4027107845),
    float3(-0.8294889696,  1.7626640603,  0.0236246858),
    float3( 0.0358458302, -0.0761723893,  0.9568845240)
);


NBL_CONSTEXPR float3x3 XYZtoDCI_P3 = float3x3(float3(1.0,0.0,0.0),float3(0.0,1.0,0.0),float3(0.0,0.0,1.0));

 
NBL_CONSTEXPR float3x3 XYZtoBT2020 = float3x3(  float3( 1.7166512, -0.3556708, -0.2533663), // TODO
                                        float3(-0.6666844,  1.6164812,  0.0157685),
                                        float3( 0.0176399, -0.0427706,  0.9421031));
 
NBL_CONSTEXPR float3x3 XYZtoHDR10_ST2084 = XYZtoBT2020;

NBL_CONSTEXPR float3x3 XYZtoDOLBYIVISION = XYZtoBT2020;

NBL_CONSTEXPR float3x3 XYZtoHDR10_HLG = XYZtoBT2020;


NBL_CONSTEXPR float3x3 XYZtoAdobeRGB = float3x3(    float3( 2.0415879038, -0.5650069743, -0.3447313508),
                                            float3(-0.9692436363, 1.8759675015, -0.0415550574),
                                            float3( 0.0134442806, -0.1183623922,  1.0151749944));


NBL_CONSTEXPR float3x3 XYZtoACES2065_1 = float3x3(  
    float3( 1.0498110175, 0.0000000000,  -0.0000974845),
    float3( -0.4959030231,  1.3733130458,  0.0982400361),
    float3(0.0000000000,  0.0000000000,  0.9912520182));


NBL_CONSTEXPR float3x3 XYZtoACEScc = float3x3(  
    float3( 1.6410233797, -0.3248032942, -0.2364246952),
    float3(-0.6636628587,  1.6153315917,  0.0167563477),
    float3( 0.0117218943, -0.0082844420,  0.9883948585));

NBL_CONSTEXPR float3x3 XYZtoACEScct = XYZtoACEScc;

}
}
}
}


#endif