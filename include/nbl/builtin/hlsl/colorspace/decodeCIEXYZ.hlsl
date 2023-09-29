
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
    float3( 3.240970f, -1.537383f, -0.498611f),
    float3(-0.969244f,  1.875968f,  0.041555f),
    float3( 0.055630f, -0.203977f,  1.056972f)
);

NBL_CONSTEXPR float3x3 XYZtosRGB = XYZtoscRGB;
NBL_CONSTEXPR float3x3 XYZtoBT709 = XYZtoscRGB;

  
NBL_CONSTEXPR float3x3 XYZtoDisplay_P3 = float3x3(
    float3( 2.4934969119f, -0.9313836179f, -0.4027107845f),
    float3(-0.8294889696f,  1.7626640603f,  0.0236246858f),
    float3( 0.0358458302f, -0.0761723893f,  0.9568845240f)
);

NBL_CONSTEXPR float3x3 XYZtoDCI_P3 = float3x3(
    float3(1.0f, 0.0f, 0.0f),
    float3(0.0f, 1.0f, 0.0f),
    float3(0.0f, 0.0f, 1.0f)
);

NBL_CONSTEXPR float3x3 XYZtoBT2020 = float3x3(  
    float3( 1.716651f, -0.355671f, -0.253366f),
    float3(-0.666684f,  1.616481f,  0.015769f),
    float3( 0.017640f, -0.042771f,  0.942103f)
);
 
NBL_CONSTEXPR float3x3 XYZtoHDR10_ST2084 = XYZtoBT2020;
NBL_CONSTEXPR float3x3 XYZtoDOLBYIVISION = XYZtoBT2020;
NBL_CONSTEXPR float3x3 XYZtoHDR10_HLG = XYZtoBT2020;

NBL_CONSTEXPR float3x3 XYZtoAdobeRGB = float3x3(
    float3( 2.0415879038f, -0.5650069743f, -0.3447313508f),
    float3(-0.9692436363f,  1.8759675015f,  0.0415550574f),
    float3( 0.0134442806f, -0.1183623922f,  1.0151749944f)
);


NBL_CONSTEXPR float3x3 XYZtoACES2065_1 = float3x3(  
    float3( 1.0498110175f, 0.0000000000f, -0.0000974845f),
    float3(-0.4959030231f, 1.3733130458f,  0.0982400361f),
    float3( 0.0000000000f, 0.0000000000f,  0.9912520182f)
);

NBL_CONSTEXPR float3x3 XYZtoACEScc = float3x3(  
    float3( 1.6410233797f, -0.3248032942f, -0.2364246952f),
    float3(-0.6636628587f,  1.6153315917f,  0.0167563477f),
    float3( 0.0117218943f, -0.0082844420f,  0.9883948585f)
);

NBL_CONSTEXPR float3x3 XYZtoACEScct = XYZtoACEScc;

}
}
}
}


#endif