

// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_COLOR_SPACE_ENCODE_CIE_XYZ_INCLUDED_
#define _NBL_BUILTIN_HLSL_COLOR_SPACE_ENCODE_CIE_XYZ_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

namespace nbl
{
namespace hlsl
{
namespace colorspace
{

NBL_CONSTEXPR float3x3 scRGBtoXYZ = float3x3(
    float3(0.412391f, 0.357584f, 0.180481f),
    float3(0.212639f, 0.715169f, 0.072192f),
    float3(0.019331f, 0.119195f, 0.950532f)
);

NBL_CONSTEXPR float3x3 sRGBtoXYZ = scRGBtoXYZ;

NBL_CONSTEXPR float3x3 BT709toXYZ = scRGBtoXYZ;


NBL_CONSTEXPR float3x3 Display_P3toXYZ = float3x3(
    float3(0.4865709486f, 0.2656676932f, 0.1982172852f),
    float3(0.2289745641f, 0.6917385218f, 0.0792869141f),
    float3(0.0000000000f, 0.0451133819f, 1.0439443689f)
);


NBL_CONSTEXPR float3x3 DCI_P3toXYZ = float3x3(
    float3(1.0f, 0.0f, 0.0f),
    float3(0.0f, 1.0f, 0.0f),
    float3(0.0f, 0.0f, 1.0f)
);


NBL_CONSTEXPR float3x3 BT2020toXYZ = float3x3(
    float3(0.636958f, 0.144617f, 0.168881f),
    float3(0.262700f, 0.677998f, 0.059302f),
    float3(0.000000f, 0.028073f, 1.060985f)
);

NBL_CONSTEXPR float3x3 HDR10_ST2084toXYZ = BT2020toXYZ;

NBL_CONSTEXPR float3x3 DOLBYIVISIONtoXYZ = BT2020toXYZ;

NBL_CONSTEXPR float3x3 HDR10_HLGtoXYZ = BT2020toXYZ;


NBL_CONSTEXPR float3x3 AdobeRGBtoXYZ = float3x3(
    float3(0.5766690429f, 0.1855582379f, 0.1882286462f),
    float3(0.2973449753f, 0.6273635663f, 0.0752914585f),
    float3(0.0270313614f, 0.0706888525f, 0.9913375368f)
);


NBL_CONSTEXPR float3x3 ACES2065_1toXYZ = float3x3(
    float3(0.9525523959f, 0.0000000000f,  0.0000936786f),
    float3(0.3439664498f, 0.7281660966f, -0.0721325464f),
    float3(0.0000000000f, 0.0000000000f,  1.0088251844f)
);


NBL_CONSTEXPR float3x3 ACEScctoXYZ = float3x3(
    float3( 0.6624541811f, 0.1340042065f, 0.1561876870f),
    float3( 0.2722287168f, 0.6740817658f, 0.0536895174f),
    float3(-0.0055746495f, 0.0040607335f, 1.0103391003f)
);

NBL_CONSTEXPR float3x3 ACESccttoXYZ = ACEScctoXYZ;

}
}
}

#endif