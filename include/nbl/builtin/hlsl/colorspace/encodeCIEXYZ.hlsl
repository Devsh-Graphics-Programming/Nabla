

// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_COLOR_SPACE_ENCODE_CIE_XYZ_INCLUDED_
#define _NBL_BUILTIN_HLSL_COLOR_SPACE_ENCODE_CIE_XYZ_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/cpp_compat.hlsl>

namespace nbl
{
namespace hlsl
{
namespace colorspace
{

NBL_CONSTEXPR float3x3 scRGBtoXYZ = float3x3(
    float3(0.412391, 0.357584, 0.180481),
    float3(0.212639, 0.715169, 0.072192),
    float3(0.019331, 0.119195, 0.950532)
);

NBL_CONSTEXPR float3x3 sRGBtoXYZ = scRGBtoXYZ;

NBL_CONSTEXPR float3x3 BT709toXYZ = scRGBtoXYZ;


NBL_CONSTEXPR float3x3 Display_P3toXYZ = float3x3(
    float3(0.4865709486, 0.2656676932, 0.1982172852),
    float3(0.2289745641, 0.6917385218, 0.0792869141),
    float3(0.0000000, 0.0451133819, 1.0439443689)
);


NBL_CONSTEXPR float3x3 DCI_P3toXYZ = float3x3(
    float3(1.0, 0.0, 0.0),
    float3(0.0, 1.0, 0.0),
    float3(0.0, 0.0, 1.0)
);


NBL_CONSTEXPR float3x3 BT2020toXYZ = float3x3( // TODO
    float3(0.6369580, 0.1446169, 0.1688810),
    float3(0.2627002, 0.6779981, 0.0593017),
    float3(0.0000000, 0.0280727, 1.0609851)
);

NBL_CONSTEXPR float3x3 HDR10_ST2084toXYZ = BT2020toXYZ;

NBL_CONSTEXPR float3x3 DOLBYIVISIONtoXYZ = BT2020toXYZ;

NBL_CONSTEXPR float3x3 HDR10_HLGtoXYZ = BT2020toXYZ;


NBL_CONSTEXPR float3x3 AdobeRGBtoXYZ = float3x3(
    float3(0.5766690429, 0.1855582379, 0.1882286462),
    float3(0.2973449753, 0.6273635663, 0.0752914585),
    float3(0.0270313614, 0.0706888525, 0.9913375368)
);


NBL_CONSTEXPR float3x3 ACES2065_1toXYZ = float3x3(
    float3(0.9525523959, 0.0000000000,  0.0000936786),
    float3(0.3439664498, 0.7281660966, -0.0721325464),
    float3(0.0000000000, 0.0000000000,  1.0088251844)
);


NBL_CONSTEXPR float3x3 ACEScctoXYZ = float3x3(
    float3( 0.6624541811, 0.1340042065, 0.1561876870),
    float3( 0.2722287168, 0.6740817658, 0.0536895174),
    float3(-0.0055746495, 0.0040607335, 1.0103391003)
);

NBL_CONSTEXPR float3x3 ACESccttoXYZ = ACEScctoXYZ;

}
}
}

#endif