

// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_COLOR_SPACE_ENCODE_CIE_XYZ_INCLUDED_
#define _NBL_BUILTIN_HLSL_COLOR_SPACE_ENCODE_CIE_XYZ_INCLUDED_

namespace nbl
{
namespace hlsl
{
namespace colorspace
{

static const float3x3 scRGBtoXYZ = float3x3(
    float3(0.4124564, 0.3575761, 0.1804375),
    float3(0.2126729, 0.7151522, 0.0721750),
    float3(0.0193339, 0.1191920, 0.9503041)
);

static const float3x3 sRGBtoXYZ = scRGBtoXYZ;

static const float3x3 BT709toXYZ = scRGBtoXYZ;


static const float3x3 Display_P3toXYZ = float3x3(
    float3(0.4865709, 0.2656677, 0.1982173),
    float3(0.2289746, 0.6917385, 0.0792869),
    float3(0.0000000, 0.0451134, 1.0439444)
);


static const float3x3 DCI_P3toXYZ = float3x3(
    float3(1.0, 0.0, 0.0),
    float3(0.0, 1.0, 0.0),
    float3(0.0, 0.0, 1.0)
);


static const float3x3 BT2020toXYZ = float3x3(
    float3(0.6369580, 0.1446169, 0.1688810),
    float3(0.2627002, 0.6779981, 0.0593017),
    float3(0.0000000, 0.0280727, 1.0609851)
);

static const float3x3 HDR10_ST2084toXYZ = BT2020toXYZ;

static const float3x3 DOLBYIVISIONtoXYZ = BT2020toXYZ;

static const float3x3 HDR10_HLGtoXYZ = BT2020toXYZ;


static const float3x3 AdobeRGBtoXYZ = float3x3(
    float3(0.57667, 0.18556, 0.18823),
    float3(0.29734, 0.62736, 0.07529),
    float3(0.02703, 0.07069, 0.99134)
);


static const float3x3 ACES2065_1toXYZ = float3x3(
    float3(0.9525523959, 0.0000000000,  0.0000936786),
    float3(0.3439664498, 0.7281660966, -0.0721325464),
    float3(0.0000000000, 0.0000000000,  1.0088251844)
);


static const float3x3 ACEScctoXYZ = float3x3(
    float3( 0.6624542, 0.1340042, 0.1561877),
    float3( 0.2722287, 0.6740818, 0.0536895),
    float3(-0.0055746, 0.6740818, 1.0103391)
);

static const float3x3 ACESccttoXYZ = ACEScctoXYZ;

}
}
}

#endif