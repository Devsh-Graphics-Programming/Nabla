// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_COLOR_SPACE_ENCODE_CIE_XYZ_INCLUDED_
#define _NBL_BUILTIN_GLSL_COLOR_SPACE_ENCODE_CIE_XYZ_INCLUDED_

const mat3 nbl_glsl_scRGBtoXYZ = mat3(
    vec3(0.4124564, 0.2126729, 0.0193339),
    vec3(0.3575761, 0.7151522, 0.1191920),
    vec3(0.1804375, 0.0721750, 0.9503041)
);

const mat3 nbl_glsl_sRGBtoXYZ = nbl_glsl_scRGBtoXYZ;

const mat3 nbl_glsl_BT709toXYZ = nbl_glsl_scRGBtoXYZ;


const mat3 nbl_glsl_Display_P3toXYZ = mat3(
    vec3(0.4865709, 0.2289746, 0.0000000),
    vec3(0.2656677, 0.6917385, 0.0451134),
    vec3(0.1982173, 0.0792869, 1.0439444)
);


const mat3 nbl_glsl_DCI_P3toXYZ = mat3(
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);


const mat3 nbl_glsl_BT2020toXYZ = mat3(
    vec3(0.6369580, 0.2627002, 0.0000000),
    vec3(0.1446169, 0.6779981, 0.0280727),
    vec3(0.1688810, 0.0593017, 1.0609851)
);

const mat3 nbl_glsl_HDR10_ST2084toXYZ = nbl_glsl_BT2020toXYZ;

const mat3 nbl_glsl_DOLBYIVISIONtoXYZ = nbl_glsl_BT2020toXYZ;

const mat3 nbl_glsl_HDR10_HLGtoXYZ = nbl_glsl_BT2020toXYZ;


const mat3 nbl_glsl_AdobeRGBtoXYZ = mat3(
    vec3(0.57667, 0.29734, 0.02703),
    vec3(0.18556, 0.62736, 0.07069),
    vec3(0.18823, 0.07529, 0.99134)
);


const mat3 nbl_glsl_ACES2065_1toXYZ = mat3(
    vec3(0.9525523959, 0.3439664498, 0.0000000000),
    vec3(0.0000000000, 0.7281660966, 0.0000000000),
    vec3(0.0000936786, -0.0721325464, 1.0088251844)
);


const mat3 nbl_glsl_ACEScctoXYZ = mat3(
    vec3(0.6624542, 0.2722287, -0.0055746),
    vec3(0.1340042, 0.6740818, 0.6740818),
    vec3(0.1561877, 0.0536895, 1.0103391)
);

const mat3 nbl_glsl_ACESccttoXYZ = nbl_glsl_ACEScctoXYZ;

#endif