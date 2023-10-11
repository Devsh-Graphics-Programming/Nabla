// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_COLOR_SPACE_ENCODE_CIE_XYZ_INCLUDED_
#define _NBL_BUILTIN_GLSL_COLOR_SPACE_ENCODE_CIE_XYZ_INCLUDED_

const mat3 nbl_glsl_scRGBtoXYZ = mat3(
    vec3(0.412391f, 0.212639f, 0.019331f),
    vec3(0.357584f, 0.715169f, 0.119195f),
    vec3(0.180481f, 0.072192f, 0.950532f)
);

const mat3 nbl_glsl_sRGBtoXYZ = nbl_glsl_scRGBtoXYZ;

const mat3 nbl_glsl_BT709toXYZ = nbl_glsl_scRGBtoXYZ;


const mat3 nbl_glsl_Display_P3toXYZ = mat3(
    vec3(0.4865709486f, 0.2289745641f, 0.0000000000f),
    vec3(0.2656676932f, 0.6917385218f, 0.0451133819f),
    vec3(0.1982172852f, 0.0792869141f, 1.0439443689f)
);


const mat3 nbl_glsl_DCI_P3toXYZ = mat3(
    vec3(1.0f, 0.0f, 0.0f),
    vec3(0.0f, 1.0f, 0.0f),
    vec3(0.0f, 0.0f, 1.0f)
);


const mat3 nbl_glsl_BT2020toXYZ = mat3(
    vec3(0.636958f, 0.262700f, 0.000000f),
    vec3(0.144617f, 0.677998f, 0.028073f),
    vec3(0.168881f, 0.059302f, 1.060985f)
);

const mat3 nbl_glsl_HDR10_ST2084toXYZ = nbl_glsl_BT2020toXYZ;

const mat3 nbl_glsl_DOLBYIVISIONtoXYZ = nbl_glsl_BT2020toXYZ;

const mat3 nbl_glsl_HDR10_HLGtoXYZ = nbl_glsl_BT2020toXYZ;


const mat3 nbl_glsl_AdobeRGBtoXYZ = mat3(
    vec3(0.5766690429f, 0.2973449753f, 0.0270313614f),
    vec3(0.1855582379f, 0.6273635663f, 0.0706888525f),
    vec3(0.1882286462f, 0.0752914585f, 0.9913375368f)
);


const mat3 nbl_glsl_ACES2065_1toXYZ = mat3(
    vec3(0.9525523959f,  0.3439664498f, 0.0000000000f),
    vec3(0.0000000000f,  0.7281660966f, 0.0000000000f),
    vec3(0.0000936786f, -0.0721325464f, 1.0088251844f)
);


const mat3 nbl_glsl_ACEScctoXYZ = mat3(
    vec3(0.6624541811f, 0.2722287168f, -0.0055746495f),
    vec3(0.1340042065f, 0.6740817658f,  0.0040607335f),
    vec3(0.1561876870f, 0.0536895174f,  1.0103391003f)
);

const mat3 nbl_glsl_ACESccttoXYZ = nbl_glsl_ACEScctoXYZ;

#endif