// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_COLOR_SPACE_DECODE_CIE_XYZ_INCLUDED_
#define _NBL_BUILTIN_GLSL_COLOR_SPACE_DECODE_CIE_XYZ_INCLUDED_

const mat3 nbl_glsl_XYZtoscRGB = mat3(
    vec3( 3.240970f, -0.969244f,  0.055630f),
    vec3(-1.537383f,  1.875968f, -0.203977f),
    vec3(-0.498611f,  0.041555f,  1.056972f)
);

const mat3 nbl_glsl_XYZtosRGB = nbl_glsl_XYZtoscRGB;

const mat3 nbl_glsl_XYZtoBT709 = nbl_glsl_XYZtoscRGB;

  
const mat3 nbl_glsl_XYZtoDisplay_P3 = mat3(
    vec3( 2.4934969119f,-0.8294889696f, 0.0358458302f),
    vec3(-0.9313836179f, 1.7626640603f,-0.0761723893f),
    vec3(-0.4027107845f, 0.0236246858f, 0.9568845240f)
);


const mat3 nbl_glsl_XYZtoDCI_P3 = mat3(
    vec3(1.0f,0.0f,0.0f),
    vec3(0.0f,1.0f,0.0f),
    vec3(0.0f,0.0f,1.0f)
);

 
const mat3 nbl_glsl_XYZtoBT2020 = mat3(
    vec3( 1.716651f,-0.666684f, 0.017640f),
    vec3(-0.355671f, 1.616481f,-0.042771f),
    vec3(-0.253366f, 0.015769f, 0.942103f)
);
 
const mat3 nbl_glsl_XYZtoHDR10_ST2084 = nbl_glsl_XYZtoBT2020;

const mat3 nbl_glsl_XYZtoDOLBYIVISION = nbl_glsl_XYZtoBT2020;

const mat3 nbl_glsl_XYZtoHDR10_HLG = nbl_glsl_XYZtoBT2020;


const mat3 nbl_glsl_XYZtoAdobeRGB = mat3(
    vec3( 2.0415879038f,-0.9692436363f, 0.0134442806f),
    vec3(-0.5650069743f, 1.8759675015f,-0.1183623922f),
    vec3(-0.3447313508f, 0.0415550574f, 1.0151749944f)
);


const mat3 nbl_glsl_XYZtoACES2065_1 = mat3(
    vec3( 1.0498110175f, -0.4959030231f, 0.0000000000f),
    vec3( 0.0000000000f,  1.3733130458f, 0.0000000000f),
    vec3(-0.0000974845f,  0.0982400361f, 0.9912520182f)
);


const mat3 nbl_glsl_XYZtoACEScc = mat3(
    vec3( 1.6410233797f,-0.6636628587f, 0.0117218943f),
    vec3(-0.3248032942f, 1.6153315917f,-0.0082844420f),
    vec3(-0.2364246952f, 0.0167563477f, 0.9883948585f)
);

const mat3 nbl_glsl_XYZtoACEScct = nbl_glsl_XYZtoACEScc;

#endif