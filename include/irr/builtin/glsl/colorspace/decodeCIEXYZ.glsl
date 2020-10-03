// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _IRR_BUILTIN_GLSL_COLOR_SPACE_DECODE_CIE_XYZ_INCLUDED_
#define _IRR_BUILTIN_GLSL_COLOR_SPACE_DECODE_CIE_XYZ_INCLUDED_

const mat3 irr_glsl_XYZtoscRGB = mat3(  vec3( 3.2404542,-0.9692660, 0.0556434),
                                        vec3(-1.5371385, 1.8760108,-0.2040259),
                                        vec3(-0.4985314, 0.0415560, 1.0572252));

const mat3 irr_glsl_XYZtosRGB = irr_glsl_XYZtoscRGB;

const mat3 irr_glsl_XYZtoBT709 = irr_glsl_XYZtoscRGB;

  
const mat3 irr_glsl_XYZtoDisplay_P3 = mat3( vec3( 2.4934969,-0.8294890, 0.0358458),
                                            vec3(-0.9313836, 1.7626641,-0.0761724),
                                            vec3(-0.4027108, 0.0236247, 0.9568845));


const mat3 irr_glsl_XYZtoDCI_P3 = mat3(vec3(1.0,0.0,0.0),vec3(0.0,1.0,0.0),vec3(0.0,0.0,1.0));

 
const mat3 irr_glsl_XYZtoBT2020 = mat3( vec3( 1.7166512,-0.6666844, 0.0176399),
                                        vec3(-0.3556708, 1.6164812,-0.0427706),
                                        vec3(-0.2533663, 0.0157685, 0.9421031));
 
const mat3 irr_glsl_XYZtoHDR10_ST2084 = irr_glsl_XYZtoBT2020;

const mat3 irr_glsl_XYZtoDOLBYIVISION = irr_glsl_XYZtoBT2020;

const mat3 irr_glsl_XYZtoHDR10_HLG = irr_glsl_XYZtoBT2020;


const mat3 irr_glsl_XYZtoAdobeRGB = mat3(   vec3( 2.04159,-0.96924, 0.01344),
                                            vec3(-0.56501, 1.87597,-0.11836),
                                            vec3(-0.34473, 0.04156, 1.01517));


const mat3 irr_glsl_XYZtoACES2065_1 = mat3( vec3( 1.0498110175, 0.0000000000,-0.0000974845),
                                            vec3(-0.4959030231, 1.3733130458, 0.0982400361),
                                            vec3( 0.0000000000, 0.0000000000, 0.9912520182));


const mat3 irr_glsl_XYZtoACEScc = mat3( vec3( 1.6410234,-0.6636629, 0.0117219),
                                        vec3(-0.3248033, 1.6153316,-0.0082844),
                                        vec3(-0.2364247, 0.0167563, 0.9883949));

const mat3 irr_glsl_XYZtoACEScct = irr_glsl_XYZtoACEScc;

#endif