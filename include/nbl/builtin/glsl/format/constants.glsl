#ifndef _NBL_BUILTIN_GLSL_FORMAT_CONSTANTS_INCLUDED_
#define _NBL_BUILTIN_GLSL_FORMAT_CONSTANTS_INCLUDED_

//rgb19e7, our custom 3 channel, shared exponent, floating point format
#define nbl_glsl_RGB19E7_MANTISSA_BITS 19
#define nbl_glsl_RGB19E7_MANTISSA_MASK 0x7ffff
#define nbl_glsl_RGB19E7_EXPONENT_BITS 7
#define nbl_glsl_RGB19E7_EXP_BIAS 63
#define nbl_glsl_MAX_RGB19E7_EXP (nbl_glsl_RGB19E7_EXP_BIAS+1)

#define nbl_glsl_MAX_RGB19E7_MANTISSA_VALUES (0x1<<nbl_glsl_RGB19E7_MANTISSA_BITS)
#define nbl_glsl_MAX_RGB19E7_MANTISSA (nbl_glsl_MAX_RGB19E7_MANTISSA_VALUES-1)
#define nbl_glsl_MAX_RGB19E7 float(nbl_glsl_MAX_RGB19E7_MANTISSA)/float(nbl_glsl_MAX_RGB19E7_MANTISSA_VALUES)*exp2(float(nbl_glsl_MAX_RGB19E7_EXP))

#define nbl_glsl_RGB19E7_COMPONENT_INDICES ivec4(0,0,1,1)
#define nbl_glsl_RGB19E7_COMPONENT_BITOFFSETS ivec4(0,nbl_glsl_RGB19E7_MANTISSA_BITS,(2*nbl_glsl_RGB19E7_MANTISSA_BITS)&31,(3*nbl_glsl_RGB19E7_MANTISSA_BITS)&31)
#define nbl_glsl_RGB19E7_G_COMPONENT_SPLIT (32-nbl_glsl_RGB19E7_MANTISSA_BITS)

#endif