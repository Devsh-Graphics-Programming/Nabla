#ifndef _NBL_BUILTIN_GLSL_FORMAT_CONSTANTS_INCLUDED_
#define _NBL_BUILTIN_GLSL_FORMAT_CONSTANTS_INCLUDED_

//rgb9e5, a standard, shared exponent, floating point format
#define nbl_glsl_RGB9E5_MANTISSA_BITS 9
#define nbl_glsl_RGB9E5_MANTISSA_MASK 0x1ff
#define nbl_glsl_RGB9E5_EXPONENT_BITS 5
#define nbl_glsl_RGB9E5_EXP_BIAS 15
#define nbl_glsl_MAX_RGB9E5_EXP (nbl_glsl_RGB9E5_EXP_BIAS+1)

#define nbl_glsl_MAX_RGB9E5_MANTISSA_VALUES (0x1<<nbl_glsl_RGB9E5_MANTISSA_BITS)
#define nbl_glsl_MAX_RGB9E5_MANTISSA (nbl_glsl_MAX_RGB9E5_MANTISSA_VALUES-1)
#define nbl_glsl_MAX_RGB9E5 float(nbl_glsl_MAX_RGB9E5_MANTISSA)/float(nbl_glsl_MAX_RGB9E5_MANTISSA_VALUES)*exp2(float(nbl_glsl_MAX_RGB9E5_EXP))

#define nbl_glsl_RGB9E5_COMPONENT_BITOFFSETS ivec4(0,nbl_glsl_RGB9E5_MANTISSA_BITS,(2*nbl_glsl_RGB9E5_MANTISSA_BITS)&31,(3*nbl_glsl_RGB9E5_MANTISSA_BITS)&31)


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

//rgb18e7s3, out custom 3 channel, shared exponent signed floating point format
#define nbl_glsl_RGB18E7S3_MANTISSA_BITS 18
#define nbl_glsl_RGB18E7S3_EXPONENT_BITS nbl_glsl_RGB19E7_EXPONENT_BITS
#define nbl_glsl_RGB18E7S3_EXP_BIAS nbl_glsl_RGB19E7_EXP_BIAS
#define nbl_glsl_MAX_RGB18E7S3_EXP (nbl_glsl_RGB18E7S3_EXP_BIAS+1)

#define nbl_glsl_MAX_RGB18E7S3_MANTISSA_VALUES (0x1<<nbl_glsl_RGB18E7S3_MANTISSA_BITS)
#define nbl_glsl_MAX_RGB18E7S3_MANTISSA (nbl_glsl_MAX_RGB18E7S3_MANTISSA_VALUES-1)
#define nbl_glsl_MAX_RGB18E7S3 float(nbl_glsl_MAX_RGB18E7S3_MANTISSA)/float(nbl_glsl_MAX_RGB18E7S3_MANTISSA_VALUES)*exp2(float(nbl_glsl_MAX_RGB18E7S3_EXP))

#define nbl_glsl_RGB18E7S3_COMPONENT_INDICES nbl_glsl_RGB19E7_COMPONENT_INDICES
#define nbl_glsl_RGB18E7S3_COMPONENT_BITOFFSETS ivec4(0,nbl_glsl_RGB18E7S3_MANTISSA_BITS,(2*nbl_glsl_RGB18E7S3_MANTISSA_BITS)&31,(3*nbl_glsl_RGB18E7S3_MANTISSA_BITS)&31)
#define nbl_glsl_RGB18E7S3_G_COMPONENT_SPLIT (32-nbl_glsl_RGB18E7S3_MANTISSA_BITS)

#endif