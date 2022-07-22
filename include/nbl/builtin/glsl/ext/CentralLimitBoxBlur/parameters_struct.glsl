#ifndef _NBL_GLSL_EXT_BLUR_PARAMETERS_STRUCT_INCLUDED_
#define _NBL_GLSL_EXT_BLUR_PARAMETERS_STRUCT_INCLUDED_

#ifdef __cplusplus
#define uint uint32_t
#endif

struct nbl_glsl_ext_Blur_Parameters_t
{
	uvec4 input_dimensions;
	uvec4 input_strides;
	uvec4 output_strides;
	float radius;
};

#ifdef __cplusplus
#undef uint
#endif

#endif