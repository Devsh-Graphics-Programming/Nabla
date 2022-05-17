#ifndef _NBL_GLSL_BLIT_NORMALIZATION_PARAMETERS_INCLUDED_
#define _NBL_GLSL_BLIT_NORMALIZATION_PARAMETERS_INCLUDED_

struct nbl_glsl_blit_normalization_parameters_t
{
	uvec3 outImageDim;
	uint padding;
	uint inPixelCount;
	float oldReferenceAlpha;
};

#endif