#ifndef _NBL_GLSL_BLIT_ALPHA_TEST_INCLUDED_
#define _NBL_GLSL_BLIT_ALPHA_TEST_INCLUDED_

#ifndef _NBL_GLSL_BLIT_ALPHA_TEST_MAIN_DEFINED_

#include <../alpha_test/parameters.glsl>
nbl_glsl_blit_alpha_test_parameters_t nbl_glsl_blit_alpha_test_getParameters();

float nbl_glsl_blit_alpha_test_getPaddedData(in uvec3 texCoords);

void nbl_glsl_blit_alpha_test_main()
{
	const float alpha = nbl_glsl_blit_alpha_test_getPaddedData(gl_GlobalInvocationID);

	if (alpha > nbl_glsl_blit_alpha_test_getParameters().referenceAlpha)
		atomicAdd(alphaTestScratch.data, 1u);
}

#define _NBL_GLSL_BLIT_ALPHA_TEST_MAIN_DEFINED_
#endif

#endif

