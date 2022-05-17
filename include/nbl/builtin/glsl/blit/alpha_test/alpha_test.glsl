#ifndef _NBL_GLSL_BLIT_ALPHA_TEST_INCLUDED_
#define _NBL_GLSL_BLIT_ALPHA_TEST_INCLUDED_

#ifndef _NBL_GLSL_BLIT_ALPHA_TEST_MAIN_DEFINED_

#include <nbl/builtin/glsl/blit/alpha_test/parameters.glsl>

#ifndef _NBL_GLSL_BLIT_ALPHA_TEST_PASSED_COUNTER_DESCRIPTOR_DEFINED_
#error _NBL_GLSL_BLIT_ALPHA_TEST_PASSED_COUNTER_DESCRIPTOR_DEFINED_ must be defined
#endif

nbl_glsl_blit_alpha_test_parameters_t nbl_glsl_blit_alpha_test_getParameters();

float nbl_glsl_blit_alpha_test_getPaddedData(in ivec3 texCoords);

void nbl_glsl_blit_alpha_test_main()
{
	const float alpha = nbl_glsl_blit_alpha_test_getPaddedData(ivec3(gl_GlobalInvocationID));

	if (alpha > nbl_glsl_blit_alpha_test_getParameters().referenceAlpha)
		atomicAdd(_NBL_GLSL_BLIT_ALPHA_TEST_PASSED_COUNTER_DESCRIPTOR_DEFINED_.data, 1u);
}

#define _NBL_GLSL_BLIT_ALPHA_TEST_MAIN_DEFINED_
#endif

#endif

