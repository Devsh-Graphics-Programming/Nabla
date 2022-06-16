#ifndef _NBL_GLSL_BLIT_ALPHA_TEST_INCLUDED_
#define _NBL_GLSL_BLIT_ALPHA_TEST_INCLUDED_

#ifndef _NBL_GLSL_BLIT_ALPHA_TEST_MAIN_DEFINED_

#include <nbl/builtin/glsl/blit/parameters.glsl>

#ifndef _NBL_GLSL_BLIT_ALPHA_TEST_PASSED_COUNTER_DESCRIPTOR_DEFINED_
#error _NBL_GLSL_BLIT_ALPHA_TEST_PASSED_COUNTER_DESCRIPTOR_DEFINED_ must be defined
#endif

nbl_glsl_blit_parameters_t nbl_glsl_blit_getParameters();

float nbl_glsl_blit_alpha_test_getData(in uvec3 coord, in uint layerIdx);

void nbl_glsl_blit_alpha_test_main()
{
	const nbl_glsl_blit_parameters_t params = nbl_glsl_blit_getParameters();

	if (all(lessThan(gl_GlobalInvocationID, params.inDim)))
	{
		const float alpha = nbl_glsl_blit_alpha_test_getData(gl_GlobalInvocationID, gl_WorkGroupID.z);
		if (alpha > params.referenceAlpha)
			atomicAdd(_NBL_GLSL_BLIT_ALPHA_TEST_PASSED_COUNTER_DESCRIPTOR_DEFINED_.data[gl_WorkGroupID.z].passedPixelCount, 1u);
	}
}

#define _NBL_GLSL_BLIT_ALPHA_TEST_MAIN_DEFINED_
#endif

#endif

