#ifndef _NBL_GLSL_BLIT_ALPHA_TEST_INCLUDED_
#define _NBL_GLSL_BLIT_ALPHA_TEST_INCLUDED_

#ifndef _NBL_GLSL_BLIT_ALPHA_TEST_MAIN_DEFINED_

#include <nbl/builtin/glsl/blit/parameters.glsl>

#ifndef _NBL_GLSL_BLIT_ALPHA_TEST_PASSED_COUNTER_DESCRIPTOR_DEFINED_
#error _NBL_GLSL_BLIT_ALPHA_TEST_PASSED_COUNTER_DESCRIPTOR_DEFINED_ must be defined
#endif

nbl_glsl_blit_parameters_t nbl_glsl_blit_alpha_test_getParameters();

float nbl_glsl_blit_alpha_test_getPaddedData(in ivec3 texCoords);

void nbl_glsl_blit_alpha_test_main()
{
	const float alpha = nbl_glsl_blit_alpha_test_getPaddedData(ivec3(gl_GlobalInvocationID));

	// Todo(achal): Need to pull this out in setData
#if NBL_GLSL_EQUAL(_NBL_GLSL_BLIT_DIM_COUNT_, 1)
	#define LAYER_IDX gl_GlobalInvocationID.y
#elif NBL_GLSL_EQUAL(_NBL_GLSL_BLIT_DIM_COUNT_, 2)
	#define LAYER_IDX gl_GlobalInvocationID.z
#elif NBL_GLSL_EQUAL(_NBL_GLSL_BLIT_DIM_COUNT_, 3)
	#define LAYER_IDX 0
#else
	#error _NBL_GLSL_BLIT_DIM_COUNT_ not supported
#endif
	if (alpha > nbl_glsl_blit_alpha_test_getParameters().referenceAlpha)
		atomicAdd(_NBL_GLSL_BLIT_ALPHA_TEST_PASSED_COUNTER_DESCRIPTOR_DEFINED_.data[LAYER_IDX].passedPixelCount, 1u);

#undef LAYER_IDX
}

#define _NBL_GLSL_BLIT_ALPHA_TEST_MAIN_DEFINED_
#endif

#endif

