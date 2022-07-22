#ifndef _NBL_GLSL_EXT_BLUR_DESCRIPTORS_DEFINED_
#define _NBL_GLSL_EXT_BLUR_DESCRIPTORS_DEFINED_

#ifndef _NBL_GLSL_EXT_BLUR_INPUT_SET_DEFINED_
#define _NBL_GLSL_EXT_BLUR_INPUT_SET_DEFINED_ 0
#endif

#ifndef _NBL_GLSL_EXT_BLUR_INPUT_BINDING_DEFINED_
#define _NBL_GLSL_EXT_BLUR_INPUT_BINDING_DEFINED_ 0
#endif

#ifndef _NBL_GLSL_EXT_BLUR_OUTPUT_SET_DEFINED_
#define _NBL_GLSL_EXT_BLUR_OUTPUT_SET_DEFINED_ 0
#endif

#ifndef _NBL_GLSL_EXT_BLUR_OUTPUT_BINDING_DEFINED_
#define _NBL_GLSL_EXT_BLUR_OUTPUT_BINDING_DEFINED_ 1
#endif

#ifndef _NBL_GLSL_EXT_BLUR_INPUT_DESCRIPTOR_DEFINED_
#define _NBL_GLSL_EXT_BLUR_INPUT_DESCRIPTOR_DEFINED_

layout (set = _NBL_GLSL_EXT_BLUR_INPUT_SET_DEFINED_, binding = _NBL_GLSL_EXT_BLUR_INPUT_BINDING_DEFINED_, std430) restrict readonly buffer InputBuffer
{
	float in_values[];
};

#endif

#ifndef _NBL_GLSL_EXT_BLUR_OUTPUT_DESCRIPTOR_DEFINED_
#define _NBL_GLSL_EXT_BLUR_OUTPUT_DESCRIPTOR_DEFINED_

layout (set = _NBL_GLSL_EXT_BLUR_OUTPUT_SET_DEFINED_, binding = _NBL_GLSL_EXT_BLUR_OUTPUT_BINDING_DEFINED_, std430) restrict writeonly buffer OutputBuffer
{
	float out_values[];
};

#endif

#endif