#ifndef _NBL_GLSL_BLIT_DESCRIPTORS_INCLUDED_
#define _NBL_GLSL_BLIT_DESCRIPTORS_INCLUDED_

#ifndef _NBL_GLSL_BLIT_DESCRIPTOR_SET_DEFINED_
#define _NBL_GLSL_BLIT_DESCRIPTOR_SET_DEFINED_ 0
#endif

#ifndef _NBL_GLSL_BLIT_WEIGHTS_DESCRIPTOR_DEFINED_
#ifndef _NBL_GLSL_BLIT_WEIGHTS_BINDING_DEFINED_
#define _NBL_GLSL_BLIT_WEIGHTS_BINDING_DEFINED_ 2
#endif
// Todo(achal): I don't think I need row_major here
layout(set = _NBL_GLSL_BLIT_DESCRIPTOR_SET_DEFINED_, binding = _NBL_GLSL_BLIT_WEIGHTS_BINDING_DEFINED_, std140, row_major) uniform Weights
{
	// Todo(achal): Put max3DWindowPixelCount*channelCount here
	float data[1024 * 1024];
} weights;
#endif

#ifndef _NBL_GLSL_BLIT_ALPHA_HISTOGRAM_DESCRIPTOR_DEFINED_
#ifndef _NBL_GLSL_BLIT_ALPHA_HISTOGRAM_BINDING_DEFINED_
#define _NBL_GLSL_BLIT_ALPHA_HISTOGRAM_BINDING_DEFINED_ 3
#endif
layout(set = _NBL_GLSL_BLIT_DESCRIPTOR_SET_DEFINED_, binding = _NBL_GLSL_BLIT_ALPHA_HISTOGRAM_BINDING_DEFINED_) buffer coherent AlphaHistogram
{
	uint data[];
} alphaHistogram;
#endif

#endif
