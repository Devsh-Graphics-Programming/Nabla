#ifndef _NBL_GLSL_BLIT_DESCRIPTORS_INCLUDED_
#define _NBL_GLSL_BLIT_DESCRIPTORS_INCLUDED_

#ifndef _NBL_GLSL_BLIT_DESCRIPTOR_SET_DEFINED_
#define _NBL_GLSL_BLIT_DESCRIPTOR_SET_DEFINED_ 0
#endif

#ifndef _NBL_GLSL_BLIT_IN_DESCRIPTOR_DEFINED_
#define _NBL_GLSL_BLIT_IN_DESCRIPTOR_DEFINED_ nbl_glsl_blit_inImage

	#ifndef _NBL_GLSL_BLIT_IN_BINDING_DEFINED_
		#define _NBL_GLSL_BLIT_IN_BINDING_DEFINED_ 0
	#endif

	#ifndef _NBL_GLSL_BLIT_IN_SAMPLER_TYPE_
		#error _NBL_GLSL_BLIT_IN_SAMPLER_TYPE_ must be defined
	#endif

	layout(set = _NBL_GLSL_BLIT_DESCRIPTOR_SET_DEFINED_, binding = _NBL_GLSL_BLIT_IN_BINDING_DEFINED_) uniform _NBL_GLSL_BLIT_IN_SAMPLER_TYPE_ _NBL_GLSL_BLIT_IN_DESCRIPTOR_DEFINED_;

#endif

#ifndef _NBL_GLSL_BLIT_OUT_DESCRIPTOR_DEFINED_
#define _NBL_GLSL_BLIT_OUT_DESCRIPTOR_DEFINED_ nbl_glsl_blit_outImage

	#ifndef _NBL_GLSL_BLIT_OUT_BINDING_DEFINED_
		#define _NBL_GLSL_BLIT_OUT_BINDING_DEFINED_ 1
	#endif

	#ifndef _NBL_GLSL_BLIT_OUT_IMAGE_FORMAT_
		#error _NBL_GLSL_BLIT_OUT_IMAGE_FORMAT_ must be defined to a valid storage image format
	#endif

	#ifndef _NBL_GLSL_BLIT_OUT_IMAGE_TYPE_
		#error _NBL_GLSL_BLIT_OUT_IMAGE_TYPE_ must be defined
	#endif

	layout(set = _NBL_GLSL_BLIT_DESCRIPTOR_SET_DEFINED_, binding = _NBL_GLSL_BLIT_OUT_BINDING_DEFINED_, _NBL_GLSL_BLIT_OUT_IMAGE_FORMAT_) uniform writeonly _NBL_GLSL_BLIT_OUT_IMAGE_TYPE_ _NBL_GLSL_BLIT_OUT_DESCRIPTOR_DEFINED_;

#endif

#ifndef _NBL_GLSL_BLIT_ALPHA_STATISTICS_DESCRIPTOR_DEFINED_
#define _NBL_GLSL_BLIT_ALPHA_STATISTICS_DESCRIPTOR_DEFINED_ nbl_glsl_blit_alphaStatistics

	#ifndef _NBL_GLSL_BLIT_ALPHA_STATISTICS_DEFINED_
		#error _NBL_GLSL_BLIT_ALPHA_STATISTICS_DEFINED_ must be defined
	#endif

	#ifndef _NBL_GLSL_BLIT_ALPHA_STATISTICS_BINDING_DEFINED_
		#define _NBL_GLSL_BLIT_ALPHA_STATISTICS_BINDING_DEFINED_ 2
	#endif

	layout(set = _NBL_GLSL_BLIT_DESCRIPTOR_SET_DEFINED_, binding = _NBL_GLSL_BLIT_ALPHA_STATISTICS_BINDING_DEFINED_) buffer coherent AlphaStatistics
	{
		_NBL_GLSL_BLIT_ALPHA_STATISTICS_DEFINED_  data[];
	} _NBL_GLSL_BLIT_ALPHA_STATISTICS_DESCRIPTOR_DEFINED_;

#endif

#ifndef _NBL_GLSL_BLIT_KERNEL_WEIGHTS_DESCRIPTOR_SET_DEFINED_
#define _NBL_GLSL_BLIT_KERNEL_WEIGHTS_DESCRIPTOR_SET_DEFINED_ 1
#endif

#ifndef _NBL_GLSL_BLIT_KERNEL_WEIGHTS_DESCRIPTOR_DEFINED_
#define _NBL_GLSL_BLIT_KERNEL_WEIGHTS_DESCRIPTOR_DEFINED_ nbl_glsl_blit_kernelWeights

	#ifndef _NBL_GLSL_BLIT_KERNEL_WEIGHTS_BINDING_DEFINED_
		#define _NBL_GLSL_BLIT_KERNEL_WEIGHTS_BINDING_DEFINED_ 0
	#endif

	layout(set = _NBL_GLSL_BLIT_KERNEL_WEIGHTS_DESCRIPTOR_SET_DEFINED_, binding = _NBL_GLSL_BLIT_KERNEL_WEIGHTS_BINDING_DEFINED_) uniform textureBuffer _NBL_GLSL_BLIT_KERNEL_WEIGHTS_DESCRIPTOR_DEFINED_;

#endif

#endif
