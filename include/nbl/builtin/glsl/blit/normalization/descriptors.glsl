#ifndef _NBL_GLSL_BLIT_NORMALIZATION_DESCRIPTORS_INCLUDED_
#define _NBL_GLSL_BLIT_NORMALIZATION_DESCRIPTORS_INCLUDED_

#ifndef _NBL_GLSL_BLIT_NORMALIZATION_DESCRIPTOR_SET_DEFINED_
#define _NBL_GLSL_BLIT_NORMALIZATION_DESCRIPTOR_SET_DEFINED_ 0
#endif

#ifndef _NBL_GLSL_BLIT_NORMALIZATION_IN_IMAGE_DESCRIPTOR_DEFINED_
#define _NBL_GLSL_BLIT_NORMALIZATION_IN_IMAGE_DESCRIPTOR_DEFINED_ nbl_glsl_blit_normalization_inImage

	#ifndef _NBL_GLSL_BLIT_NORMALIZATION_IN_IMAGE_BINDING_DEFINED_
		#define _NBL_GLSL_BLIT_NORMALIZATION_IN_IMAGE_BINDING_DEFINED_ 0
	#endif

	#ifndef _NBL_GLSL_BLIT_NORMALIZATION_IN_SAMPLER_TYPE_
		#error _NBL_GLSL_BLIT_NORMALIZATION_IN_SAMPLER_TYPE_ must be defined to any of sampler1D/sampler2D/sampler3D
	#endif

	layout(set = _NBL_GLSL_BLIT_NORMALIZATION_DESCRIPTOR_SET_DEFINED_, binding = _NBL_GLSL_BLIT_NORMALIZATION_IN_IMAGE_BINDING_DEFINED_) uniform _NBL_GLSL_BLIT_NORMALIZATION_IN_SAMPLER_TYPE_ _NBL_GLSL_BLIT_NORMALIZATION_IN_IMAGE_DESCRIPTOR_DEFINED_;

#endif

#ifndef _NBL_GLSL_BLIT_NORMALIZATION_OUT_IMAGE_DESCRIPTOR_DEFINED_
#define _NBL_GLSL_BLIT_NORMALIZATION_OUT_IMAGE_DESCRIPTOR_DEFINED_ nbl_glsl_blit_normalization_outImage

	#ifndef _NBL_GLSL_BLIT_NORMALIZATION_OUT_IMAGE_BINDING_DEFINED_
		#define _NBL_GLSL_BLIT_NORMALIZATION_OUT_IMAGE_BINDING_DEFINED_ 1
	#endif

	#ifndef _NBL_GLSL_BLIT_NORMALIZATION_OUT_IMAGE_TYPE_
		#error _NBL_GLSL_BLIT_NORMALIZATION_OUT_IMAGE_TYPE_ must be defined
	#endif

	#ifndef _NBL_GLSL_BLIT_NORMALIZATION_OUT_IMAGE_FORMAT_
		#error _NBL_GLSL_BLIT_NORMALIZATION_OUT_IMAGE_FORMAT_ must be defined to a valid storage image format
	#endif

	layout(set = _NBL_GLSL_BLIT_NORMALIZATION_DESCRIPTOR_SET_DEFINED_, binding = _NBL_GLSL_BLIT_NORMALIZATION_OUT_IMAGE_BINDING_DEFINED_, _NBL_GLSL_BLIT_NORMALIZATION_OUT_IMAGE_FORMAT_) uniform _NBL_GLSL_BLIT_NORMALIZATION_OUT_IMAGE_TYPE_ _NBL_GLSL_BLIT_NORMALIZATION_OUT_IMAGE_DESCRIPTOR_DEFINED_;

#endif

#ifndef _NBL_GLSL_BLIT_NORMALIZATION_PASSED_COUNTER_ALPHA_HISTOGRAM_DESCRIPTOR_DEFINED_
#define _NBL_GLSL_BLIT_NORMALIZATION_PASSED_COUNTER_ALPHA_HISTOGRAM_DESCRIPTOR_DEFINED_ nbl_glsl_blit_normalization_passedCounterAlphaHistogram

	#ifndef _NBL_GLSL_BLIT_ALPHA_STATISTICS_DEFINED_
		#error _NBL_GLSL_BLIT_ALPHA_STATISTICS_DEFINED_ must be defined
	#endif

	#ifndef _NBL_GLSL_BLIT_NORMALIZATION_PASSED_COUNTER_ALPHA_HISTOGRAM_BINDING_DEFINED_
		#define _NBL_GLSL_BLIT_NORMALIZATION_PASSED_COUNTER_ALPHA_HISTOGRAM_BINDING_DEFINED_ 2
	#endif

	layout(set = _NBL_GLSL_BLIT_NORMALIZATION_DESCRIPTOR_SET_DEFINED_, binding = _NBL_GLSL_BLIT_NORMALIZATION_PASSED_COUNTER_ALPHA_HISTOGRAM_BINDING_DEFINED_) buffer restrict readonly PassedCounterAlphaHistogram
	{
		_NBL_GLSL_BLIT_ALPHA_STATISTICS_DEFINED_ data[];
	} _NBL_GLSL_BLIT_NORMALIZATION_PASSED_COUNTER_ALPHA_HISTOGRAM_DESCRIPTOR_DEFINED_;

#endif

#endif