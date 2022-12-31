
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_HLSL_BLIT_NORMALIZATION_DESCRIPTORS_INCLUDED_
#define _NBL_HLSL_BLIT_NORMALIZATION_DESCRIPTORS_INCLUDED_


namespace nbl
{
namespace hlsl
{
namespace blit
{
namespace normalization
{



#ifndef DESCRIPTOR_SET_DEFINED_
#define DESCRIPTOR_SET_DEFINED_ 0
#endif

#ifndef IN_IMAGE_DESCRIPTOR_DEFINED_
#define IN_IMAGE_DESCRIPTOR_DEFINED_ inImage

	#ifndef IN_IMAGE_BINDING_DEFINED_
		#define IN_IMAGE_BINDING_DEFINED_ 0
	#endif

	#ifndef IN_SAMPLER_TYPE_
		#error IN_SAMPLER_TYPE_ must be defined to any of sampler1D/sampler2D/sampler3D
	#endif

	layout(set = DESCRIPTOR_SET_DEFINED_, binding = IN_IMAGE_BINDING_DEFINED_) uniform IN_SAMPLER_TYPE_ IN_IMAGE_DESCRIPTOR_DEFINED_;

#endif

#ifndef OUT_IMAGE_DESCRIPTOR_DEFINED_
#define OUT_IMAGE_DESCRIPTOR_DEFINED_ outImage

	#ifndef OUT_IMAGE_BINDING_DEFINED_
		#define OUT_IMAGE_BINDING_DEFINED_ 1
	#endif

	#ifndef OUT_IMAGE_TYPE_
		#error OUT_IMAGE_TYPE_ must be defined
	#endif

	#ifndef OUT_IMAGE_FORMAT_
		#error OUT_IMAGE_FORMAT_ must be defined to a valid storage image format
	#endif

	layout(set = DESCRIPTOR_SET_DEFINED_, binding = OUT_IMAGE_BINDING_DEFINED_, OUT_IMAGE_FORMAT_) uniform OUT_IMAGE_TYPE_ OUT_IMAGE_DESCRIPTOR_DEFINED_;

#endif

#ifndef PASSED_COUNTER_ALPHA_HISTOGRAM_DESCRIPTOR_DEFINED_
#define PASSED_COUNTER_ALPHA_HISTOGRAM_DESCRIPTOR_DEFINED_ passedCounterAlphaHistogram

	#ifndef ALPHA_STATISTICS_DEFINED_
		#error ALPHA_STATISTICS_DEFINED_ must be defined
	#endif

	#ifndef PASSED_COUNTER_ALPHA_HISTOGRAM_BINDING_DEFINED_
		#define PASSED_COUNTER_ALPHA_HISTOGRAM_BINDING_DEFINED_ 2
	#endif

	layout(set = DESCRIPTOR_SET_DEFINED_, binding = PASSED_COUNTER_ALPHA_HISTOGRAM_BINDING_DEFINED_) buffer restrict readonly PassedCounterAlphaHistogram
	{
		ALPHA_STATISTICS_DEFINED_ data[];
	} PASSED_COUNTER_ALPHA_HISTOGRAM_DESCRIPTOR_DEFINED_;

#endif



}
}
}
}

#endif