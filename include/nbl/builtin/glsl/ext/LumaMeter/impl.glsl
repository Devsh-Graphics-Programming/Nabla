// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_GLSL_EXT_LUMA_METER_IMPL_INCLUDED_
#define _NBL_GLSL_EXT_LUMA_METER_IMPL_INCLUDED_


#ifndef _NBL_GLSL_EXT_LUMA_METER_FIRST_PASS_DEFINED_
#define _NBL_GLSL_EXT_LUMA_METER_FIRST_PASS_DEFINED_
#endif
#include "nbl/builtin/glsl/ext/LumaMeter/common.glsl"



#ifndef _NBL_GLSL_EXT_LUMA_METER_EOTF_DEFINED_
#error "You need to define `_NBL_GLSL_EXT_LUMA_METER_EOTF_DEFINED_` !"
#endif

#ifndef _NBL_GLSL_EXT_LUMA_METER_XYZ_CONVERSION_MATRIX_DEFINED_
#error "You need to define `_NBL_GLSL_EXT_LUMA_METER_XYZ_CONVERSION_MATRIX_DEFINED_` !"
#endif


#ifndef _NBL_GLSL_EXT_LUMA_METER_GET_COLOR_DECLARED_
#define _NBL_GLSL_EXT_LUMA_METER_GET_COLOR_DECLARED_
vec3 nbl_glsl_ext_LumaMeter_getColor(bool wgExecutionMask);
#endif

#ifndef _NBL_GLSL_EXT_LUMA_METER_GET_COLOR_DEFINED_
#error "You need to define `nbl_glsl_ext_LumaMeter_getColor` and mark `_NBL_GLSL_EXT_LUMA_METER_GET_COLOR_DEFINED_`!"
#endif

#ifndef _NBL_GLSL_EXT_LUMA_METER_IMPL_DECLARED_
#define _NBL_GLSL_EXT_LUMA_METER_IMPL_DECLARED_
void nbl_glsl_ext_LumaMeter(in bool wgExecutionMask);
#endif

float nbl_glsl_ext_LumaMeter_local_process(in bool wgExecutionMask, in vec3 color)
{
	float scaledLogLuma = 0.f; // this default kind-of makes sense
	// linearize
	if (wgExecutionMask)
	{
		vec3 linear = _NBL_GLSL_EXT_LUMA_METER_EOTF_DEFINED_(color);
		// transform to CIE
		float luma = dot(transpose(_NBL_GLSL_EXT_LUMA_METER_XYZ_CONVERSION_MATRIX_DEFINED_)[1],linear);
		// clamp to sane values
		const float MinLuma = intBitsToFloat(_NBL_GLSL_EXT_LUMA_METER_MIN_LUMA_DEFINED_);
		const float MaxLuma = intBitsToFloat(_NBL_GLSL_EXT_LUMA_METER_MAX_LUMA_DEFINED_);
		luma = clamp(luma,MinLuma,MaxLuma);

		scaledLogLuma = log2(luma/MinLuma)/log2(MaxLuma/MinLuma);
	}

	#if _NBL_GLSL_EXT_LUMA_METER_MODE_DEFINED_==_NBL_GLSL_EXT_LUMA_METER_MODE_MEDIAN
		// compute histogram index
		int histogramIndex;
		if (wgExecutionMask)
		{
			histogramIndex = int(scaledLogLuma*float(_NBL_GLSL_EXT_LUMA_METER_BIN_COUNT-1u)+0.5);
			histogramIndex += int(gl_LocalInvocationIndex&uint(_NBL_GLSL_EXT_LUMA_METER_LOCAL_REPLICATION-1))*_NBL_GLSL_EXT_LUMA_METER_PADDED_BIN_COUNT;
		}
		// barrier so we "see" the cleared histogram
		barrier();
		if (wgExecutionMask)
			atomicAdd(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[histogramIndex],1u);
		// no barrier on shared memory because we read from it later and we need all atomics to be done before we read
		barrier();
	#endif

	return scaledLogLuma;
}

#if _NBL_GLSL_EXT_LUMA_METER_MODE_DEFINED_==_NBL_GLSL_EXT_LUMA_METER_MODE_GEOM_MEAN
#include "nbl/builtin/glsl/workgroup/arithmetic.glsl"
#endif

nbl_glsl_ext_LumaMeter_WriteOutValue_t nbl_glsl_ext_LumaMeter_workgroup_process(in float scaledLogLuma)
{
	#if _NBL_GLSL_EXT_LUMA_METER_MODE_DEFINED_==_NBL_GLSL_EXT_LUMA_METER_MODE_MEDIAN
		// join the histograms across workgroups
		uint writeOutVal = _NBL_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex];
		for (int i=1; i<_NBL_GLSL_EXT_LUMA_METER_LOCAL_REPLICATION; i++)
			writeOutVal += _NBL_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex+i*_NBL_GLSL_EXT_LUMA_METER_PADDED_BIN_COUNT];
		return writeOutVal;
	#elif _NBL_GLSL_EXT_LUMA_METER_MODE_DEFINED_==_NBL_GLSL_EXT_LUMA_METER_MODE_GEOM_MEAN
		return nbl_glsl_workgroupAdd(scaledLogLuma);
	#endif
}

#ifndef _NBL_GLSL_EXT_LUMA_METER_IMPL_DEFINED_
#define _NBL_GLSL_EXT_LUMA_METER_IMPL_DEFINED_
void nbl_glsl_ext_LumaMeter(in bool wgExecutionMask)
{
	vec3 color = nbl_glsl_ext_LumaMeter_getColor(wgExecutionMask);
	#if _NBL_GLSL_EXT_LUMA_METER_MODE_DEFINED_==_NBL_GLSL_EXT_LUMA_METER_MODE_MEDIAN
		nbl_glsl_ext_LumaMeter_clearHistogram();
	#endif
	nbl_glsl_ext_LumaMeter_clearFirstPassOutput();

	const float scaledLogLuma = nbl_glsl_ext_LumaMeter_local_process(wgExecutionMask,color);
	const nbl_glsl_ext_LumaMeter_WriteOutValue_t writeOutVal = nbl_glsl_ext_LumaMeter_workgroup_process(scaledLogLuma);

	nbl_glsl_ext_LumaMeter_setFirstPassOutput(writeOutVal);
}
#endif

#endif