#ifndef _IRR_GLSL_EXT_LUMA_METER_IMPL_INCLUDED_
#define _IRR_GLSL_EXT_LUMA_METER_IMPL_INCLUDED_


#ifndef _IRR_GLSL_EXT_LUMA_METER_FIRST_PASS_DEFINED_
#define _IRR_GLSL_EXT_LUMA_METER_FIRST_PASS_DEFINED_
#endif
#include "irr/builtin/glsl/ext/LumaMeter/common.glsl"



#ifndef _IRR_GLSL_EXT_LUMA_METER_EOTF_DEFINED_
#error "You need to define `_IRR_GLSL_EXT_LUMA_METER_EOTF_DEFINED_` !"
#endif

#ifndef _IRR_GLSL_EXT_LUMA_METER_XYZ_CONVERSION_MATRIX_DEFINED_
#error "You need to define `_IRR_GLSL_EXT_LUMA_METER_XYZ_CONVERSION_MATRIX_DEFINED_` !"
#endif


#ifndef _IRR_GLSL_EXT_LUMA_METER_GET_COLOR_DECLARED_
#define _IRR_GLSL_EXT_LUMA_METER_GET_COLOR_DECLARED_
vec3 irr_glsl_ext_LumaMeter_getColor(bool wgExecutionMask);
#endif

#ifndef _IRR_GLSL_EXT_LUMA_METER_GET_COLOR_DEFINED_
#error "You need to define `irr_glsl_ext_LumaMeter_getColor` and mark `_IRR_GLSL_EXT_LUMA_METER_GET_COLOR_DEFINED_`!"
#endif

#ifndef _IRR_GLSL_EXT_LUMA_METER_IMPL_DECLARED_
#define _IRR_GLSL_EXT_LUMA_METER_IMPL_DECLARED_
void irr_glsl_ext_LumaMeter(bool wgExecutionMask);
#endif

#ifndef _IRR_GLSL_EXT_LUMA_METER_IMPL_DEFINED_
#define _IRR_GLSL_EXT_LUMA_METER_IMPL_DEFINED_
void irr_glsl_ext_LumaMeter(bool wgExecutionMask)
{
	vec3 color = irr_glsl_ext_LumaMeter_getColor(wgExecutionMask);
	#if _IRR_GLSL_EXT_LUMA_METER_MODE_DEFINED_==_IRR_GLSL_EXT_LUMA_METER_MODE_MEDIAN
		irr_glsl_ext_LumaMeter_clearHistogram();
	#endif
	irr_glsl_ext_LumaMeter_clearFirstPassOutput();

	float logLuma;
	// linearize
	if (wgExecutionMask)
	{
		vec3 linear = _IRR_GLSL_EXT_LUMA_METER_EOTF_DEFINED_(color);
		// transform to CIE
		float luma = dot(transpose(_IRR_GLSL_EXT_LUMA_METER_XYZ_CONVERSION_MATRIX_DEFINED_)[1],linear);
		// clamp to sane values
		const float MinLuma = intBitsToFloat(_IRR_GLSL_EXT_LUMA_METER_MIN_LUMA_DEFINED_);
		const float MaxLuma = intBitsToFloat(_IRR_GLSL_EXT_LUMA_METER_MAX_LUMA_DEFINED_);
		luma = clamp(luma,MinLuma,MaxLuma);

		logLuma = log2(luma/MinLuma)/log2(MaxLuma/MinLuma);
	}

	#if _IRR_GLSL_EXT_LUMA_METER_MODE_DEFINED_==_IRR_GLSL_EXT_LUMA_METER_MODE_MEDIAN
		// compute histogram index
		int histogramIndex;
		if (wgExecutionMask)
		{
			histogramIndex = int(logLuma*float(_IRR_GLSL_EXT_LUMA_METER_BIN_COUNT-1u)+0.5);
			histogramIndex += int(gl_LocalInvocationIndex&uint(_IRR_GLSL_EXT_LUMA_METER_LOCAL_REPLICATION-1))*_IRR_GLSL_EXT_LUMA_METER_PADDED_BIN_COUNT;
		}
		// barrier so we "see" the cleared histogram
		barrier();
		memoryBarrierShared();
		if (wgExecutionMask)
			atomicAdd(_IRR_GLSL_SCRATCH_SHARED_DEFINED_[histogramIndex],1u);

		// no barrier on shared memory cause if we use it with atomics the writes and reads be coherent
		barrier();

		// join the histograms across workgroups
		uint writeOutVal = _IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex];
		for (int i=1; i<_IRR_GLSL_EXT_LUMA_METER_LOCAL_REPLICATION; i++)
			writeOutVal += _IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex+i*_IRR_GLSL_EXT_LUMA_METER_PADDED_BIN_COUNT];
	#elif _IRR_GLSL_EXT_LUMA_METER_MODE_DEFINED_==_IRR_GLSL_EXT_LUMA_METER_MODE_GEOM_MEAN
		_IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex] = wgExecutionMask ? floatBitsToUint(logLuma):0u;
		for (int i=_IRR_GLSL_EXT_LUMA_METER_INVOCATION_COUNT>>1; i>1; i>>=1)
		{
			barrier();
			memoryBarrierShared();
			if (gl_LocalInvocationIndex<i)
			{
				_IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex] = floatBitsToUint
				(
					uintBitsToFloat(_IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex])+
					uintBitsToFloat(_IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex+i])
				);
			}
		}
		barrier();
		memoryBarrierShared();
		float writeOutVal = uintBitsToFloat(_IRR_GLSL_SCRATCH_SHARED_DEFINED_[0])+uintBitsToFloat(_IRR_GLSL_SCRATCH_SHARED_DEFINED_[1]);
	#endif

	irr_glsl_ext_LumaMeter_setFirstPassOutput(writeOutVal);
}
#endif

#endif