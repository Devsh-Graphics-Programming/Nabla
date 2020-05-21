#include "../ext/LumaMeter/CLumaMeter.h"
#include "../source/Irrlicht/COpenGLExtensionHandler.h"

#include <cstdio>

using namespace irr;
using namespace irr::asset;
using namespace irr::video;
using namespace ext::LumaMeter;


void CLumaMeter::registerBuiltinGLSLIncludes(asset::IGLSLCompiler* compilerToAddBuiltinIncludeTo)
{
	static bool addedBuiltinHeader = false;
	if (addedBuiltinHeader)
		return;

	if (!compilerToAddBuiltinIncludeTo)
		return;

	compilerToAddBuiltinIncludeTo->getIncludeHandler()->addBuiltinIncludeLoader(core::make_smart_refctd_ptr<CGLSLLumaBuiltinIncludeLoader>());
	addedBuiltinHeader = true;
}


core::smart_refctd_ptr<asset::ICPUSpecializedShader> CLumaMeter::createShader(
	asset::IGLSLCompiler* compilerToAddBuiltinIncludeTo,
	const std::tuple<E_FORMAT,E_COLOR_PRIMARIES,ELECTRO_OPTICAL_TRANSFER_FUNCTION>& inputColorSpace,
	E_METERING_MODE meterMode, float minLuma, float maxLuma
)
{
	auto format = std::get<E_FORMAT>(inputColorSpace);
	if (isIntegerFormat(format))
		return nullptr;

	auto transferFunction = std::get<ELECTRO_OPTICAL_TRANSFER_FUNCTION>(inputColorSpace);
	// I'm expecting scene referred HDR values, it would be insane to quantize them to display electrical signals before tonemapping
	if (transferFunction != EOTF_IDENTITY)
		return nullptr;

	// little check for sanity
	if (isSRGBFormat(format) && transferFunction!=EOTF_sRGB)
		return nullptr;

	auto colorPrimaries = std::get<E_COLOR_PRIMARIES>(inputColorSpace);

	const char* sourceFmt =
R"===(#version 430 core


#ifndef _IRR_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_X_DEFINED_
#define _IRR_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_X_DEFINED_ 16
#endif

#ifndef _IRR_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_Y_DEFINED_
#define _IRR_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_Y_DEFINED_ 16
#endif


#define _IRR_GLSL_EXT_LUMA_METER_MIN_LUMA_DEFINED_ %d
#define _IRR_GLSL_EXT_LUMA_METER_MAX_LUMA_DEFINED_ %d

#ifndef _IRR_GLSL_EXT_LUMA_METER_MODE_DEFINED_
#define _IRR_GLSL_EXT_LUMA_METER_MODE_DEFINED_ %d
#endif

#include "irr/builtin/glsl/colorspace/EOTF.glsl"
#include "irr/builtin/glsl/colorspace/encodeCIEXYZ.glsl"
#include "irr/builtin/glsl/colorspace/decodeCIEXYZ.glsl"
#include "irr/builtin/glsl/colorspace/OETF.glsl"

#define _IRR_GLSL_EXT_LUMA_METER_FIRST_PASS_DEFINED_
#include "irr/builtin/glsl/ext/LumaMeter/common.glsl"

#if _IRR_GLSL_EXT_LUMA_METER_INVOCATION_COUNT!=_IRR_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_X_DEFINED_*_IRR_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_Y_DEFINED_
	#error "_IRR_GLSL_EXT_LUMA_METER_INVOCATION_COUNT does not equal the product of the dispatch sizes!"
#endif
layout(local_size_x=_IRR_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_X_DEFINED_, local_size_y=_IRR_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_Y_DEFINED_) in;



#ifndef _IRR_GLSL_EXT_LUMA_METER_UNIFORMS_DESCRIPTOR_DEFINED_
#define _IRR_GLSL_EXT_LUMA_METER_UNIFORMS_DESCRIPTOR_DEFINED_
layout(set=_IRR_GLSL_EXT_LUMA_METER_UNIFORMS_SET_DEFINED_, binding=_IRR_GLSL_EXT_LUMA_METER_UNIFORMS_BINDING_DEFINED_) uniform Uniforms
{
	irr_glsl_ext_LumaMeter_Uniforms_t inParams;
};
#endif



#ifndef _IRR_GLSL_EXT_LUMA_METER_GET_COLOR_DEFINED_
#define _IRR_GLSL_EXT_LUMA_METER_GET_COLOR_DEFINED_
vec3 irr_glsl_ext_LumaMeter_getColor()
{
	vec2 uv = vec2(gl_GlobalInvocationID.xy)*inParams.meteringWindowScale+inParams.meteringWindowOffset;
	return textureLod(inputImage,vec3(uv,float(gl_GlobalInvocationID.z)),0.0).rgb;
}
#endif



#if _IRR_GLSL_EXT_LUMA_METER_MODE_DEFINED_==_IRR_GLSL_EXT_LUMA_METER_MODE_MODE
	void irr_glsl_ext_LumaMeter_clearHistogram()
	{
		for (int i=0; i<_IRR_GLSL_EXT_LUMA_METER_LOCAL_REPLICATION; i++)
			_IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex+i*_IRR_GLSL_EXT_LUMA_METER_INVOCATION_COUNT] = 0u;

		#if _IRR_GLSL_EXT_LUMA_METER_LOCAL_REPLICATION_POW_DEFINED_>0
			if (gl_LocalInvocationIndex<_IRR_GLSL_EXT_LUMA_METER_LOCAL_REPLICATION)
				_IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex+_IRR_GLSL_EXT_LUMA_METER_LOCAL_REPLICATION*_IRR_GLSL_EXT_LUMA_METER_INVOCATION_COUNT] = 0u;
		#endif
	}
#endif


#ifndef _IRR_GLSL_EXT_LUMA_METER_EOTF_DEFINED_
#define _IRR_GLSL_EXT_LUMA_METER_EOTF_DEFINED_ %s
#endif

#ifndef _IRR_GLSL_EXT_LUMA_METER_XYZ_CONVERSION_MATRIX_DEFINED_
#define _IRR_GLSL_EXT_LUMA_METER_XYZ_CONVERSION_MATRIX_DEFINED_ %s
#endif

#ifndef _IRR_GLSL_EXT_LUMA_METER_IMPL_DEFINED_
#define _IRR_GLSL_EXT_LUMA_METER_IMPL_DEFINED_
void irr_glsl_ext_LumaMeter() // bool wgExecutionMask, then do if(any(wgExecutionMask))
{
	vec3 color = irr_glsl_ext_LumaMeter_getColor();
	#if _IRR_GLSL_EXT_LUMA_METER_MODE_DEFINED_==_IRR_GLSL_EXT_LUMA_METER_MODE_MODE
		irr_glsl_ext_LumaMeter_clearHistogram();
	#endif
	irr_glsl_ext_LumaMeter_clearFirstPassOutput();

	// linearize
	vec3 linear = _IRR_GLSL_EXT_LUMA_METER_EOTF_DEFINED_(color);
	// transform to CIE
	float luma = dot(transpose(_IRR_GLSL_EXT_LUMA_METER_XYZ_CONVERSION_MATRIX_DEFINED_)[1],linear);
	// clamp to sane values
	const float MinLuma = intBitsToFloat(_IRR_GLSL_EXT_LUMA_METER_MIN_LUMA_DEFINED_);
	const float MaxLuma = intBitsToFloat(_IRR_GLSL_EXT_LUMA_METER_MAX_LUMA_DEFINED_);
	luma = clamp(luma,MinLuma,MaxLuma);

	float logLuma = log2(luma/MinLuma)/log2(MaxLuma/MinLuma);
	#if _IRR_GLSL_EXT_LUMA_METER_MODE_DEFINED_==_IRR_GLSL_EXT_LUMA_METER_MODE_MODE
		// compute histogram index
		int histogramIndex = int(logLuma*float(_IRR_GLSL_EXT_LUMA_METER_BIN_COUNT-1u)+0.5);
		histogramIndex += int(gl_LocalInvocationIndex&uint(_IRR_GLSL_EXT_LUMA_METER_LOCAL_REPLICATION-1))*_IRR_GLSL_EXT_LUMA_METER_PADDED_BIN_COUNT;
		// barrier so we "see" the cleared histogram
		barrier();
		memoryBarrierShared();
		atomicAdd(_IRR_GLSL_SCRATCH_SHARED_DEFINED_[histogramIndex],1u);

		// no barrier on shared memory cause if we use it with atomics the writes and reads be coherent
		barrier();

		// join the histograms across workgroups
		uint writeOutVal = _IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex];
		for (int i=1; i<_IRR_GLSL_EXT_LUMA_METER_LOCAL_REPLICATION; i++)
			writeOutVal += _IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex+i*_IRR_GLSL_EXT_LUMA_METER_PADDED_BIN_COUNT];
	#elif _IRR_GLSL_EXT_LUMA_METER_MODE_DEFINED_==_IRR_GLSL_EXT_LUMA_METER_MODE_GEOM_MEAN
		_IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex] = floatBitsToUint(logLuma);
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

#ifndef _IRR_GLSL_EXT_LUMA_METER_MAIN_DEFINED_
#define _IRR_GLSL_EXT_LUMA_METER_MAIN_DEFINED_
void main()
{
	irr_glsl_ext_LumaMeter();
}
#endif
)===";

	constexpr char* eotf = "irr_glsl_eotf_identity";
	constexpr char* xyzMatrices[ECP_COUNT] = 
	{
		"irr_glsl_sRGBtoXYZ",
		"irr_glsl_Display_P3toXYZ",
		"irr_glsl_DCI_P3toXYZ",
		"irr_glsl_BT2020toXYZ",
		"irr_glsl_AdobeRGBtoXYZ",
		"irr_glsl_ACES2065_1toXYZ",
		"irr_glsl_ACEScctoXYZ",
		"#error \"UNDEFINED_COLOR_PRIMARIES\""
	};
	const char* xyzMatrix = xyzMatrices[colorPrimaries];

	constexpr size_t lumaChars = 10ull*2ull;
	constexpr size_t meterModeChars = 1ull;
	const size_t eotfChars = strlen(eotf);
	const size_t xyzMatrixChars = strlen(xyzMatrix);
	const size_t extraSize = lumaChars+meterModeChars+eotfChars+xyzMatrixChars;

	auto shader = core::make_smart_refctd_ptr<ICPUBuffer>(strlen(sourceFmt)+extraSize+1u);
	snprintf(
		reinterpret_cast<char*>(shader->getPointer()),shader->getSize(),sourceFmt,
		reinterpret_cast<const int32_t&>(minLuma),reinterpret_cast<const int32_t&>(maxLuma),meterMode,
		eotf,xyzMatrix
	);

	registerBuiltinGLSLIncludes(compilerToAddBuiltinIncludeTo);
	return core::make_smart_refctd_ptr<ICPUSpecializedShader>(
		core::make_smart_refctd_ptr<ICPUShader>(std::move(shader),ICPUShader::buffer_contains_glsl),
		ISpecializedShader::SInfo{nullptr, nullptr, "main", asset::ISpecializedShader::ESS_COMPUTE}
	);
}

void CLumaMeter::defaultBarrier()
{
	COpenGLExtensionHandler::pGlMemoryBarrier(GL_UNIFORM_BARRIER_BIT|GL_SHADER_STORAGE_BARRIER_BIT|GL_BUFFER_UPDATE_BARRIER_BIT);
}