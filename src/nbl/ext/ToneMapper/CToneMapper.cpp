// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/ToneMapper/CToneMapper.h"
// #include "../source/Nabla/COpenGLExtensionHandler.h"

#include <cstdio>

using namespace nbl;
using namespace nbl::asset;
using namespace nbl::video;
using namespace ext::ToneMapper;


core::SRange<const IGPUDescriptorSetLayout::SBinding> CToneMapper::getDefaultBindings(ILogicalDevice* device, bool usingLumaMeter)
{
	auto lumaBindings = ext::LumaMeter::CLumaMeter::getDefaultBindings(device);
	const auto inputImageBinding = lumaBindings.begin()[2];
	if (usingLumaMeter)
	{
		assert(lumaBindings.size()==3ull);
		const auto uniformBinding = lumaBindings.begin()[0];
		const auto ssboBinding = lumaBindings.begin()[1];
		static const IGPUDescriptorSetLayout::SBinding bnd[DEFAULT_MAX_DESCRIPTOR_COUNT] =
		{
			uniformBinding,
			ssboBinding,
			inputImageBinding,
			{
				3u,
				EDT_STORAGE_IMAGE,
				1u,
				IShader::ESS_COMPUTE,
				nullptr
			}
		};
		return {bnd,bnd+sizeof(bnd)/sizeof(IGPUDescriptorSetLayout::SBinding)};
	}
	else
	{
		static const IGPUDescriptorSetLayout::SBinding bnd[] =
		{
			{
				0u,
				EDT_STORAGE_IMAGE,
				1u,
				IShader::ESS_COMPUTE,
				nullptr
			},
			{
				1u,
				EDT_STORAGE_BUFFER,
				1u,
				IShader::ESS_COMPUTE,
				nullptr
			},
			{
				2u,
				EDT_COMBINED_IMAGE_SAMPLER,
				1u,
				IShader::ESS_COMPUTE,
				inputImageBinding.samplers
			}
		};
		return {bnd,bnd+sizeof(bnd)/sizeof(IGPUDescriptorSetLayout::SBinding)};
	}
}


core::smart_refctd_ptr<ICPUSpecializedShader> CToneMapper::createShader(
	IGLSLCompiler* compilerToAddBuiltinIncludeTo,
	const std::tuple<E_FORMAT,E_COLOR_PRIMARIES,ELECTRO_OPTICAL_TRANSFER_FUNCTION>& inputColorSpace,
	const std::tuple<E_FORMAT,E_COLOR_PRIMARIES,OPTICO_ELECTRICAL_TRANSFER_FUNCTION>& outputColorSpace,
	E_OPERATOR _operator, bool usingLumaMeter, ext::LumaMeter::CLumaMeter::E_METERING_MODE meterMode, float minLuma, float maxLuma,
	bool usingTemporalAdaptation
)
{
	const char* eotfs[EOTF_UNKNOWN+1] =
	{
		"nbl_glsl_eotf_identity",
		"nbl_glsl_eotf_sRGB",
		"nbl_glsl_eotf_DCI_P3_XYZ",
		"nbl_glsl_eotf_SMPTE_170M",
		"nbl_glsl_eotf_SMPTE_ST2084",
		"nbl_glsl_eotf_HDR10_HLG",
		"nbl_glsl_eotf_Gamma_2_2",
		"nbl_glsl_eotf_ACEScc",
		"nbl_glsl_eotf_ACEScct",
		"#error \"UNDEFINED EOTF!\""
	};
	const char* inXYZMatrices[ECP_COUNT+1] =
	{
		"nbl_glsl_sRGBtoXYZ",
		"nbl_glsl_Display_P3toXYZ",
		"nbl_glsl_DCI_P3toXYZ",
		"nbl_glsl_BT2020toXYZ",
		"nbl_glsl_AdobeRGBtoXYZ",
		"nbl_glsl_ACES2065_1toXYZ",
		"nbl_glsl_ACEScctoXYZ",
		"#error \"Passthrough Color Space not supported!\"",
		"#error \"UNDEFINED_COLOR_PRIMARIES\""
	};
	const char* outXYZMatrices[ECP_COUNT+1] =
	{
		"nbl_glsl_XYZtosRGB",
		"nbl_glsl_XYZtoDisplay_P3",
		"nbl_glsl_XYZtoDCI_P3",
		"nbl_glsl_XYZtoBT2020",
		"nbl_glsl_XYZtoAdobeRGB",
		"nbl_glsl_XYZtoACES2065_1",
		"nbl_glsl_XYZtoACEScc",
		"#error \"Passthrough Color Space not supported!\"",
		"#error \"UNDEFINED_COLOR_PRIMARIES\""
	};
	const char* oetfs[EOTF_UNKNOWN+1] =
	{
		"nbl_glsl_oetf_identity",
		"nbl_glsl_oetf_sRGB",
		"nbl_glsl_oetf_DCI_P3_XYZ",
		"nbl_glsl_oetf_SMPTE_170M",
		"nbl_glsl_oetf_SMPTE_ST2084",
		"nbl_glsl_oetf_HDR10_HLG",
		"nbl_glsl_oetf_Gamma_2_2",
		"nbl_glsl_oetf_ACEScc",
		"nbl_glsl_oetf_ACEScct",
		"#error \"UNDEFINED OETF!\""
	};

	const auto inputFormat = std::get<E_FORMAT>(inputColorSpace);
	auto outputFormat = std::get<E_FORMAT>(outputColorSpace);

	const auto inViewFormat = getInputViewFormat(inputFormat);
	if (inViewFormat==EF_UNKNOWN)
		return nullptr;
	const auto outViewFormat = getOutputViewFormat(outputFormat);
	if (outViewFormat==EF_UNKNOWN)
		return nullptr;

	auto inputOETF = std::get<ELECTRO_OPTICAL_TRANSFER_FUNCTION>(inputColorSpace);
	if (isSRGBFormat(inputFormat))
		inputOETF = EOTF_sRGB;
	auto outputOETF = std::get<OPTICO_ELECTRICAL_TRANSFER_FUNCTION>(outputColorSpace);
	if (isSRGBFormat(outputFormat))
		outputOETF = OETF_sRGB;

	const char* eotf = eotfs[inputOETF];
	const char* inXYZMatrix = inXYZMatrices[std::get<E_COLOR_PRIMARIES>(inputColorSpace)];
	const char* outXYZMatrix = outXYZMatrices[std::get<E_COLOR_PRIMARIES>(outputColorSpace)];
	const char* oetf = oetfs[outputOETF];

	static const core::map<E_FORMAT,const char*> quantizations = 
	{
		{EF_R8G8B8A8_UNORM,"quantizedColor[0] = packUnorm4x8(vec4(color+ditherVal/255.0,colorCIEXYZ.a));"},
		{EF_R8G8B8A8_SRGB,"quantizedColor[0] = packUnorm4x8(vec4(color+ditherVal/255.0,colorCIEXYZ.a));"},
		{EF_A2B10G10R10_UNORM_PACK32,R"===(
	const vec4 limits = vec4(1023.0,1023.0,1023.0,3.0);
	uvec4 preQuant = uvec4(clamp(vec4(color,colorCIEXYZ.a)*limits+ditherVal,vec4(0.0),limits));
	quantizedColor[0] = preQuant.r;
	quantizedColor[0] = bitfieldInsert(quantizedColor[0],preQuant.g,10,10);
	quantizedColor[0] = bitfieldInsert(quantizedColor[0],preQuant.b,20,10);
	quantizedColor[0] = bitfieldInsert(quantizedColor[0],preQuant.a,30,2);
		)==="},
		{EF_R16G16B16A16_UNORM,R"===(
	quantizedColor[0] = packUnorm2x16(color.rg+ditherVal.rg/65535.0);
	quantizedColor[1] = packUnorm2x16(vec2(color.b+ditherVal.b/65535.0,colorCIEXYZ.a));
		)==="},
		{EF_R16G16B16A16_SFLOAT,R"===(
	ivec3 exponent;
	vec3 significant = frexp(color,exponent);
	significant += ditherVal/1024.0;
	vec4 preQuant = vec4(ldexp(significant,exponent),colorCIEXYZ.a);
	quantizedColor[0] = packHalf2x16(preQuant.rg);
	quantizedColor[1] = packHalf2x16(preQuant.ba);
		)==="},
	};

	// B8G8R8A8_SRGB can get identical treatment to R8G8B8A8_SRGB, but
	// the `quantizations` map above don't know that, so silently change
	// B8G8R8A8_SRGB to R8G8B8A8_SRGB, just to get the correct quantization
	if (outputFormat == asset::EF_B8G8R8A8_SRGB)
		outputFormat = asset::EF_R8G8B8A8_SRGB;

	const char* quantization = quantizations.find(outputFormat)->second;
	
	const char* outViewFormatQualifier;
	switch (outViewFormat)
	{
		case EF_R32_UINT:
			outViewFormatQualifier = "r32ui";
			break;
		case EF_R32G32_UINT:
			outViewFormatQualifier = "rg32ui";
			break;
		default:
			outViewFormatQualifier = nullptr;
			break;
	}

	const char* usingLumaMeterDefine = usingLumaMeter ? "#define _NBL_GLSL_EXT_TONE_MAPPER_USING_LUMA_METER_DEFINED_":"";

	const char* usingTemporalAdaptationDefine = usingTemporalAdaptation ? "#define _NBL_GLSL_EXT_TONE_MAPPER_USING_TEMPORAL_ADAPTATION_DEFINED_":"";

	const char* sourceFmt =
R"===(#version 430 core


#ifndef _NBL_GLSL_EXT_TONE_MAPPER_DISPATCH_SIZE_X_DEFINED_
#define _NBL_GLSL_EXT_TONE_MAPPER_DISPATCH_SIZE_X_DEFINED_ %d
#endif

#ifndef _NBL_GLSL_EXT_TONE_MAPPER_DISPATCH_SIZE_Y_DEFINED_
#define _NBL_GLSL_EXT_TONE_MAPPER_DISPATCH_SIZE_Y_DEFINED_ %d
#endif



#define _NBL_GLSL_EXT_TONE_MAPPER_OPERATOR_DEFINED_ %d

#include "nbl/builtin/glsl/ext/ToneMapper/operators.glsl"

#ifndef nbl_glsl_ext_ToneMapper_Params_t
	#if _NBL_GLSL_EXT_TONE_MAPPER_OPERATOR_DEFINED_==_NBL_GLSL_EXT_TONE_MAPPER_REINHARD_OPERATOR
		#define nbl_glsl_ext_ToneMapper_Params_t nbl_glsl_ext_ToneMapper_ReinhardParams_t
	#elif _NBL_GLSL_EXT_TONE_MAPPER_OPERATOR_DEFINED_==_NBL_GLSL_EXT_TONE_MAPPER_ACES_OPERATOR
		#define nbl_glsl_ext_ToneMapper_Params_t nbl_glsl_ext_ToneMapper_ACESParams_t
	#else
		#error "Unsupported Tonemapping Operator"
	#endif
#endif


#ifndef _NBL_GLSL_EXT_TONE_MAPPER_OUTPUT_IMAGE_SET_DEFINED_
#define _NBL_GLSL_EXT_TONE_MAPPER_OUTPUT_IMAGE_SET_DEFINED_ 0
#endif


#ifndef _NBL_GLSL_EXT_TONE_MAPPER_PARAMETERS_SET_DEFINED_
#define _NBL_GLSL_EXT_TONE_MAPPER_PARAMETERS_SET_DEFINED_ 0
#endif

#ifndef _NBL_GLSL_EXT_TONE_MAPPER_PARAMETERS_BINDING_DEFINED_
#define _NBL_GLSL_EXT_TONE_MAPPER_PARAMETERS_BINDING_DEFINED_ 1
#endif

#ifndef _NBL_GLSL_EXT_TONE_MAPPER_INPUT_IMAGE_SET_DEFINED_
#define _NBL_GLSL_EXT_TONE_MAPPER_INPUT_IMAGE_SET_DEFINED_ 0
#endif

#ifndef _NBL_GLSL_EXT_TONE_MAPPER_INPUT_IMAGE_BINDING_DEFINED_
#define _NBL_GLSL_EXT_TONE_MAPPER_INPUT_IMAGE_BINDING_DEFINED_ 2
#endif


%s // _NBL_GLSL_EXT_TONE_MAPPER_USING_LUMA_METER_DEFINED_

#ifdef _NBL_GLSL_EXT_TONE_MAPPER_USING_LUMA_METER_DEFINED_
	#define _NBL_GLSL_WORKGROUP_SIZE_ (_NBL_GLSL_EXT_TONE_MAPPER_DISPATCH_SIZE_X_DEFINED_*_NBL_GLSL_EXT_TONE_MAPPER_DISPATCH_SIZE_Y_DEFINED_)

	#define _NBL_GLSL_EXT_LUMA_METER_BIN_COUNT %d
	#define _NBL_GLSL_EXT_LUMA_METER_BIN_GLOBAL_REPLICATION %d

	#define _NBL_GLSL_EXT_LUMA_METER_MIN_LUMA_DEFINED_ %d
	#define _NBL_GLSL_EXT_LUMA_METER_MAX_LUMA_DEFINED_ %d

	#define _NBL_GLSL_EXT_LUMA_METER_MODE_DEFINED_ %d

	#include "nbl/builtin/glsl/ext/LumaMeter/common.glsl"


	#ifndef _NBL_GLSL_EXT_TONE_MAPPER_UNIFORMS_DEFINED_
	#define _NBL_GLSL_EXT_TONE_MAPPER_UNIFORMS_DEFINED_
	layout(set=_NBL_GLSL_EXT_LUMA_METER_UNIFORMS_SET_DEFINED_, binding=_NBL_GLSL_EXT_LUMA_METER_UNIFORMS_BINDING_DEFINED_) uniform LumaPassInfo
	{
		nbl_glsl_ext_LumaMeter_Uniforms_t padding0;
		nbl_glsl_ext_LumaMeter_PassInfo_t lumaPassInfo;
	};
	#endif


	#if _NBL_GLSL_EXT_TONE_MAPPER_PARAMETERS_SET_DEFINED_!=_NBL_GLSL_EXT_LUMA_METER_OUTPUT_SET_DEFINED_ || _NBL_GLSL_EXT_TONE_MAPPER_PARAMETERS_BINDING_DEFINED_!=_NBL_GLSL_EXT_LUMA_METER_OUTPUT_BINDING_DEFINED_
		#error "Luma/Tonemapper SSBO Set or Binding don't match!"
	#endif

	#if _NBL_GLSL_EXT_TONE_MAPPER_INPUT_IMAGE_SET_DEFINED_!=_NBL_GLSL_EXT_LUMA_METER_INPUT_IMAGE_SET_DEFINED_ || _NBL_GLSL_EXT_TONE_MAPPER_INPUT_IMAGE_BINDING_DEFINED_!=_NBL_GLSL_EXT_LUMA_METER_INPUT_IMAGE_BINDING_DEFINED_
		#error "Input Image Set or Binding don't match!"
	#endif


	#ifndef _NBL_GLSL_EXT_TONE_MAPPER_OUTPUT_IMAGE_BINDING_DEFINED_
	#define _NBL_GLSL_EXT_TONE_MAPPER_OUTPUT_IMAGE_BINDING_DEFINED_ 3
	#endif
#else
	#ifndef _NBL_GLSL_EXT_TONE_MAPPER_OUTPUT_IMAGE_BINDING_DEFINED_
	#define _NBL_GLSL_EXT_TONE_MAPPER_OUTPUT_IMAGE_BINDING_DEFINED_ 0
	#endif
#endif

layout(local_size_x=_NBL_GLSL_EXT_TONE_MAPPER_DISPATCH_SIZE_X_DEFINED_, local_size_y=_NBL_GLSL_EXT_TONE_MAPPER_DISPATCH_SIZE_Y_DEFINED_) in;


#include "nbl/builtin/glsl/colorspace/EOTF.glsl"
#include "nbl/builtin/glsl/colorspace/encodeCIEXYZ.glsl"
#include "nbl/builtin/glsl/colorspace/decodeCIEXYZ.glsl"
#include "nbl/builtin/glsl/colorspace/OETF.glsl"


%s // _NBL_GLSL_EXT_TONE_MAPPER_USING_TEMPORAL_ADAPTATION_DEFINED_


#if defined(_NBL_GLSL_EXT_TONE_MAPPER_USING_LUMA_METER_DEFINED_)||defined(_NBL_GLSL_EXT_TONE_MAPPER_USING_TEMPORAL_ADAPTATION_DEFINED_)
	#ifndef _NBL_GLSL_EXT_TONE_MAPPER_PUSH_CONSTANTS_DEFINED_
	#define _NBL_GLSL_EXT_TONE_MAPPER_PUSH_CONSTANTS_DEFINED_
	layout(push_constant) uniform PushConstants
	{
		int currentFirstPassOutput;
	} pc;
	#endif
#endif


#ifdef _NBL_GLSL_EXT_TONE_MAPPER_USING_LUMA_METER_DEFINED_
	nbl_glsl_ext_LumaMeter_PassInfo_t nbl_glsl_ext_ToneMapper_getLumaMeterInfo()
	{
		return lumaPassInfo;
	}
#endif


#ifdef _NBL_GLSL_EXT_TONE_MAPPER_USING_TEMPORAL_ADAPTATION_DEFINED_
	#define _NBL_GLSL_EXT_TONE_MAPPER_PARAMETERS_QUALIFIERS restrict
#else
	#define _NBL_GLSL_EXT_TONE_MAPPER_PARAMETERS_QUALIFIERS restrict readonly
#endif

struct nbl_glsl_ext_ToneMapper_input_t
{
	uint lastFrameExtraEV; // packed stuff
	uint packedExposureAdaptationFactors; // first is up, then down
	nbl_glsl_ext_ToneMapper_Params_t inParams;
};

#ifndef _NBL_GLSL_EXT_TONE_MAPPER_SSBO_DESCRIPTOR_DEFINED_
#define _NBL_GLSL_EXT_TONE_MAPPER_SSBO_DESCRIPTOR_DEFINED_
layout(set=_NBL_GLSL_EXT_TONE_MAPPER_PARAMETERS_SET_DEFINED_, binding=_NBL_GLSL_EXT_TONE_MAPPER_PARAMETERS_BINDING_DEFINED_) _NBL_GLSL_EXT_TONE_MAPPER_PARAMETERS_QUALIFIERS buffer ParameterBuffer
{
	nbl_glsl_ext_ToneMapper_input_t toneMapperParams;
	uvec4 padding1[15];
	#ifdef _NBL_GLSL_EXT_TONE_MAPPER_USING_LUMA_METER_DEFINED_
		nbl_glsl_ext_LumaMeter_output_t lumaParams[];
	#endif
};
#endif


nbl_glsl_ext_ToneMapper_Params_t nbl_glsl_ext_ToneMapper_getToneMapperParams()
{
	return toneMapperParams.inParams;
}


#ifdef _NBL_GLSL_EXT_TONE_MAPPER_USING_TEMPORAL_ADAPTATION_DEFINED_
	float nbl_glsl_ext_ToneMapper_getLastFrameLuma()
	{
		return unpackHalf2x16(toneMapperParams.lastFrameExtraEV)[pc.currentFirstPassOutput];
	}
	void nbl_glsl_ext_ToneMapper_setLastFrameLuma(in float thisLuma)
	{
		if (all(equal(uvec3(0,0,0),gl_WorkGroupID)))
		{
			vec2 wholeVal = vec2(thisLuma,thisLuma);
			wholeVal[pc.currentFirstPassOutput] = nbl_glsl_ext_ToneMapper_getLastFrameLuma();
			toneMapperParams.lastFrameExtraEV = packHalf2x16(wholeVal);
		}
	}

	float nbl_glsl_ext_ToneMapper_getExposureAdaptationFactor(in float toLastLumaDiff)
	{
		return unpackHalf2x16(toneMapperParams.packedExposureAdaptationFactors)[toLastLumaDiff<0.f ? 0:1];
	}
#endif


#ifndef _NBL_GLSL_EXT_TONE_MAPPER_INPUT_IMAGE_DESCRIPTOR_DEFINED_
#define _NBL_GLSL_EXT_TONE_MAPPER_INPUT_IMAGE_DESCRIPTOR_DEFINED_
layout(set=_NBL_GLSL_EXT_TONE_MAPPER_INPUT_IMAGE_SET_DEFINED_, binding=_NBL_GLSL_EXT_TONE_MAPPER_INPUT_IMAGE_BINDING_DEFINED_) uniform sampler2DArray inputImage;
#endif

#ifndef _NBL_GLSL_EXT_TONE_MAPPER_OUTPUT_IMAGE_DESCRIPTOR_DEFINED_
#define _NBL_GLSL_EXT_TONE_MAPPER_OUTPUT_IMAGE_DESCRIPTOR_DEFINED_
layout(set=_NBL_GLSL_EXT_TONE_MAPPER_OUTPUT_IMAGE_SET_DEFINED_, binding=_NBL_GLSL_EXT_TONE_MAPPER_OUTPUT_IMAGE_BINDING_DEFINED_, %s) uniform uimage2DArray outputImage;
#endif


#ifdef _NBL_GLSL_EXT_TONE_MAPPER_USING_LUMA_METER_DEFINED_
	nbl_glsl_ext_LumaMeter_output_SPIRV_CROSS_is_dumb_t nbl_glsl_ext_ToneMapper_getLumaMeterOutput()
	{
		nbl_glsl_ext_LumaMeter_output_SPIRV_CROSS_is_dumb_t retval;
		#define FETCH_STRUCT lumaParams[(pc.currentFirstPassOutput!=0 ? textureSize(inputImage,0).z:0)+int(gl_WorkGroupID.z)]

		#if _NBL_GLSL_EXT_LUMA_METER_MODE_DEFINED_==_NBL_GLSL_EXT_LUMA_METER_MODE_MEDIAN
			retval = FETCH_STRUCT.packedHistogram[gl_LocalInvocationIndex];
			for (int i=1; i<_NBL_GLSL_EXT_LUMA_METER_BIN_GLOBAL_REPLICATION; i++)
				retval += FETCH_STRUCT.packedHistogram[gl_LocalInvocationIndex+i*_NBL_GLSL_EXT_LUMA_METER_BIN_COUNT];
		#elif _NBL_GLSL_EXT_LUMA_METER_MODE_DEFINED_==_NBL_GLSL_EXT_LUMA_METER_MODE_GEOM_MEAN
			retval = FETCH_STRUCT.unormAverage;
		#endif

		#undef FETCH_STRUCT
		return retval;
	}
#endif


vec4 nbl_glsl_ext_ToneMapper_readColor()
{
	ivec3 uv = ivec3(gl_GlobalInvocationID);
	vec4 color = texelFetch(inputImage,uv,0);
	color.rgb = %s(color.rgb);

	const mat3 xyzMatrix = %s;
	color.rgb = xyzMatrix*color.rgb;

	return color;
}

void nbl_glsl_ext_ToneMapper_writeColor(in vec4 colorCIEXYZ, in vec3 ditherVal)
{
	const mat3 xyzMatrix = %s;
	const vec3 color = %s(xyzMatrix*colorCIEXYZ.rgb);

	uvec4 quantizedColor;
	%s

	ivec3 uv = ivec3(gl_GlobalInvocationID);
	imageStore(outputImage,uv,quantizedColor);
}


#ifndef _NBL_GLSL_EXT_TONE_MAPPER_IMPL_DEFINED_
#define _NBL_GLSL_EXT_TONE_MAPPER_IMPL_DEFINED_
void nbl_glsl_ext_ToneMapper() // bool wgExecutionMask, then do if(any(wgExecutionMask))
{
	ivec3 uv = ivec3(gl_GlobalInvocationID);
	bool alive = any(lessThan(uv,textureSize(inputImage,0)));

	vec4 colorCIEXYZ;
	if (alive)
		colorCIEXYZ = nbl_glsl_ext_ToneMapper_readColor();

	nbl_glsl_ext_ToneMapper_Params_t params = nbl_glsl_ext_ToneMapper_getToneMapperParams();

	float extraNegEV = 0.0;
#ifdef _NBL_GLSL_EXT_TONE_MAPPER_USING_LUMA_METER_DEFINED_
	extraNegEV = nbl_glsl_ext_LumaMeter_getMeasuredLumaLog2(nbl_glsl_ext_ToneMapper_getLumaMeterOutput(),nbl_glsl_ext_ToneMapper_getLumaMeterInfo());
#endif
#ifdef _NBL_GLSL_EXT_TONE_MAPPER_USING_TEMPORAL_ADAPTATION_DEFINED_
	float toLastLumaDiff = nbl_glsl_ext_ToneMapper_getLastFrameLuma()-extraNegEV;
	extraNegEV += toLastLumaDiff*nbl_glsl_ext_ToneMapper_getExposureAdaptationFactor(toLastLumaDiff);
	nbl_glsl_ext_ToneMapper_setLastFrameLuma(extraNegEV);
	#if _NBL_GLSL_EXT_TONE_MAPPER_OPERATOR_DEFINED_==_NBL_GLSL_EXT_TONE_MAPPER_REINHARD_OPERATOR
		params.keyAndManualLinearExposure *= exp2(-extraNegEV);
		colorCIEXYZ.rgb = nbl_glsl_ext_ToneMapper_Reinhard(params,colorCIEXYZ.rgb);
	#elif _NBL_GLSL_EXT_TONE_MAPPER_OPERATOR_DEFINED_==_NBL_GLSL_EXT_TONE_MAPPER_ACES_OPERATOR
		params.exposure -= extraNegEV;
		colorCIEXYZ.rgb = nbl_glsl_ext_ToneMapper_ACES(params,colorCIEXYZ.rgb);
	#endif
#endif

	// TODO: Add dithering
	vec3 rand = vec3(0.5);
	if (alive)
		nbl_glsl_ext_ToneMapper_writeColor(colorCIEXYZ,rand);
}
#endif

#ifndef _NBL_GLSL_EXT_TONE_MAPPER_MAIN_DEFINED_
#define _NBL_GLSL_EXT_TONE_MAPPER_MAIN_DEFINED_
void main()
{
	nbl_glsl_ext_ToneMapper();
}
#endif
)===";

	constexpr size_t wgDimAndOperatorChars = 1ull+4ull*2ull;
	const size_t usingLumaMeterDefineChars = strlen(usingLumaMeterDefine);
	constexpr size_t lumaChars = 10ull*2ull;
	constexpr size_t meterModeChars = 1ull;
	const size_t usingTemporalAdaptationDefineChars = strlen(usingTemporalAdaptationDefine);
	const size_t outViewFormatQualifierChars = strlen(outViewFormatQualifier);
	const size_t eotfChars = strlen(eotf);
	const size_t inXYZMatrixChars = strlen(inXYZMatrix);
	const size_t outXYZMatrixChars = strlen(outXYZMatrix);
	const size_t oetfChars = strlen(oetf);
	const size_t quantizationChars = strlen(quantization);
	const size_t extraSize =	wgDimAndOperatorChars+usingLumaMeterDefineChars+lumaChars+meterModeChars+
								usingTemporalAdaptationDefineChars+
								outViewFormatQualifierChars+eotfChars+inXYZMatrixChars+outXYZMatrixChars+oetfChars+quantizationChars;

	auto shader = core::make_smart_refctd_ptr<ICPUBuffer>(strlen(sourceFmt)+extraSize+1u);
	snprintf(
		reinterpret_cast<char*>(shader->getPointer()),shader->getSize(),sourceFmt,
		DEFAULT_WORKGROUP_DIM,DEFAULT_WORKGROUP_DIM,_operator,
		usingLumaMeterDefine,DEFAULT_WORKGROUP_DIM*DEFAULT_WORKGROUP_DIM,LumaMeter::CLumaMeter::DEFAULT_BIN_GLOBAL_REPLICATION,
		reinterpret_cast<const int32_t&>(minLuma),reinterpret_cast<const int32_t&>(maxLuma),meterMode,
		usingTemporalAdaptationDefine,
		outViewFormatQualifier,eotf,inXYZMatrix,outXYZMatrix,oetf,quantization
	);

	return core::make_smart_refctd_ptr<ICPUSpecializedShader>(
		core::make_smart_refctd_ptr<ICPUShader>(std::move(shader),ICPUShader::buffer_contains_glsl, asset::IShader::ESS_COMPUTE, "????"),
		ISpecializedShader::SInfo{nullptr, nullptr, "main"});
}