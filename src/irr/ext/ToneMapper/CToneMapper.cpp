#include "irr/ext/ToneMapper/CToneMapper.h"
#include "../source/Irrlicht/COpenGLExtensionHandler.h"

#include <cstdio>

using namespace irr;
using namespace irr::asset;
using namespace irr::video;
using namespace ext::ToneMapper;


core::SRange<const IGPUDescriptorSetLayout::SBinding> CToneMapper::getDefaultBindings(IVideoDriver* driver, bool usingLumaMeter)
{
	auto lumaBindings = ext::LumaMeter::CLumaMeter::getDefaultBindings(driver);
	const auto inputImageBinding = lumaBindings.begin()[2];
	if (usingLumaMeter)
	{
		assert(lumaBindings.size()==3ull);
		const auto uniformBinding = lumaBindings.begin()[0];
		const auto ssboBinding = lumaBindings.begin()[1];
		static const IGPUDescriptorSetLayout::SBinding bnd[MAX_DESCRIPTOR_COUNT] =
		{
			uniformBinding,
			ssboBinding,
			inputImageBinding,
			{
				3u,
				EDT_STORAGE_IMAGE,
				1u,
				ISpecializedShader::ESS_COMPUTE,
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
				ISpecializedShader::ESS_COMPUTE,
				nullptr
			},
			{
				1u,
				EDT_STORAGE_BUFFER_DYNAMIC,
				1u,
				ISpecializedShader::ESS_COMPUTE,
				nullptr
			},
			{
				2u,
				EDT_COMBINED_IMAGE_SAMPLER,
				1u,
				ISpecializedShader::ESS_COMPUTE,
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
	constexpr char* eotfs[EOTF_UNKNOWN+1] =
	{
		"irr_glsl_eotf_identity",
		"irr_glsl_eotf_sRGB",
		"irr_glsl_eotf_DCI_P3_XYZ",
		"irr_glsl_eotf_SMPTE_170M",
		"irr_glsl_eotf_SMPTE_ST2084",
		"irr_glsl_eotf_HDR10_HLG",
		"irr_glsl_eotf_Gamma_2_2",
		"irr_glsl_eotf_ACEScc",
		"irr_glsl_eotf_ACEScct",
		"#error \"UNDEFINED EOTF!\""
	};
	constexpr char* inXYZMatrices[ECP_COUNT+1] =
	{
		"irr_glsl_sRGBtoXYZ",
		"irr_glsl_Display_P3toXYZ",
		"irr_glsl_DCI_P3toXYZ",
		"irr_glsl_BT2020toXYZ",
		"irr_glsl_AdobeRGBtoXYZ",
		"irr_glsl_ACES2065_1toXYZ",
		"irr_glsl_ACEScctoXYZ",
		"#error \"Passthrough Color Space not supported!\"",
		"#error \"UNDEFINED_COLOR_PRIMARIES\""
	};
	constexpr char* outXYZMatrices[ECP_COUNT+1] =
	{
		"irr_glsl_XYZtosRGB",
		"irr_glsl_XYZtoDisplay_P3",
		"irr_glsl_XYZtoDCI_P3",
		"irr_glsl_XYZtoBT2020",
		"irr_glsl_XYZtoAdobeRGB",
		"irr_glsl_XYZtoACES2065_1",
		"irr_glsl_XYZtoACEScc",
		"#error \"Passthrough Color Space not supported!\"",
		"#error \"UNDEFINED_COLOR_PRIMARIES\""
	};
	constexpr char* oetfs[EOTF_UNKNOWN+1] =
	{
		"irr_glsl_oetf_identity",
		"irr_glsl_oetf_sRGB",
		"irr_glsl_oetf_DCI_P3_XYZ",
		"irr_glsl_oetf_SMPTE_170M",
		"irr_glsl_oetf_SMPTE_ST2084",
		"irr_glsl_oetf_HDR10_HLG",
		"irr_glsl_oetf_Gamma_2_2",
		"irr_glsl_oetf_ACEScc",
		"irr_glsl_oetf_ACEScct",
		"#error \"UNDEFINED OETF!\""
	};

	const auto inputFormat = std::get<E_FORMAT>(inputColorSpace);
	const auto outputFormat = std::get<E_FORMAT>(outputColorSpace);

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

	const char* usingLumaMeterDefine = usingLumaMeter ? "#define _IRR_GLSL_EXT_TONE_MAPPER_USING_LUMA_METER_DEFINED_":"";

	const char* usingTemporalAdaptationDefine = usingTemporalAdaptation ? "#define _IRR_GLSL_EXT_TONE_MAPPER_USING_TEMPORAL_ADAPTATION_DEFINED_":"";

	constexpr char* sourceFmt =
R"===(#version 430 core


#ifndef _IRR_GLSL_EXT_TONE_MAPPER_DISPATCH_SIZE_X_DEFINED_
#define _IRR_GLSL_EXT_TONE_MAPPER_DISPATCH_SIZE_X_DEFINED_ %d
#endif

#ifndef _IRR_GLSL_EXT_TONE_MAPPER_DISPATCH_SIZE_Y_DEFINED_
#define _IRR_GLSL_EXT_TONE_MAPPER_DISPATCH_SIZE_Y_DEFINED_ %d
#endif



#define _IRR_GLSL_EXT_TONE_MAPPER_OPERATOR_DEFINED_ %d

#include "irr/builtin/glsl/ext/ToneMapper/operators.glsl"

#ifndef irr_glsl_ext_ToneMapper_Params_t
	#if _IRR_GLSL_EXT_TONE_MAPPER_OPERATOR_DEFINED_==_IRR_GLSL_EXT_TONE_MAPPER_REINHARD_OPERATOR
		#define irr_glsl_ext_ToneMapper_Params_t irr_glsl_ext_ToneMapper_ReinhardParams_t
	#elif _IRR_GLSL_EXT_TONE_MAPPER_OPERATOR_DEFINED_==_IRR_GLSL_EXT_TONE_MAPPER_ACES_OPERATOR
		#define irr_glsl_ext_ToneMapper_Params_t irr_glsl_ext_ToneMapper_ACESParams_t
	#else
		#error "Unsupported Tonemapping Operator"
	#endif
#endif


#ifndef _IRR_GLSL_EXT_TONE_MAPPER_OUTPUT_IMAGE_SET_DEFINED_
#define _IRR_GLSL_EXT_TONE_MAPPER_OUTPUT_IMAGE_SET_DEFINED_ 0
#endif


#ifndef _IRR_GLSL_EXT_TONE_MAPPER_PARAMETERS_SET_DEFINED_
#define _IRR_GLSL_EXT_TONE_MAPPER_PARAMETERS_SET_DEFINED_ 0
#endif

#ifndef _IRR_GLSL_EXT_TONE_MAPPER_PARAMETERS_BINDING_DEFINED_
#define _IRR_GLSL_EXT_TONE_MAPPER_PARAMETERS_BINDING_DEFINED_ 1
#endif

#ifndef _IRR_GLSL_EXT_TONE_MAPPER_INPUT_IMAGE_SET_DEFINED_
#define _IRR_GLSL_EXT_TONE_MAPPER_INPUT_IMAGE_SET_DEFINED_ 0
#endif

#ifndef _IRR_GLSL_EXT_TONE_MAPPER_INPUT_IMAGE_BINDING_DEFINED_
#define _IRR_GLSL_EXT_TONE_MAPPER_INPUT_IMAGE_BINDING_DEFINED_ 2
#endif


%s // _IRR_GLSL_EXT_TONE_MAPPER_USING_LUMA_METER_DEFINED_

#ifdef _IRR_GLSL_EXT_TONE_MAPPER_USING_LUMA_METER_DEFINED_
	#define _IRR_GLSL_EXT_LUMA_METER_INVOCATION_COUNT (_IRR_GLSL_EXT_TONE_MAPPER_DISPATCH_SIZE_X_DEFINED_*_IRR_GLSL_EXT_TONE_MAPPER_DISPATCH_SIZE_Y_DEFINED_)

	#define _IRR_GLSL_EXT_LUMA_METER_BIN_COUNT %d
	#define _IRR_GLSL_EXT_LUMA_METER_BIN_GLOBAL_REPLICATION %d

	#define _IRR_GLSL_EXT_LUMA_METER_MIN_LUMA_DEFINED_ %d
	#define _IRR_GLSL_EXT_LUMA_METER_MAX_LUMA_DEFINED_ %d

	#define _IRR_GLSL_EXT_LUMA_METER_MODE_DEFINED_ %d

	#include "irr/builtin/glsl/ext/LumaMeter/common.glsl"


	#ifndef _IRR_GLSL_EXT_TONE_MAPPER_UNIFORMS_DEFINED_
	#define _IRR_GLSL_EXT_TONE_MAPPER_UNIFORMS_DEFINED_
	layout(set=_IRR_GLSL_EXT_LUMA_METER_UNIFORMS_SET_DEFINED_, binding=_IRR_GLSL_EXT_LUMA_METER_UNIFORMS_BINDING_DEFINED_) uniform LumaPassInfo
	{
		irr_glsl_ext_LumaMeter_Uniforms_t padding0;
		irr_glsl_ext_LumaMeter_PassInfo_t lumaPassInfo;
	};
	#endif


	#if _IRR_GLSL_EXT_TONE_MAPPER_PARAMETERS_SET_DEFINED_!=_IRR_GLSL_EXT_LUMA_METER_OUTPUT_SET_DEFINED_ || _IRR_GLSL_EXT_TONE_MAPPER_PARAMETERS_BINDING_DEFINED_!=_IRR_GLSL_EXT_LUMA_METER_OUTPUT_BINDING_DEFINED_
		#error "Luma/Tonemapper SSBO Set or Binding don't match!"
	#endif

	#if _IRR_GLSL_EXT_TONE_MAPPER_INPUT_IMAGE_SET_DEFINED_!=_IRR_GLSL_EXT_LUMA_METER_INPUT_IMAGE_SET_DEFINED_ || _IRR_GLSL_EXT_TONE_MAPPER_INPUT_IMAGE_BINDING_DEFINED_!=_IRR_GLSL_EXT_LUMA_METER_INPUT_IMAGE_BINDING_DEFINED_
		#error "Input Image Set or Binding don't match!"
	#endif


	#ifndef _IRR_GLSL_EXT_TONE_MAPPER_OUTPUT_IMAGE_BINDING_DEFINED_
	#define _IRR_GLSL_EXT_TONE_MAPPER_OUTPUT_IMAGE_BINDING_DEFINED_ 3
	#endif
#else
	#ifndef _IRR_GLSL_EXT_TONE_MAPPER_OUTPUT_IMAGE_BINDING_DEFINED_
	#define _IRR_GLSL_EXT_TONE_MAPPER_OUTPUT_IMAGE_BINDING_DEFINED_ 0
	#endif
#endif

layout(local_size_x=_IRR_GLSL_EXT_TONE_MAPPER_DISPATCH_SIZE_X_DEFINED_, local_size_y=_IRR_GLSL_EXT_TONE_MAPPER_DISPATCH_SIZE_Y_DEFINED_) in;


#include "irr/builtin/glsl/colorspace/EOTF.glsl"
#include "irr/builtin/glsl/colorspace/encodeCIEXYZ.glsl"
#include "irr/builtin/glsl/colorspace/decodeCIEXYZ.glsl"
#include "irr/builtin/glsl/colorspace/OETF.glsl"


%s // _IRR_GLSL_EXT_TONE_MAPPER_USING_TEMPORAL_ADAPTATION_DEFINED_


#if defined(_IRR_GLSL_EXT_TONE_MAPPER_USING_LUMA_METER_DEFINED_)||defined(_IRR_GLSL_EXT_TONE_MAPPER_USING_TEMPORAL_ADAPTATION_DEFINED_)
	#ifndef _IRR_GLSL_EXT_TONE_MAPPER_PUSH_CONSTANTS_DEFINED_
	#define _IRR_GLSL_EXT_TONE_MAPPER_PUSH_CONSTANTS_DEFINED_
	layout(push_constant) uniform PushConstants
	{
		int currentFirstPassOutput;
	} pc;
	#endif
#endif


#ifdef _IRR_GLSL_EXT_TONE_MAPPER_USING_LUMA_METER_DEFINED_
	irr_glsl_ext_LumaMeter_PassInfo_t irr_glsl_ext_ToneMapper_getLumaMeterInfo()
	{
		return lumaPassInfo;
	}
#endif


#ifdef _IRR_GLSL_EXT_TONE_MAPPER_USING_TEMPORAL_ADAPTATION_DEFINED_
	#define _IRR_GLSL_EXT_TONE_MAPPER_PARAMETERS_QUALIFIERS restrict
#else
	#define _IRR_GLSL_EXT_TONE_MAPPER_PARAMETERS_QUALIFIERS restrict readonly
#endif

struct irr_glsl_ext_ToneMapper_input_t
{
	uint lastFrameExtraEV; // packed stuff
	uint packedExposureAdaptationFactors; // first is up, then down
	irr_glsl_ext_ToneMapper_Params_t inParams;
};

#ifndef _IRR_GLSL_EXT_TONE_MAPPER_SSBO_DESCRIPTOR_DEFINED_
#define _IRR_GLSL_EXT_TONE_MAPPER_SSBO_DESCRIPTOR_DEFINED_
layout(set=_IRR_GLSL_EXT_TONE_MAPPER_PARAMETERS_SET_DEFINED_, binding=_IRR_GLSL_EXT_TONE_MAPPER_PARAMETERS_BINDING_DEFINED_) _IRR_GLSL_EXT_TONE_MAPPER_PARAMETERS_QUALIFIERS buffer ParameterBuffer
{
	irr_glsl_ext_ToneMapper_input_t toneMapperParams;
	uvec4 padding1[15];
	#ifdef _IRR_GLSL_EXT_TONE_MAPPER_USING_LUMA_METER_DEFINED_
		irr_glsl_ext_LumaMeter_output_t lumaParams[];
	#endif
};
#endif


irr_glsl_ext_ToneMapper_Params_t irr_glsl_ext_ToneMapper_getToneMapperParams()
{
	return toneMapperParams.inParams;
}


#ifdef _IRR_GLSL_EXT_TONE_MAPPER_USING_TEMPORAL_ADAPTATION_DEFINED_
	float irr_glsl_ext_ToneMapper_getLastFrameLuma()
	{
		return unpackHalf2x16(toneMapperParams.lastFrameExtraEV)[pc.currentFirstPassOutput];
	}
	void irr_glsl_ext_ToneMapper_setLastFrameLuma(in float thisLuma)
	{
		if (all(equal(uvec3(0,0,0),gl_WorkGroupID)))
		{
			vec2 wholeVal = vec2(thisLuma,thisLuma);
			wholeVal[pc.currentFirstPassOutput] = irr_glsl_ext_ToneMapper_getLastFrameLuma();
			toneMapperParams.lastFrameExtraEV = packHalf2x16(wholeVal);
		}
	}

	float irr_glsl_ext_ToneMapper_getExposureAdaptationFactor(in float toLastLumaDiff)
	{
		return unpackHalf2x16(toneMapperParams.packedExposureAdaptationFactors)[toLastLumaDiff<0.f ? 0:1];
	}
#endif


#ifndef _IRR_GLSL_EXT_TONE_MAPPER_INPUT_IMAGE_DESCRIPTOR_DEFINED_
#define _IRR_GLSL_EXT_TONE_MAPPER_INPUT_IMAGE_DESCRIPTOR_DEFINED_
layout(set=_IRR_GLSL_EXT_TONE_MAPPER_INPUT_IMAGE_SET_DEFINED_, binding=_IRR_GLSL_EXT_TONE_MAPPER_INPUT_IMAGE_BINDING_DEFINED_) uniform sampler2DArray inputImage;
#endif

#ifndef _IRR_GLSL_EXT_TONE_MAPPER_OUTPUT_IMAGE_DESCRIPTOR_DEFINED_
#define _IRR_GLSL_EXT_TONE_MAPPER_OUTPUT_IMAGE_DESCRIPTOR_DEFINED_
layout(set=_IRR_GLSL_EXT_TONE_MAPPER_OUTPUT_IMAGE_SET_DEFINED_, binding=_IRR_GLSL_EXT_TONE_MAPPER_OUTPUT_IMAGE_BINDING_DEFINED_, %s) uniform uimage2DArray outputImage;
#endif


#ifdef _IRR_GLSL_EXT_TONE_MAPPER_USING_LUMA_METER_DEFINED_
	irr_glsl_ext_LumaMeter_output_SPIRV_CROSS_is_dumb_t irr_glsl_ext_ToneMapper_getLumaMeterOutput()
	{
		irr_glsl_ext_LumaMeter_output_SPIRV_CROSS_is_dumb_t retval;
		#define FETCH_STRUCT lumaParams[(pc.currentFirstPassOutput!=0 ? textureSize(inputImage,0).z:0)+int(gl_WorkGroupID.z)]

		#if _IRR_GLSL_EXT_LUMA_METER_MODE_DEFINED_==_IRR_GLSL_EXT_LUMA_METER_MODE_MEDIAN
			retval = FETCH_STRUCT.packedHistogram[gl_LocalInvocationIndex];
			for (int i=1; i<_IRR_GLSL_EXT_LUMA_METER_BIN_GLOBAL_REPLICATION; i++)
				retval += FETCH_STRUCT.packedHistogram[gl_LocalInvocationIndex+i*_IRR_GLSL_EXT_LUMA_METER_BIN_COUNT];
		#elif _IRR_GLSL_EXT_LUMA_METER_MODE_DEFINED_==_IRR_GLSL_EXT_LUMA_METER_MODE_GEOM_MEAN
			retval = FETCH_STRUCT.unormAverage;
		#endif

		#undef FETCH_STRUCT
		return retval;
	}
#endif


vec4 irr_glsl_ext_ToneMapper_readColor()
{
	ivec3 uv = ivec3(gl_GlobalInvocationID);
	vec4 color = texelFetch(inputImage,uv,0);
	color.rgb = %s(color.rgb);

	const mat3 xyzMatrix = %s;
	color.rgb = xyzMatrix*color.rgb;

	return color;
}

void irr_glsl_ext_ToneMapper_writeColor(in vec4 colorCIEXYZ, in vec3 ditherVal)
{
	const mat3 xyzMatrix = %s;
	const vec3 color = %s(xyzMatrix*colorCIEXYZ.rgb);

	uvec4 quantizedColor;
	%s

	ivec3 uv = ivec3(gl_GlobalInvocationID);
	imageStore(outputImage,uv,quantizedColor);
}


#ifndef _IRR_GLSL_EXT_TONE_MAPPER_IMPL_DEFINED_
#define _IRR_GLSL_EXT_TONE_MAPPER_IMPL_DEFINED_
void irr_glsl_ext_ToneMapper() // bool wgExecutionMask, then do if(any(wgExecutionMask))
{
	ivec3 uv = ivec3(gl_GlobalInvocationID);
	bool alive = any(lessThan(uv,textureSize(inputImage,0)));

	vec4 colorCIEXYZ;
	if (alive)
		colorCIEXYZ = irr_glsl_ext_ToneMapper_readColor();

	irr_glsl_ext_ToneMapper_Params_t params = irr_glsl_ext_ToneMapper_getToneMapperParams();

	float extraNegEV = 0.0;
#ifdef _IRR_GLSL_EXT_TONE_MAPPER_USING_LUMA_METER_DEFINED_
	extraNegEV = irr_glsl_ext_LumaMeter_getMeasuredLumaLog2(irr_glsl_ext_ToneMapper_getLumaMeterOutput(),irr_glsl_ext_ToneMapper_getLumaMeterInfo());
#endif
#ifdef _IRR_GLSL_EXT_TONE_MAPPER_USING_TEMPORAL_ADAPTATION_DEFINED_
	float toLastLumaDiff = irr_glsl_ext_ToneMapper_getLastFrameLuma()-extraNegEV;
	extraNegEV += toLastLumaDiff*irr_glsl_ext_ToneMapper_getExposureAdaptationFactor(toLastLumaDiff);
	irr_glsl_ext_ToneMapper_setLastFrameLuma(extraNegEV);
	#if _IRR_GLSL_EXT_TONE_MAPPER_OPERATOR_DEFINED_==_IRR_GLSL_EXT_TONE_MAPPER_REINHARD_OPERATOR
		params.keyAndManualLinearExposure *= exp2(-extraNegEV);
		colorCIEXYZ.rgb = irr_glsl_ext_ToneMapper_Reinhard(params,colorCIEXYZ.rgb);
	#elif _IRR_GLSL_EXT_TONE_MAPPER_OPERATOR_DEFINED_==_IRR_GLSL_EXT_TONE_MAPPER_ACES_OPERATOR
		params.exposure -= extraNegEV;
		colorCIEXYZ.rgb = irr_glsl_ext_ToneMapper_ACES(params,colorCIEXYZ.rgb);
	#endif
#endif

	// TODO: Add dithering
	vec3 rand = vec3(0.5);
	if (alive)
		irr_glsl_ext_ToneMapper_writeColor(colorCIEXYZ,rand);
}
#endif

#ifndef _IRR_GLSL_EXT_TONE_MAPPER_MAIN_DEFINED_
#define _IRR_GLSL_EXT_TONE_MAPPER_MAIN_DEFINED_
void main()
{
	irr_glsl_ext_ToneMapper();
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
		core::make_smart_refctd_ptr<ICPUShader>(std::move(shader),ICPUShader::buffer_contains_glsl),
		ISpecializedShader::SInfo{nullptr, nullptr, "main", asset::ISpecializedShader::ESS_COMPUTE}
	);
}

void CToneMapper::defaultBarrier()
{
	COpenGLExtensionHandler::pGlMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_PIXEL_BUFFER_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT | GL_FRAMEBUFFER_BARRIER_BIT);
}