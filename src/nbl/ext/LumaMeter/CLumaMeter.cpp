// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/LumaMeter/CLumaMeter.h"
#include "../../../../source/Nabla/COpenGLExtensionHandler.h"

#include <cstdio>

using namespace nbl;
using namespace nbl::asset;
using namespace nbl::video;
using namespace ext::LumaMeter;



core::SRange<const asset::SPushConstantRange> CLumaMeter::getDefaultPushConstantRanges()
{
	static const asset::SPushConstantRange range =
	{
		ISpecializedShader::ESS_COMPUTE,
		0u,
		sizeof(uint32_t)
	};
	return {&range,&range+1};
}

core::SRange<const video::IGPUDescriptorSetLayout::SBinding> CLumaMeter::getDefaultBindings(video::IVideoDriver* driver)
{
	static core::smart_refctd_ptr<IGPUSampler> sampler;
	static const IGPUDescriptorSetLayout::SBinding bnd[] =
	{
		{
			0u,
			EDT_UNIFORM_BUFFER_DYNAMIC,
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
			&sampler
		}
	};
	if (!sampler)
	{
		IGPUSampler::SParams params =
		{
			{
				ISampler::ETC_CLAMP_TO_EDGE,
				ISampler::ETC_CLAMP_TO_EDGE,
				ISampler::ETC_CLAMP_TO_EDGE,
				ISampler::ETBC_FLOAT_OPAQUE_BLACK,
				ISampler::ETF_LINEAR,
				ISampler::ETF_LINEAR,
				ISampler::ESMM_NEAREST,
				0u,
				0u,
				ISampler::ECO_ALWAYS
			}
		};
		sampler = driver->createSampler(params);
	}
	return {bnd,bnd+sizeof(bnd)/sizeof(IGPUDescriptorSetLayout::SBinding)};
}

core::smart_refctd_ptr<asset::ICPUSpecializedShader> CLumaMeter::createShader(
	asset::CGLSLCompiler* compilerToAddBuiltinIncludeTo,
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


#ifndef _NBL_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_X_DEFINED_
#define _NBL_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_X_DEFINED_ 16
#endif

#ifndef _NBL_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_Y_DEFINED_
#define _NBL_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_Y_DEFINED_ 16
#endif

#define _NBL_GLSL_WORKGROUP_SIZE_ (_NBL_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_X_DEFINED_*_NBL_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_Y_DEFINED_)

#define _NBL_GLSL_EXT_LUMA_METER_BIN_COUNT %d
#define _NBL_GLSL_EXT_LUMA_METER_BIN_GLOBAL_REPLICATION %d


#define _NBL_GLSL_EXT_LUMA_METER_MIN_LUMA_DEFINED_ %d
#define _NBL_GLSL_EXT_LUMA_METER_MAX_LUMA_DEFINED_ %d

#ifndef _NBL_GLSL_EXT_LUMA_METER_MODE_DEFINED_
#define _NBL_GLSL_EXT_LUMA_METER_MODE_DEFINED_ %d
#endif

#include "nbl/builtin/glsl/colorspace/EOTF.glsl"
#include "nbl/builtin/glsl/colorspace/encodeCIEXYZ.glsl"
#include "nbl/builtin/glsl/colorspace/decodeCIEXYZ.glsl"
#include "nbl/builtin/glsl/colorspace/OETF.glsl"

#define _NBL_GLSL_EXT_LUMA_METER_EOTF_DEFINED_ %s
#define _NBL_GLSL_EXT_LUMA_METER_XYZ_CONVERSION_MATRIX_DEFINED_ %s
#define _NBL_GLSL_EXT_LUMA_METER_GET_COLOR_DEFINED_
#include "nbl/builtin/glsl/ext/LumaMeter/impl.glsl"


layout(local_size_x=_NBL_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_X_DEFINED_, local_size_y=_NBL_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_Y_DEFINED_) in;



layout(set=_NBL_GLSL_EXT_LUMA_METER_UNIFORMS_SET_DEFINED_, binding=_NBL_GLSL_EXT_LUMA_METER_UNIFORMS_BINDING_DEFINED_) uniform Uniforms
{
	nbl_glsl_ext_LumaMeter_Uniforms_t inParams;
};


vec3 nbl_glsl_ext_LumaMeter_getColor(bool wgExecutionMask)
{
	vec3 retval;
	if (wgExecutionMask)
	{
		vec2 uv = vec2(gl_GlobalInvocationID.xy)*inParams.meteringWindowScale+inParams.meteringWindowOffset;
		retval = textureLod(inputImage,vec3(uv,float(gl_GlobalInvocationID.z)),0.0).rgb;
	}
	return retval;
}


void main()
{
	nbl_glsl_ext_LumaMeter(true);
}
)===";

	constexpr char* eotf = "nbl_glsl_eotf_identity";
	constexpr char* xyzMatrices[ECP_COUNT] = 
	{
		"nbl_glsl_sRGBtoXYZ",
		"nbl_glsl_Display_P3toXYZ",
		"nbl_glsl_DCI_P3toXYZ",
		"nbl_glsl_BT2020toXYZ",
		"nbl_glsl_AdobeRGBtoXYZ",
		"nbl_glsl_ACES2065_1toXYZ",
		"nbl_glsl_ACEScctoXYZ",
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
		DEFAULT_BIN_COUNT,DEFAULT_BIN_GLOBAL_REPLICATION,
		reinterpret_cast<const int32_t&>(minLuma),reinterpret_cast<const int32_t&>(maxLuma),meterMode,
		eotf,xyzMatrix
	);

	return core::make_smart_refctd_ptr<ICPUSpecializedShader>(
		core::make_smart_refctd_ptr<ICPUShader>(std::move(shader),ICPUShader::buffer_contains_glsl),
		ISpecializedShader::SInfo{nullptr, nullptr, "main", asset::ISpecializedShader::ESS_COMPUTE}
	);
}

void CLumaMeter::defaultBarrier()
{
	COpenGLExtensionHandler::pGlMemoryBarrier(GL_UNIFORM_BARRIER_BIT|GL_SHADER_STORAGE_BARRIER_BIT|GL_BUFFER_UPDATE_BARRIER_BIT);
}