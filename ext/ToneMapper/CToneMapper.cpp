#include "../ext/AutoExposure/CToneMapper.h"


using namespace irr;
using namespace irr::asset;
using namespace irr::video;
using namespace ext::ToneMapper;

#if 0

core::smart_refctd_ptr<CToneMapper> CToneMapper::create(IVideoDriver* _driver, asset::E_FORMAT inputFormat, const asset::IGLSLCompiler* compiler)
{
	if (!_driver)
		return nullptr;
	if (inputFormat!=asset::EF_R16G16B16A16_SFLOAT)
		return nullptr;

	constexpr char* formatSrc = 
R"===(#version 430 core


#define _IRR_GLSL_EXT_TONE_MAPPER_OPERATOR_DEFINED_ %d

#include "irr/builtin/glsl/ext/ToneMapper/operators.glsl"


#define _IRR_GLSL_EXT_LUMA_METER_MIN_LUMA_DEFINED_ %d
#define _IRR_GLSL_EXT_LUMA_METER_MAX_LUMA_DEFINED_ %d

#define _IRR_GLSL_EXT_LUMA_METER_MODE_DEFINED_ %d

#include "irr/builtin/glsl/ext/LumaMeter/common.glsl"

layout(local_size_x=_IRR_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_DEFINED_, local_size_y=_IRR_GLSL_EXT_LUMA_METER_DISPATCH_SIZE_DEFINED_) in;

// TODO: wrap
layout(set=3, binding=0) uniform TonemappingParameters
{
	irr_glsl_ext_ToneMapper_Params_t params;
};


#include "irr/builtin/glsl/colorspace/EOTF.glsl"
#include "irr/builtin/glsl/colorspace/encodeCIEXYZ.glsl"
#include "irr/builtin/glsl/colorspace/decodeCIEXYZ.glsl"
#include "irr/builtin/glsl/colorspace/OETF.glsl"


#ifndef _IRR_GLSL_EXT_TONE_MAPPER_INPUT_IMAGE_SET_DEFINED_
#define _IRR_GLSL_EXT_TONE_MAPPER_INPUT_IMAGE_SET_DEFINED_ 0
#endif

#ifndef _IRR_GLSL_EXT_TONE_MAPPER_INPUT_IMAGE_BINDING_DEFINED_
#define _IRR_GLSL_EXT_TONE_MAPPER_INPUT_IMAGE_BINDING_DEFINED_ 1
#endif

#ifndef _IRR_GLSL_EXT_TONE_MAPPER_OUTPUT_IMAGE_SET_DEFINED_
#define _IRR_GLSL_EXT_TONE_MAPPER_OUTPUT_IMAGE_SET_DEFINED_ 0
#endif

#ifndef _IRR_GLSL_EXT_TONE_MAPPER_OUTPUT_IMAGE_BINDING_DEFINED_
#define _IRR_GLSL_EXT_TONE_MAPPER_OUTPUT_IMAGE_BINDING_DEFINED_ 3
#endif

#ifndef _IRR_GLSL_EXT_TONE_MAPPER_IMAGES_DEFINED_
#define _IRR_GLSL_EXT_TONE_MAPPER_IMAGES_DEFINED_
layout(set=_IRR_GLSL_EXT_TONE_MAPPER_INPUT_IMAGE_SET_DEFINED_, binding=_IRR_GLSL_EXT_TONE_MAPPER_INPUT_IMAGE_BINDING_DEFINED_) sampler2DArray inputImage;
layout(set=_IRR_GLSL_EXT_TONE_MAPPER_OUTPUT_IMAGE_SET_DEFINED_, binding=_IRR_GLSL_EXT_TONE_MAPPER_OUTPUT_IMAGE_BINDING_DEFINED_, %s) %simage2DArray outputImage;
#endif


#ifndef _IRR_GLSL_EXT_LUMA_METER_OUTPUT_SET_DEFINED_
#define _IRR_GLSL_EXT_LUMA_METER_OUTPUT_SET_DEFINED_ 0
#endif

#ifndef _IRR_GLSL_EXT_LUMA_METER_OUTPUT_BINDING_DEFINED_
#define _IRR_GLSL_EXT_LUMA_METER_OUTPUT_BINDING_DEFINED_ 2
#endif

#ifndef _IRR_GLSL_EXT_LUMA_METER_OUTPUT_DEFINED_
#define _IRR_GLSL_EXT_LUMA_METER_OUTPUT_DEFINED_
layout(set=_IRR_GLSL_EXT_LUMA_METER_OUTPUT_SET_DEFINED_, binding=_IRR_GLSL_EXT_LUMA_METER_OUTPUT_BINDING_DEFINED_) restrict readonly buffer PreviousPassBuffer
{
	irr_glsl_ext_LumaMeter_output_t outParams[_IRR_GLSL_EXT_LUMA_METER_LAYERS_TO_PROCESS_DEFINED_];
};
#endif

void main()
{
	//uint data = textureGatherOffset(); // later optimization

	ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(uv,textureSize(inImage,0))))
		return;

	uvec2 data = texelFetch(inImage,uv,0).rg;
	vec3 hdr = vec3(unpackHalf2x16(data[0]).rg,unpackHalf2x16(data[1])[0]);

	vec4 ldr = vec4(irr_ext_Autoexposure_ToneMapReinhard(params,,irr_glsl_ext_LumaMeter_getMeasuredLumaLog2(,)),1.0);
	vec4 ldr = vec4(irr_ext_Autoexposure_ToneMapReinhard(params,hdr),1.0);
	// TODO: Add dithering

	imageStore(outImage,uv,uvec4(packUnorm4x8(ldr),0u,0u,0u));
}
)===";
	auto spirv = compiler->createSPIRVFromGLSL(glsl.str().c_str(),asset::ISpecializedShader::ESS_COMPUTE,"main","CToneMapper");
	auto shader = _driver->createGPUShader(std::move(spirv));
	
	asset::ISpecializedShader::SInfo specInfo(nullptr, nullptr, "main", asset::ISpecializedShader::ESS_COMPUTE);

	_driver->createGPUSpecializedShader(shader.get(),std::move(specInfo));
}

CToneMapper::CToneMapper(	IVideoDriver* _driver, asset::E_FORMAT inputFormat,
							core::smart_refctd_ptr<video::IGPUDescriptorSetLayout>&& _dsLayout,
							core::smart_refctd_ptr<video::IGPUPipelineLayout>&& _pipelineLayout,
							core::smart_refctd_ptr<video::IGPUComputePipeline>&& _computePipeline) :
									m_driver(_driver), format(inputFormat), dsLayout(std::move(_dsLayout)),
									pipelineLayout(std::move(_pipelineLayout)), computePipeline(std::move(_computePipeline))
{
	if (format==asset::EF_R16G16B16A16_SFLOAT)
		viewFormat = asset::EF_R32G32_UINT;
}
#endif