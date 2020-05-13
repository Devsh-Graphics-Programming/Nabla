#include "../ext/AutoExposure/CToneMapper.h"


using namespace irr;
using namespace irr::asset;
using namespace irr::video;
using namespace ext::ToneMapper;



core::smart_refctd_ptr<CToneMapper> CToneMapper::create(IVideoDriver* _driver, asset::E_FORMAT inputFormat, const asset::IGLSLCompiler* compiler)
{
	if (!_driver)
		return nullptr;
	if (inputFormat!=asset::EF_R16G16B16A16_SFLOAT)
		return nullptr;

	std::ostringstream glsl;
	glsl << "#version 430 core\n";
	glsl << "layout (local_size_x = "<<DISPATCH_SIZE<<", local_size_y = "<<DISPATCH_SIZE<<") in;\n";
	glsl << "layout(set=3, binding=1) uniform usampler2D inImage;\n";
	glsl << "layout(set=3, binding=2, r32ui) uniform writeonly uimage2D outImage;\n";
	glsl << getInclude();
	glsl << R"===(layout(set=3, binding=0) uniform TonemappingParameters
{
	irr_ext_Autoexposure_ReinhardParams params;
};

void main()
{
	//uint data = textureGatherOffset(); // later optimization

	ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(uv,textureSize(inImage,0))))
		return;

	uvec2 data = texelFetch(inImage,uv,0).rg;
	vec3 hdr = vec3(unpackHalf2x16(data[0]).rg,unpackHalf2x16(data[1])[0]);
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

bool CToneMapper::tonemap(video::IGPUImageView* inputThatsInTheSet, video::IGPUDescriptorSet* set, uint32_t parameterUBOOffset)
{
	const auto& params = inputThatsInTheSet->getCreationParameters();
	if (params.format!=viewFormat)
		return false;
	if (params.image->getCreationParameters().format!=format)
		return false;

	auto offsets = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<uint32_t> >(1u);
	offsets->operator[](0u) = parameterUBOOffset;

	m_driver->bindComputePipeline(computePipeline.get());
	m_driver->bindDescriptorSets(video::EPBP_COMPUTE,pipelineLayout.get(),3,1,const_cast<const video::IGPUDescriptorSet**>(&set),&offsets);

	auto imgViewSize = params.image->getMipSize(params.subresourceRange.baseMipLevel);
	imgViewSize /= DISPATCH_SIZE;
	m_driver->dispatch(imgViewSize.x,imgViewSize.y,1u);
	return true;
}