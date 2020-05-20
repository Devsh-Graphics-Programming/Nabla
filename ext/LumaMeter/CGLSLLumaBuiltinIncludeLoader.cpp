#include "../ext/LumaMeter/CGLSLLumaBuiltinIncludeLoader.h"


using namespace irr;
using namespace irr::asset;
using namespace irr::video;
using namespace ext::LumaMeter;


core::SRange<const asset::SPushConstantRange> CGLSLLumaBuiltinIncludeLoader::getDefaultPushConstantRanges()
{
	static const asset::SPushConstantRange range =
	{
		ISpecializedShader::ESS_COMPUTE,
		0u,
		sizeof(uint32_t)
	};
	return {&range,&range+1};
}

core::SRange<const IGPUDescriptorSetLayout::SBinding> CGLSLLumaBuiltinIncludeLoader::getDefaultBindings(IVideoDriver* driver)
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
		sampler = driver->createGPUSampler(params);
	}
	return {bnd,bnd+sizeof(bnd)/sizeof(IGPUDescriptorSetLayout::SBinding)};
}