// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include <nbl/ext/DepthPyramidGenerator/DepthPyramidGenerator.h>

using namespace nbl;
using namespace nbl::core;
using namespace nbl::asset;
using namespace nbl::video;

namespace nbl
{
namespace ext
{
namespace DepthPyramidGenerator
{

DepthPyramidGenerator::DepthPyramidGenerator(IVideoDriver* driver, IAssetManager* am, core::smart_refctd_ptr<IGPUImageView> inputDepthImageView,
	core::smart_refctd_ptr<IGPUImageView>* outputDepthPyramidMips,
	const Config& config)
	: m_driver(driver)
{
	// TODO: complete supported formats
	switch (config.outputFormat)
	{
	case EF_R16_SFLOAT:
	case EF_R32_SFLOAT:
		break;
	case EF_R32G32_SFLOAT:
	case EF_R16G16_SFLOAT:
		if (config.op != E_MIPMAP_GENERATION_OPERATOR::BOTH)
			assert(false);
		break;
	default:
		assert(false);
	}


	core::smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout;
	{
		IGPUSampler::SParams params;
		params.TextureWrapU = ISampler::ETC_MIRROR;
		params.TextureWrapV = ISampler::ETC_MIRROR;
		params.TextureWrapW = ISampler::ETC_MIRROR;
		params.BorderColor = ISampler::ETBC_FLOAT_OPAQUE_BLACK;
		params.MinFilter = ISampler::ETF_NEAREST;
		params.MaxFilter = ISampler::ETF_NEAREST;
		params.MipmapMode = ISampler::ESMM_NEAREST;
		params.AnisotropicFilter = 0;
		params.CompareEnable = 0;
		auto sampler = m_driver->createGPUSampler(params);

		// create pipeline
		IGPUDescriptorSetLayout::SBinding bindings[2];
		bindings[0].binding = 0u;
		bindings[0].count = 1u;
		bindings[0].samplers = &sampler;
		bindings[0].stageFlags = ISpecializedShader::ESS_COMPUTE;
		bindings[0].type = EDT_COMBINED_IMAGE_SAMPLER;

		bindings[1].binding = 1u;
		bindings[1].count = 1u;
		bindings[1].samplers = nullptr;
		bindings[1].stageFlags = ISpecializedShader::ESS_COMPUTE;
		bindings[1].type = EDT_STORAGE_IMAGE;

		dsLayout = driver->createGPUDescriptorSetLayout(bindings, bindings + sizeof(bindings) / sizeof(IGPUDescriptorSetLayout::SBinding));
		m_ds = driver->createGPUDescriptorSet(core::smart_refctd_ptr(dsLayout));
	}

	configureMipImages(inputDepthImageView, outputDepthPyramidMips, config);

	{
		IGPUDescriptorSet::SDescriptorInfo infos[2];
		infos[0].desc = inputDepthImageView;
		infos[0].image.sampler = nullptr;

		infos[1].desc = core::smart_refctd_ptr(*outputDepthPyramidMips);
		infos[1].image.sampler = nullptr;

		IGPUDescriptorSet::SWriteDescriptorSet writes[2];
		writes[0].dstSet = m_ds.get();
		writes[0].binding = 0;
		writes[0].arrayElement = 0u;
		writes[0].count = 1u;
		writes[0].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
		writes[0].info = &infos[0];

		writes[1].dstSet = m_ds.get();
		writes[1].binding = 1;
		writes[1].arrayElement = 0u;
		writes[1].count = 1u;
		writes[1].descriptorType = EDT_STORAGE_IMAGE;
		writes[1].info = &infos[1];

		m_driver->updateDescriptorSets(sizeof(writes) / sizeof(IGPUDescriptorSet::SWriteDescriptorSet), writes, 0u, nullptr);
	}

	// create extra shader code
	constexpr uint32_t maxExtraCodeSize = 256u;
	std::string defines;
	defines.reserve(maxExtraCodeSize);
	{
		switch (config.workGroupSize)
		{
		case E_WORK_GROUP_SIZE::E32x32x1:
			defines += std::string("#define WORKGROUP_X_AND_Y_SIZE 32\n");
			break;
		case E_WORK_GROUP_SIZE::E16x16x1:
			defines += std::string("#define WORKGROUP_X_AND_Y_SIZE 16\n");
			break;
		}

		// TODO: complete supported formats
		switch (config.outputFormat)
		{
		case EF_R16_SFLOAT:
			defines += std::string("#define MIP_IMAGE_FORMAT r16f\n");
			break;
		case EF_R32_SFLOAT:
			defines += std::string("#define MIP_IMAGE_FORMAT r32f\n");
			break;
		case EF_R16G16_SFLOAT:
			defines += std::string("#define MIP_IMAGE_FORMAT r16g16f\n");
			break;
		case EF_R32G32_SFLOAT:
			defines += std::string("#define MIP_IMAGE_FORMAT r32g32f\n");
			break;
		default:
			assert(false);
		}

		if (!config.roundUpToPoTWithPadding)
			defines += std::string("#define STRETCH_MIN\n");
	}

	asset::IAssetLoader::SAssetLoadParams lp;
	auto shader = IAsset::castDown<ICPUSpecializedShader>(*am->getAsset("../../../include/nbl/builtin/glsl/ext/DepthPyramidGenerator/depth_pyramid_generator.comp", lp).getContents().begin());
	assert(shader);
	const asset::ICPUShader* unspec = shader->getUnspecialized();
	assert(unspec->containsGLSL());

	//	override shader code
	auto begin = reinterpret_cast<const char*>(unspec->getSPVorGLSL()->getPointer());
	const std::string_view origSource(begin, unspec->getSPVorGLSL()->getSize());

	const size_t firstNewlineAfterVersion = origSource.find("\n", origSource.find("#version "));
	assert(firstNewlineAfterVersion != std::string_view::npos);
	const std::string_view sourceWithoutVersion(begin + firstNewlineAfterVersion, origSource.size() - firstNewlineAfterVersion);

	std::string newSource("#version 460 core\n");
	newSource += defines;
	newSource += sourceWithoutVersion;

	auto unspecNew = core::make_smart_refctd_ptr<asset::ICPUShader>(newSource.c_str());
	auto specinfo = shader->getSpecializationInfo();

	auto newSpecShader = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(std::move(unspecNew), std::move(specinfo));
	auto gpuShader = driver->getGPUObjectsFromAssets(&newSpecShader.get(), &newSpecShader.get() + 1u)->begin()[0];

	m_ppln = m_driver->createGPUComputePipeline(nullptr, m_driver->createGPUPipelineLayout(nullptr, nullptr, core::smart_refctd_ptr(dsLayout)), std::move(gpuShader));
}


void DepthPyramidGenerator::generateMipMaps()
{
	m_driver->bindDescriptorSets(video::EPBP_COMPUTE, m_ppln->getLayout(), 0u, 1u, &m_ds.get(), nullptr);
	m_driver->bindComputePipeline(m_ppln.get());

	m_driver->dispatch(m_globalWorkGroupSize.X, m_globalWorkGroupSize.Y, 1u);
}

inline VkExtent3D calcLvl0MipExtent(const VkExtent3D& sourceImageExtent, bool roundUpToPoTWithPadding)
{
	VkExtent3D lvl0MipExtent;

	lvl0MipExtent.width = core::roundUpToPoT(sourceImageExtent.width);
	lvl0MipExtent.height = core::roundUpToPoT(sourceImageExtent.height);

	if (!roundUpToPoTWithPadding)
	{
		if (!core::isPoT(sourceImageExtent.width))
			lvl0MipExtent.width >>= 1u;
		if (!core::isPoT(sourceImageExtent.height))
			lvl0MipExtent.height >>= 1u;
	}

	return lvl0MipExtent;
}

void DepthPyramidGenerator::configureMipImages(core::smart_refctd_ptr<IGPUImageView> inputDepthImageView, core::smart_refctd_ptr<IGPUImageView>* outputDepthPyramidMips, const Config& config)
{
	VkExtent3D currMipExtent = calcLvl0MipExtent(
		inputDepthImageView->getCreationParameters().image->getCreationParameters().extent, config.roundUpToPoTWithPadding);

	m_globalWorkGroupSize = vector2du32_SIMD(currMipExtent.width / static_cast<uint32_t>(config.workGroupSize), currMipExtent.height / static_cast<uint32_t>(config.workGroupSize));
	assert(m_globalWorkGroupSize.x > 0u && m_globalWorkGroupSize.y > 0u);

	IGPUImage::SCreationParams imgParams;
	imgParams.flags = static_cast<IImage::E_CREATE_FLAGS>(0u);
	imgParams.type = IImage::ET_2D;
	imgParams.format = config.outputFormat;
	imgParams.mipLevels = 1u;
	imgParams.arrayLayers = 1u;
	imgParams.samples = IImage::ESCF_1_BIT;

	IGPUImageView::SCreationParams imgViewParams;
	imgViewParams.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
	imgViewParams.image = nullptr;
	imgViewParams.viewType = IGPUImageView::ET_2D;
	imgViewParams.format = config.outputFormat;
	imgViewParams.components = {};
	imgViewParams.subresourceRange = {};
	imgViewParams.subresourceRange.levelCount = 1u;
	imgViewParams.subresourceRange.layerCount = 1u;

	while (currMipExtent.width > 0u && currMipExtent.height > 0u)
	{
		core::smart_refctd_ptr<IGPUImage> image;

		imgParams.extent = { currMipExtent.width, currMipExtent.height, 1u };
		image = m_driver->createDeviceLocalGPUImageOnDedMem(IGPUImage::SCreationParams(imgParams));
		assert(image);

		imgViewParams.image = std::move(image);
		*outputDepthPyramidMips = m_driver->createGPUImageView(IGPUImageView::SCreationParams(imgViewParams));
		assert(*outputDepthPyramidMips);

		currMipExtent.width >>= 1u;
		currMipExtent.height >>= 1u;

		outputDepthPyramidMips++;

		// tmp
		break;
	}
}

}
}
}


