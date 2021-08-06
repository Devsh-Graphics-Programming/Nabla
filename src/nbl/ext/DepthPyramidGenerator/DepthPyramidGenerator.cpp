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
		const Config& config)
	: m_driver(driver), m_config(config)
{
	const char* source =
		R"(#version 430 core
#define WORKGROUP_X_AND_Y_SIZE %u
#define MIP_IMAGE_FORMAT %s
#define STRETCH_MIN

layout(local_size_x = WORKGROUP_X_AND_Y_SIZE, local_size_y = WORKGROUP_X_AND_Y_SIZE) in;
 
#include "../../../include/nbl/builtin/glsl/ext/DepthPyramidGenerator/depth_pyramid_generator_impl.glsl"
)";

	constexpr char* imageFormats[] =
	{
		"r16f", "r32f", "r16g16f", "r32g32f"
	};

	const char* format;
	switch (config.outputFormat)
	{
	case E_IMAGE_FORMAT::EIF_R16_FLOAT:
		format = imageFormats[0];
		break;
	case E_IMAGE_FORMAT::EIF_R32_FLOAT:
		format = imageFormats[1];
		break;
	case E_IMAGE_FORMAT::EIF_R16G16_FLOAT:
		format = imageFormats[2];
		break;
	case E_IMAGE_FORMAT::EIF_R32G32_FLOAT:
		format = imageFormats[3];
		break;
	default:
		assert(false);
	}

	constexpr size_t extraSize = 16u;
	auto shaderCode = core::make_smart_refctd_ptr<ICPUBuffer>(strlen(source) + extraSize + 1u);
	snprintf(reinterpret_cast<char*>(shaderCode->getPointer()), shaderCode->getSize(), source, static_cast<uint32_t>(m_config.workGroupSize), format);

	auto cpuSpecializedShader = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(
		core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(shaderCode), asset::ICPUShader::buffer_contains_glsl),
		asset::ISpecializedShader::SInfo{ nullptr, nullptr, "main", asset::ISpecializedShader::ESS_COMPUTE });

	auto gpuShader = driver->createGPUShader(core::smart_refctd_ptr<const asset::ICPUShader>(cpuSpecializedShader->getUnspecialized()));
	m_shader = driver->createGPUSpecializedShader(gpuShader.get(), cpuSpecializedShader->getSpecializationInfo());

}

// returns count of mip levels
uint32_t DepthPyramidGenerator::createMipMapImageViews(IVideoDriver* driver, core::smart_refctd_ptr<IGPUImageView> inputDepthImageView, core::smart_refctd_ptr<IGPUImageView>* outputDepthPyramidMips, const Config& config)
{
	VkExtent3D currMipExtent = calcLvl0MipExtent(
		inputDepthImageView->getCreationParameters().image->getCreationParameters().extent, config.roundUpToPoTWithPadding);

	// TODO: `calcLvl0MipExtent` will be called second time here, fix it
	const uint32_t mipmapsCnt = getMaxMipCntFromImage(inputDepthImageView, config.roundUpToPoTWithPadding);

	if (outputDepthPyramidMips == nullptr)
	{
		if (config.lvlLimit == 0u)
			return mipmapsCnt;

		return std::min(config.lvlLimit, mipmapsCnt);
	}

	IGPUImage::SCreationParams imgParams;
	imgParams.flags = static_cast<IImage::E_CREATE_FLAGS>(0u);
	imgParams.type = IImage::ET_2D;
	imgParams.format = static_cast<E_FORMAT>(config.outputFormat);
	imgParams.mipLevels = 1u;
	imgParams.arrayLayers = 1u;
	imgParams.samples = IImage::ESCF_1_BIT;

	IGPUImageView::SCreationParams imgViewParams;
	imgViewParams.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
	imgViewParams.image = nullptr;
	imgViewParams.viewType = IGPUImageView::ET_2D;
	imgViewParams.format = static_cast<E_FORMAT>(config.outputFormat);
	imgViewParams.components = {};
	imgViewParams.subresourceRange = {};
	imgViewParams.subresourceRange.levelCount = 1u;
	imgViewParams.subresourceRange.layerCount = 1u;

	while (currMipExtent.width > 0u && currMipExtent.height > 0u)
	{
		core::smart_refctd_ptr<IGPUImage> image;

		imgParams.extent = { currMipExtent.width, currMipExtent.height, 1u };
		image = driver->createDeviceLocalGPUImageOnDedMem(IGPUImage::SCreationParams(imgParams));
		assert(image);

		imgViewParams.image = std::move(image);
		*outputDepthPyramidMips = driver->createGPUImageView(IGPUImageView::SCreationParams(imgViewParams));
		assert(*outputDepthPyramidMips);

		currMipExtent.width >>= 1u;
		currMipExtent.height >>= 1u;

		outputDepthPyramidMips++;

		// tmp
		break;
	}

	return mipmapsCnt;
}

void DepthPyramidGenerator::createPipeline(
	IVideoDriver* driver, core::smart_refctd_ptr<IGPUImageView> inputDepthImageView, core::smart_refctd_ptr<IGPUImageView>* inputDepthPyramidMips,
	core::smart_refctd_ptr<IGPUDescriptorSet>& outputDs, core::smart_refctd_ptr<IGPUComputePipeline>& outputPpln, const Config& config)
{
	// TODO: complete supported formats
	switch (config.outputFormat)
	{
	case E_IMAGE_FORMAT::EIF_R16_FLOAT:
	case E_IMAGE_FORMAT::EIF_R32_FLOAT:
		break;
	case E_IMAGE_FORMAT::EIF_R16G16_FLOAT:
	case E_IMAGE_FORMAT::EIF_R32G32_FLOAT:
		if (config.op != E_MIPMAP_GENERATION_OPERATOR::EMGO_BOTH)
			assert(false);
		break;
	default:
		assert(false);
	}


	core::smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout;
	{
		IGPUSampler::SParams params;
		params.TextureWrapU = ISampler::ETC_CLAMP_TO_BORDER;
		params.TextureWrapV = ISampler::ETC_CLAMP_TO_BORDER;
		params.TextureWrapW = ISampler::ETC_CLAMP_TO_BORDER;
		params.BorderColor = ISampler::ETBC_FLOAT_OPAQUE_BLACK;
		params.MinFilter = ISampler::ETF_NEAREST;
		params.MaxFilter = ISampler::ETF_NEAREST;
		params.MipmapMode = ISampler::ESMM_NEAREST;
		params.AnisotropicFilter = 0;
		params.CompareEnable = 0;
		auto sampler = driver->createGPUSampler(params);

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
		outputDs = driver->createGPUDescriptorSet(core::smart_refctd_ptr(dsLayout));
	}

	{
		IGPUDescriptorSet::SDescriptorInfo infos[2];
		infos[0].desc = inputDepthImageView;
		infos[0].image.sampler = nullptr;

		infos[1].desc = core::smart_refctd_ptr(*inputDepthPyramidMips);
		infos[1].image.sampler = nullptr;

		IGPUDescriptorSet::SWriteDescriptorSet writes[2];
		writes[0].dstSet = outputDs.get();
		writes[0].binding = 0;
		writes[0].arrayElement = 0u;
		writes[0].count = 1u;
		writes[0].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
		writes[0].info = &infos[0];

		writes[1].dstSet = outputDs.get();
		writes[1].binding = 1;
		writes[1].arrayElement = 0u;
		writes[1].count = 1u;
		writes[1].descriptorType = EDT_STORAGE_IMAGE;
		writes[1].info = &infos[1];

		driver->updateDescriptorSets(sizeof(writes) / sizeof(IGPUDescriptorSet::SWriteDescriptorSet), writes, 0u, nullptr);
	}

	outputPpln = driver->createGPUComputePipeline(nullptr, driver->createGPUPipelineLayout(nullptr, nullptr, core::smart_refctd_ptr(dsLayout)), core::smart_refctd_ptr(m_shader));
}

void DepthPyramidGenerator::generateMipMaps(const core::smart_refctd_ptr<IGPUImageView>& inputImage, core::smart_refctd_ptr<IGPUComputePipeline>& ppln, core::smart_refctd_ptr<IGPUDescriptorSet>& ds, bool issueDefaultBarrier)
{
	const VkExtent3D lvl0MipExtent = calcLvl0MipExtent(inputImage->getCreationParameters().image->getCreationParameters().extent, m_config.roundUpToPoTWithPadding);

	const vector2du32_SIMD globalWorkGroupSize = vector2du32_SIMD(lvl0MipExtent.width / static_cast<uint32_t>(m_config.workGroupSize), lvl0MipExtent.height / static_cast<uint32_t>(m_config.workGroupSize));
	assert(m_globalWorkGroupSize.x > 0u && m_globalWorkGroupSize.y > 0u);

	m_driver->bindDescriptorSets(video::EPBP_COMPUTE, ppln->getLayout(), 0u, 1u, &ds.get(), nullptr);
	m_driver->bindComputePipeline(ppln.get());

	m_driver->dispatch(globalWorkGroupSize.X, globalWorkGroupSize.Y, 1u);

	if (issueDefaultBarrier)
		defaultBarrier();
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

}
}
}


