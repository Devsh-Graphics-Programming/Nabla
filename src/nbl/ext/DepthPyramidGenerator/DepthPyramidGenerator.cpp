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
		R"(#version 460 core
#define WORKGROUP_X_AND_Y_SIZE %u
#define MIPMAP_LEVELS_PER_PASS 8u
#define MIP_IMAGE_FORMAT %s
#define %s
#define %s

layout(local_size_x = WORKGROUP_X_AND_Y_SIZE, local_size_y = WORKGROUP_X_AND_Y_SIZE) in;
 
#include <nbl/builtin/glsl/ext/DepthPyramidGenerator/depth_pyramid_generator_impl.glsl>
)";

	constexpr char* imageFormats[] =
	{
		"r16f", "r32f", "rg16f", "rg32f",
	};

	const char* format;
	switch (config.outputFormat)
	{
	case EF_R16_SFLOAT:
		format = imageFormats[0];
		break;
	case EF_R32_SFLOAT:
		format = imageFormats[1];
		break;
	case EF_R16G16_SFLOAT:
		format = imageFormats[2];
		break;
	case EF_R32G32_SFLOAT:
		format = imageFormats[3];
		break;
	default:
		assert(false);
	}

	constexpr char* redOps[] =
	{
		"REDUCION_OP_MIN", "REDUCION_OP_MAX", "REDUCION_OP_BOTH"
	};

	const char* redOp;
	switch (config.op)
	{
	case E_MIPMAP_GENERATION_OPERATOR::EMGO_MIN:
		redOp = redOps[0];
		break;
	case E_MIPMAP_GENERATION_OPERATOR::EMGO_MAX:
		redOp = redOps[1];
		break;
	case E_MIPMAP_GENERATION_OPERATOR::EMGO_BOTH:
		redOp = redOps[2];
		break;
	default:
		assert(false);
	}

	constexpr char* mipScalingOptions[] =
	{
		"STRETCH_MIN", "PAD_MAX"
	};

	// TODO: use `CGLSLCompiler::createOverridenCopy` after #68 PR merge

	const char* mipScaling = config.roundUpToPoTWithPadding ? mipScalingOptions[1] : mipScalingOptions[0];

	const uint32_t perPassMipCnt = static_cast<uint32_t>(config.workGroupSize) == 32u ? 6u : 5u;

	constexpr size_t extraSize = 32u;
	auto shaderCode = ICPUBuffer::create({ strlen(source) + extraSize + 1u });
	snprintf(reinterpret_cast<char*>(shaderCode->getPointer()), shaderCode->getSize(), source, static_cast<uint32_t>(m_config.workGroupSize), format, redOp, mipScaling);

	auto cpuSpecializedShader = core::make_smart_refctd_ptr<asset::ICPUSpecializedShader>(
		core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(shaderCode), asset::ICPUShader::buffer_contains_glsl),
		asset::ISpecializedShader::SInfo{ nullptr, nullptr, "main", asset::ISpecializedShader::ESS_COMPUTE });

	auto gpuShader = driver->createShader(core::smart_refctd_ptr<const asset::ICPUShader>(cpuSpecializedShader->getUnspecialized()));
	m_shader = driver->createSpecializedShader(gpuShader.get(), cpuSpecializedShader->getSpecializationInfo());
}

uint32_t DepthPyramidGenerator::createMipMapImages(IVideoDriver* driver, core::smart_refctd_ptr<IGPUImageView> inputDepthImageView, core::smart_refctd_ptr<IGPUImage>* outputDepthPyramidMipImages, const Config& config)
{
	VkExtent3D currMipExtent = calcLvl0MipExtent(
		inputDepthImageView->getCreationParameters().image->getCreationParameters().extent, config.roundUpToPoTWithPadding);

	const uint32_t mipmapsCnt = getMaxMipCntFromLvl0Mipextent(currMipExtent);

	if (outputDepthPyramidMipImages == nullptr)
	{
		if (config.lvlLimit == 0u)
			return mipmapsCnt;

		return std::min(config.lvlLimit, mipmapsCnt);
	}

	IGPUImage::SCreationParams imgParams;
	imgParams.flags = static_cast<IImage::E_CREATE_FLAGS>(0u);
	imgParams.type = IImage::ET_2D;
	imgParams.format = config.outputFormat;
	imgParams.mipLevels = 1u;
	imgParams.arrayLayers = 1u;
	imgParams.samples = IImage::ESCF_1_BIT;

	uint32_t i = 0u;
	while (currMipExtent.width > 0u && currMipExtent.height > 0u)
	{
		imgParams.extent = { currMipExtent.width, currMipExtent.height, 1u };
		*outputDepthPyramidMipImages = driver->createDeviceLocalGPUImageOnDedMem(IGPUImage::SCreationParams(imgParams));
		assert(*outputDepthPyramidMipImages);

		currMipExtent.width >>= 1u;
		currMipExtent.height >>= 1u;

		outputDepthPyramidMipImages++;
		i++;

		if (config.lvlLimit && i >= config.lvlLimit)
			break;
	}

	return mipmapsCnt;
}

// returns count of mip levels
uint32_t DepthPyramidGenerator::createMipMapImageViews(IVideoDriver* driver, core::smart_refctd_ptr<IGPUImageView> inputDepthImageView, core::smart_refctd_ptr<IGPUImage>* inputMipImages, core::smart_refctd_ptr<IGPUImageView>* outputMips, const Config& config)
{
	VkExtent3D currMipExtent = calcLvl0MipExtent(
		inputDepthImageView->getCreationParameters().image->getCreationParameters().extent, config.roundUpToPoTWithPadding);

	const uint32_t mipmapsCnt = getMaxMipCntFromLvl0Mipextent(currMipExtent);

	if (outputMips == nullptr)
	{
		if (config.lvlLimit == 0u)
			return mipmapsCnt;

		return std::min(config.lvlLimit, mipmapsCnt);
	}

	IGPUImageView::SCreationParams imgViewParams;
	imgViewParams.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
	imgViewParams.image = nullptr;
	imgViewParams.viewType = IGPUImageView::ET_2D;
	imgViewParams.format = config.outputFormat;
	imgViewParams.components = {};
	imgViewParams.subresourceRange = {};
	imgViewParams.subresourceRange.levelCount = 1u;
	imgViewParams.subresourceRange.layerCount = 1u;

	uint32_t i = 0u;
	while (currMipExtent.width > 0u && currMipExtent.height > 0u)
	{
		assert(*inputMipImages);
		imgViewParams.image = core::smart_refctd_ptr(*inputMipImages);
		*outputMips = driver->createImageView(IGPUImageView::SCreationParams(imgViewParams));
		assert(*outputMips);

		currMipExtent.width >>= 1u;
		currMipExtent.height >>= 1u;

		inputMipImages++;
		outputMips++;
		i++;

		if (config.lvlLimit && i >= config.lvlLimit)
			break;
	}

	return mipmapsCnt;
}

core::smart_refctd_ptr<IGPUDescriptorSetLayout> DepthPyramidGenerator::createDescriptorSetLayout(IVideoDriver* driver, const Config& config)
{
	constexpr uint32_t perPassMipCnt = 8u;

	IGPUSampler::SParams params;
	params.TextureWrapU = ISampler::ETC_CLAMP_TO_BORDER;
	params.TextureWrapV = ISampler::ETC_CLAMP_TO_BORDER;
	params.TextureWrapW = ISampler::ETC_CLAMP_TO_BORDER;
	params.MinFilter = ISampler::ETF_NEAREST;
	params.MaxFilter = ISampler::ETF_NEAREST;
	params.MipmapMode = ISampler::ESMM_NEAREST;
	params.AnisotropicFilter = 0;
	params.CompareEnable = 0;
	
	if (config.roundUpToPoTWithPadding)
	{
		switch (config.op)
		{
		case E_MIPMAP_GENERATION_OPERATOR::EMGO_MAX:
			params.BorderColor = ISampler::ETBC_FLOAT_OPAQUE_BLACK;
			break;
		case E_MIPMAP_GENERATION_OPERATOR::EMGO_MIN:
			params.BorderColor = ISampler::ETBC_FLOAT_OPAQUE_WHITE;
			break;
		case E_MIPMAP_GENERATION_OPERATOR::EMGO_BOTH:
			// TODO: fix
			params.BorderColor = ISampler::ETBC_FLOAT_OPAQUE_BLACK;
			break;
		default:
			assert(false);
		}
	}
	else
	{
		params.BorderColor = ISampler::ETBC_FLOAT_OPAQUE_BLACK;
	}
	
	auto sampler = driver->createSampler(params);
	
	IGPUDescriptorSetLayout::SBinding bindings[4];
	bindings[0].binding = 0u;
	bindings[0].count = 1u;
	bindings[0].samplers = nullptr;
	bindings[0].stageFlags = ISpecializedShader::ESS_COMPUTE;
	bindings[0].type = EDT_STORAGE_BUFFER;
	
	bindings[1].binding = 1u;
	bindings[1].count = 1u;
	bindings[1].samplers = nullptr;
	bindings[1].stageFlags = ISpecializedShader::ESS_COMPUTE;
	bindings[1].type = EDT_STORAGE_BUFFER;
	
	bindings[2].binding = 2u;
	bindings[2].count = 1u;
	bindings[2].samplers = &sampler;
	bindings[2].stageFlags = ISpecializedShader::ESS_COMPUTE;
	bindings[2].type = EDT_COMBINED_IMAGE_SAMPLER;
	
	bindings[3].binding = 3u;
	bindings[3].count = perPassMipCnt; // for convenience it's always 8, even if not all bindings are being used. compiler doesn't complain, but is it correct?
	bindings[3].samplers = nullptr;
	bindings[3].stageFlags = ISpecializedShader::ESS_COMPUTE;
	bindings[3].type = EDT_STORAGE_IMAGE;
	
	return driver->createDescriptorSetLayout(bindings, bindings + sizeof(bindings) / sizeof(IGPUDescriptorSetLayout::SBinding));
}

uint32_t DepthPyramidGenerator::createDescriptorSets(IVideoDriver* driver, core::smart_refctd_ptr<IGPUImageView> inputDepthImageView, core::smart_refctd_ptr<IGPUImageView>* inputDepthPyramidMips, 
		core::smart_refctd_ptr<IGPUDescriptorSetLayout>& inputDsLayout, core::smart_refctd_ptr<IGPUDescriptorSet>* outputDs, DispatchData* outputDispatchData, const Config& config)
{
	uint32_t mipCnt = getMaxMipCntFromImage(inputDepthImageView, config);
	if (config.lvlLimit)
	{
		if (config.lvlLimit < mipCnt) //TODO: if (config.lvlLimit < (mipCnt - 1u)) ?
			mipCnt = config.lvlLimit;
	}

	constexpr uint32_t perPassMipCnt = 8u;
	const uint32_t outputDsCnt = (mipCnt + perPassMipCnt - 1u) / perPassMipCnt;

	if (outputDs == nullptr)
		return outputDsCnt;

	assert(inputDepthPyramidMips); 

	switch (config.outputFormat)
	{
	case EF_R16_SFLOAT:
	case EF_R32_SFLOAT:
		break;
	case EF_R16G16_SFLOAT:
	case EF_R32G32_SFLOAT:
		if (config.op != E_MIPMAP_GENERATION_OPERATOR::EMGO_BOTH)
			assert(false);
		break;
	default:
		assert(false);
	}

	uint32_t virtualWorkGroupContents[2] = { 0u, 0u };
	core::smart_refctd_ptr<IGPUBuffer> virtualWorkGroup = driver->createFilledDeviceLocalBufferOnDedMem(sizeof(virtualWorkGroupContents), virtualWorkGroupContents);

	// NOTE: it is writen solely for 8 image binding limit
	core::smart_refctd_ptr<IGPUBuffer> virtualWorkGroupData;
	{
		const uint32_t maxMipCntForMainDispatch = static_cast<uint32_t>(config.workGroupSize) == 32u ? 6u : 5u;
		const uint32_t maxMipCntForVirtualDispatch = perPassMipCnt - maxMipCntForMainDispatch;
		const uint32_t lastDispatchMipLvlCnt = mipCnt % perPassMipCnt;

		uint32_t virtualDispatchCnt = ((mipCnt + perPassMipCnt - 1u) / perPassMipCnt) * 2u;
		if (lastDispatchMipLvlCnt <= maxMipCntForMainDispatch && lastDispatchMipLvlCnt != 0u)
			virtualDispatchCnt -= 1u;

		core::vector<core::vector2d<uint32_t>> virtualWorkGroupDataContents(virtualDispatchCnt);

		VkExtent3D currMipExtent = inputDepthPyramidMips[0]->getCreationParameters().image->getCreationParameters().extent;
		virtualWorkGroupDataContents[0] = core::vector2d<uint32_t>(currMipExtent.width / static_cast<uint32_t>(config.workGroupSize), currMipExtent.height / static_cast<uint32_t>(config.workGroupSize));
		outputDispatchData[0].globalWorkGroupSize = virtualWorkGroupDataContents[0];

		outputDispatchData[0].pcData.mainDispatchFirstMipExtent = { currMipExtent.width, currMipExtent.height };

		for (uint32_t i = 1u; i < virtualDispatchCnt; i++)
		{
			currMipExtent.width >>= (i % 2u == 0) ? maxMipCntForVirtualDispatch : maxMipCntForMainDispatch;
			currMipExtent.height >>= (i % 2u == 0) ? maxMipCntForVirtualDispatch : maxMipCntForMainDispatch;

			virtualWorkGroupDataContents[i].X = (currMipExtent.width + static_cast<uint32_t>(config.workGroupSize) - 1u) / static_cast<uint32_t>(config.workGroupSize);
			virtualWorkGroupDataContents[i].Y = (currMipExtent.height + static_cast<uint32_t>(config.workGroupSize) - 1u) / static_cast<uint32_t>(config.workGroupSize);

			assert(virtualWorkGroupDataContents[i].X);
			assert(virtualWorkGroupDataContents[i].Y);

			if (i % 2u == 0)
			{
				outputDispatchData[i / 2u].globalWorkGroupSize = virtualWorkGroupDataContents[i];
				outputDispatchData[i / 2u].pcData.mainDispatchFirstMipExtent = { currMipExtent.width, currMipExtent.height };
			}
			else
			{
				outputDispatchData[i / 2u].pcData.virtualDispatchFirstMipExtent = { currMipExtent.width, currMipExtent.height };
			}
		}

		for (uint32_t i = 0u; i < outputDsCnt; i++)
		{
			outputDispatchData[i].pcData.mainDispatchMipCnt = maxMipCntForMainDispatch;
			outputDispatchData[i].pcData.virtualDispatchMipCnt = maxMipCntForVirtualDispatch;
			outputDispatchData[i].pcData.maxMetaZLayerCnt = 2u;
			outputDispatchData[i].pcData.virtualDispatchIndex = i * 2u;
			outputDispatchData[i].pcData.sourceImageIsDepthOriginalDepthBuffer = 0u;
		}

		outputDispatchData[0].pcData.sourceImageIsDepthOriginalDepthBuffer = 1u;

		if (lastDispatchMipLvlCnt)
		{
			if (lastDispatchMipLvlCnt > maxMipCntForMainDispatch)
			{
				outputDispatchData[outputDsCnt - 1u].pcData.mainDispatchMipCnt = maxMipCntForMainDispatch;
				outputDispatchData[outputDsCnt - 1u].pcData.virtualDispatchMipCnt = lastDispatchMipLvlCnt - maxMipCntForMainDispatch;
			}
			else
			{
				outputDispatchData[outputDsCnt - 1u].pcData.mainDispatchMipCnt = lastDispatchMipLvlCnt;
				outputDispatchData[outputDsCnt - 1u].pcData.virtualDispatchMipCnt = 0u;
			}
		}

		if(virtualDispatchCnt % 2u)
			outputDispatchData[outputDsCnt - 1u].pcData.maxMetaZLayerCnt = 1u;

		virtualWorkGroupData = driver->createFilledDeviceLocalBufferOnDedMem(sizeof(core::vector2d<uint32_t>) * virtualWorkGroupDataContents.size(), virtualWorkGroupDataContents.data());
	}

	uint32_t mipLvlsRemaining = mipCnt;
	for (uint32_t i = 0u; i < outputDsCnt; i++)
	{
		core::smart_refctd_ptr<IGPUDescriptorSet>& currDs = *(outputDs + i);
		currDs = driver->createDescriptorSet(core::smart_refctd_ptr(inputDsLayout));

		const uint32_t thisPassMipCnt = mipLvlsRemaining > perPassMipCnt ? perPassMipCnt : mipLvlsRemaining;

		{
			IGPUDescriptorSet::SDescriptorInfo infos[3u + perPassMipCnt];
			infos[0].desc = virtualWorkGroup;
			infos[0].buffer.offset = 0u;
			infos[0].buffer.size = virtualWorkGroup->getSize();

			infos[1].desc = virtualWorkGroupData;
			infos[1].buffer.offset = 0u;
			infos[1].buffer.size = virtualWorkGroupData->getSize();

			infos[2].desc = i == 0u ? inputDepthImageView : *(inputDepthPyramidMips - 1u);
			infos[2].image.sampler = nullptr;

			

			for (uint32_t j = 3u; j < thisPassMipCnt + 3u; j++)
			{
				infos[j].desc = core::smart_refctd_ptr(*inputDepthPyramidMips);
				infos[j].image.sampler = nullptr;

				inputDepthPyramidMips++;
			}

			IGPUDescriptorSet::SWriteDescriptorSet writes[3u + perPassMipCnt];
			writes[0].dstSet = currDs.get();
			writes[0].binding = 0;
			writes[0].arrayElement = 0u;
			writes[0].count = 1u;
			writes[0].descriptorType = EDT_STORAGE_BUFFER;
			writes[0].info = &infos[0];

			writes[1].dstSet = currDs.get();
			writes[1].binding = 1;
			writes[1].arrayElement = 0u;
			writes[1].count = 1u;
			writes[1].descriptorType = EDT_STORAGE_BUFFER;
			writes[1].info = &infos[1];

			writes[2].dstSet = currDs.get();
			writes[2].binding = 2;
			writes[2].arrayElement = 0u;
			writes[2].count = 1u;
			writes[2].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
			writes[2].info = &infos[2];

			for (uint32_t j = 3; j < thisPassMipCnt + 3u; j++)
			{
				writes[j].dstSet = currDs.get();
				writes[j].binding = 3;
				writes[j].arrayElement = j - 3u;
				writes[j].count = 1u;
				writes[j].descriptorType = EDT_STORAGE_IMAGE;
				writes[j].info = &infos[j];
			}

			driver->updateDescriptorSets(3u + thisPassMipCnt, writes, 0u, nullptr);
		}

		mipLvlsRemaining -= thisPassMipCnt;
	}

	return outputDsCnt;
}

void DepthPyramidGenerator::createPipeline(IVideoDriver* driver, core::smart_refctd_ptr<IGPUDescriptorSetLayout>& dsLayout, core::smart_refctd_ptr<IGPUComputePipeline>& outputPpln)
{
	SPushConstantRange pcRange;
	pcRange.size = sizeof(nbl_glsl_depthPyramid_PushConstantsData);
	pcRange.offset = 0u;
	pcRange.stageFlags = ISpecializedShader::ESS_COMPUTE;

	outputPpln = driver->createComputePipeline(nullptr, driver->createPipelineLayout(&pcRange, &pcRange + 1, core::smart_refctd_ptr(dsLayout)), core::smart_refctd_ptr(m_shader));
}

void DepthPyramidGenerator::generateMipMaps(const core::smart_refctd_ptr<IGPUImageView>& inputImage, core::smart_refctd_ptr<IGPUComputePipeline>& ppln, core::smart_refctd_ptr<IGPUDescriptorSet>& ds, const DispatchData& dispatchData, bool issueDefaultBarrier)
{
	m_driver->bindDescriptorSets(video::EPBP_COMPUTE, ppln->getLayout(), 0u, 1u, &ds.get(), nullptr);
	m_driver->bindComputePipeline(ppln.get());
	m_driver->pushConstants(ppln->getLayout(), ISpecializedShader::ESS_COMPUTE, 0u, sizeof(nbl_glsl_depthPyramid_PushConstantsData), &dispatchData.pcData);

	m_driver->dispatch(dispatchData.globalWorkGroupSize.X, dispatchData.globalWorkGroupSize.Y, 1u);

	if (issueDefaultBarrier)
		defaultBarrier();
}

}
}
}


