#include "nbl/video/sampling/CEnvmapWarpGenerator.h"
#include "nbl/builtin/hlsl/sampling/hierarchical_image/common.hlsl"
#include "nlohmann/detail/input/parser.hpp"

using namespace nbl;
using namespace core;
using namespace video;
using namespace system;
using namespace asset;
using namespace hlsl;
using namespace nbl::hlsl::sampling::hierarchical_image;

namespace nbl::video
{

class CEnvmapWarpGenerator;

namespace
{
	constexpr std::string_view NBL_WORKING_DIRECTORY = "nbl/builtin/hlsl/sampling/hierarchical_image/";

	core::smart_refctd_ptr<IGPUImageView> createTexture(video::ILogicalDevice* device, const asset::VkExtent3D extent, E_FORMAT format, uint32_t mipLevels = 1u, uint32_t layers = 0u)
	{
		const auto realLayers = layers ? layers:1u;

		IGPUImage::SCreationParams imgParams;
		imgParams.extent = extent;
		imgParams.arrayLayers = realLayers;
		imgParams.flags = static_cast<IImage::E_CREATE_FLAGS>(0);
		imgParams.format = format;
		imgParams.mipLevels = mipLevels;
		imgParams.samples = IImage::ESCF_1_BIT;
		imgParams.type = IImage::ET_2D;
		imgParams.usage = IImage::EUF_STORAGE_BIT | IImage::EUF_TRANSFER_SRC_BIT | IImage::EUF_TRANSFER_DST_BIT | IImage::EUF_SAMPLED_BIT;
		const auto image = device->createImage(std::move(imgParams));
		auto imageMemReqs = image->getMemoryReqs();
		imageMemReqs.memoryTypeBits &= device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
		device->allocate(imageMemReqs, image.get());

		IGPUImageView::SCreationParams viewparams;
		viewparams.subUsages = IImage::EUF_STORAGE_BIT | IImage::EUF_SAMPLED_BIT;
		viewparams.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0);
		viewparams.format = format;
		viewparams.image = std::move(image);
		viewparams.viewType = layers ? IGPUImageView::ET_2D_ARRAY:IGPUImageView::ET_2D;
		viewparams.subresourceRange.aspectMask = IImage::EAF_COLOR_BIT;
		viewparams.subresourceRange.baseArrayLayer = 0u;
		viewparams.subresourceRange.layerCount = realLayers;
		viewparams.subresourceRange.baseMipLevel = 0u;
		viewparams.subresourceRange.levelCount = mipLevels;

		return device->createImageView(std::move(viewparams));
	}

	core::smart_refctd_ptr<IShader> getShaderSource( asset::IAssetManager* assetManager, std::string_view filePath, system::ILogger* logger)
	{
		IAssetLoader::SAssetLoadParams lparams = {};
		lparams.logger = logger;
		lparams.workingDirectory = NBL_WORKING_DIRECTORY;
		const auto filePathStr = std::string(filePath);
		auto bundle = assetManager->getAsset(filePathStr, lparams);
		if (bundle.getContents().empty() || bundle.getAssetType()!=IAsset::ET_SHADER)
		{
			const auto assetType = bundle.getAssetType();
			logger->log("Shader %s not found!", ILogger::ELL_ERROR, filePathStr);
			exit(-1);
		}
		auto firstAssetInBundle = bundle.getContents()[0];
		return smart_refctd_ptr_static_cast<IShader>(firstAssetInBundle);
	}
}

core::smart_refctd_ptr<CEnvmapWarpGenerator> CEnvmapWarpGenerator::create(SCreationParameters&& params)
{
	auto* const logger = params.utilities->getLogger();

	if (!params.validate())
	{
		logger->log("Failed creation parameters validation!", ILogger::ELL_ERROR);
		return nullptr;
	}

	const auto device = params.utilities->getLogicalDevice();

	ConstructorParams constructorParams;

	const auto pipelineLayout = createPipelineLayout(device);

	constructorParams.genLumaPipeline = createPipeline(params, pipelineLayout.get(), "gen_luma.comp.hlsl");
	constructorParams.genWarpPipeline = createPipeline(params, pipelineLayout.get(), "gen_warp.comp.hlsl");

	const auto descriptorPool = device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT, pipelineLayout->getDescriptorSetLayouts());
	const auto descriptorSet = descriptorPool->createDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(pipelineLayout->getDescriptorSetLayouts()[0]));

	constructorParams.creationParams = std::move(params);

	return core::smart_refctd_ptr<CEnvmapWarpGenerator>(new CEnvmapWarpGenerator(std::move(constructorParams)));
}

core::smart_refctd_ptr<video::IGPUImageView> CEnvmapWarpGenerator::createLumaMap(video::ILogicalDevice* device, asset::VkExtent3D extent, uint32_t mipCount, uint32_t layerCount, const std::string_view debugName)
{
	return createTexture(device, extent, EF_R32_SFLOAT, mipCount, layerCount);
}

core::smart_refctd_ptr<video::IGPUImageView> CEnvmapWarpGenerator::createWarpMap(video::ILogicalDevice* device, asset::VkExtent3D extent, uint32_t layerCount, const std::string_view debugName)
{
	return createTexture(device, extent, EF_R32G32_SFLOAT, 1u, layerCount);
}

core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> CEnvmapWarpGenerator::createDescriptorSetLayout(video::ILogicalDevice* device)
{
	const IGPUDescriptorSetLayout::SBinding bindings[] = {
		// Gen luma input
		{
			.binding = 0u,
			.type = nbl::asset::IDescriptor::E_TYPE::ET_SAMPLED_IMAGE,
			.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
			.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
			.count = 1u
		},
		// Gen luma output
		{
			.binding = 1u,
			.type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
			.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
			.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
			.count = 1u
		},
		// Gen warp input
		{
			.binding = 2u,
			.type = nbl::asset::IDescriptor::E_TYPE::ET_SAMPLED_IMAGE,
			.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
			.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
			.count = 1u,
		},
		// Gen warp output
		{
			.binding = 3u,
			.type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
			.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
			.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
			.count = 1u
		}
	};
	return device->createDescriptorSetLayout(bindings);
}

core::smart_refctd_ptr<video::IGPUPipelineLayout> CEnvmapWarpGenerator::createPipelineLayout(video::ILogicalDevice* device)
{
	const auto dsLayout = createDescriptorSetLayout(device);
	asset::SPushConstantRange pcRange = {
	  .stageFlags = hlsl::ESS_COMPUTE, 
		.offset = 0,
		.size = std::max<uint32_t>(sizeof(SLumaGenPushConstants), sizeof(SWarpGenPushConstants))
	};
	return device->createPipelineLayout({ &pcRange, 1 }, dsLayout);
}

core::smart_refctd_ptr<video::IGPUComputePipeline> CEnvmapWarpGenerator::createPipeline(const SCreationParameters& params, const video::IGPUPipelineLayout* layout, std::string_view shaderPath)
{
	system::logger_opt_ptr logger = params.utilities->getLogger();
	auto system = smart_refctd_ptr<ISystem>(params.assetManager->getSystem());
	auto* device = params.utilities->getLogicalDevice();

	const auto shaderSource = getShaderSource(params.assetManager.get(), shaderPath, logger.get());
	auto compiler = make_smart_refctd_ptr<asset::CHLSLCompiler>(smart_refctd_ptr(system));
	CHLSLCompiler::SOptions options = {};
	options.stage = IShader::E_SHADER_STAGE::ESS_COMPUTE;
	options.preprocessorOptions.targetSpirvVersion = device->getPhysicalDevice()->getLimits().spirvVersion;
	options.spirvOptimizer = nullptr;

#ifndef _NBL_DEBUG
		ISPIRVOptimizer::E_OPTIMIZER_PASS optPasses = ISPIRVOptimizer::EOP_STRIP_DEBUG_INFO;
		auto opt = make_smart_refctd_ptr<ISPIRVOptimizer>(std::span<ISPIRVOptimizer::E_OPTIMIZER_PASS>(&optPasses, 1));
		options.spirvOptimizer = opt.get();
#else
		options.debugInfoFlags |= IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_LINE_BIT;
#endif
	options.preprocessorOptions.sourceIdentifier = shaderSource->getFilepathHint();
	options.preprocessorOptions.logger = logger.get();
	options.preprocessorOptions.includeFinder = compiler->getDefaultIncludeFinder();

	const auto overridenUnspecialized = compiler->compileToSPIRV(static_cast<const char*>(shaderSource->getContent()->getPointer()), options);
	const auto shader = device->compileShader({ overridenUnspecialized.get() });
	if (!shader)
	{
		logger.log("Could not compile shaders!", ILogger::ELL_ERROR);
		return nullptr;
	}

	video::IGPUComputePipeline::SCreationParams pipelineParams[1] = {};
	pipelineParams[0].layout = layout;
	pipelineParams[0].shader = { .shader = shader.get(), .entryPoint = "main" };

	smart_refctd_ptr<IGPUComputePipeline> pipeline;
	params.utilities->getLogicalDevice()->createComputePipelines(nullptr, pipelineParams, &pipeline);
	if (!pipeline)
	{
		logger.log("Could not create pipeline!", ILogger::ELL_ERROR);
		return nullptr;
	}

	return pipeline;
}

core::smart_refctd_ptr<CEnvmapWarpGenerator::SSession> CEnvmapWarpGenerator::createSession(core::smart_refctd_ptr<IGPUImageView>&& envMap, uint16_t upscaleLog2)
{

	const auto device = m_params.utilities->getLogicalDevice();

	SSession::SCachedCreationParams sessionParams;
	sessionParams.generator = core::smart_refctd_ptr<CEnvmapWarpGenerator>(this);
	sessionParams.envMap = std::move(envMap);

	const auto& envmapParams = sessionParams.envMap->getCreationParameters().image->getCreationParameters();
	const auto envmapExtent = envmapParams.extent;
	const auto envmapLayers = envmapParams.arrayLayers;

	// we don't need the 1x1 mip for anything
	const uint32_t mipCountLuminance = IImage::calculateFullMipPyramidLevelCount(envmapExtent,IImage::ET_2D) - 1;
	const auto envmapPotExtent = [mipCountLuminance, envmapLayers]() -> asset::VkExtent3D
	{
		const uint32_t width = 0x1u << mipCountLuminance;
		return { width, width >> 1u, envmapLayers };
	}();
	auto calcWorkgroupSize = [envmapLayers](const asset::VkExtent3D extent, const uint32_t workgroupDimension) -> uint32_t2
	{
		return uint32_t2(extent.width - 1, extent.height - 1) / workgroupDimension + uint32_t2(envmapLayers);
	};

	sessionParams.genLumaWorkgroupCount = calcWorkgroupSize(envmapPotExtent, GenLumaWorkgroupDim);
	sessionParams.lumaMap = createLumaMap(device, envmapPotExtent, mipCountLuminance, envmapLayers);

	const asset::VkExtent3D warpMapExtent = {envmapPotExtent.width << upscaleLog2, envmapPotExtent.height << upscaleLog2, envmapPotExtent.depth };
	sessionParams.genWarpWorkgroupCount = calcWorkgroupSize(warpMapExtent, GenWarpWorkgroupDim);
  sessionParams.warpMap = createWarpMap(device, warpMapExtent, envmapLayers);

	const auto dsLayouts = m_genLumaPipeline->getLayout()->getDescriptorSetLayouts();
	const auto descriptorPool = device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT, dsLayouts);
	sessionParams.descriptorSet = descriptorPool->createDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(dsLayouts[0]));

	IGPUDescriptorSet::SDescriptorInfo envMapDescriptorInfo; 
	envMapDescriptorInfo.desc = sessionParams.envMap;
	envMapDescriptorInfo.info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;

	IGPUDescriptorSet::SDescriptorInfo lumaMapGeneralDescriptorInfo;
  lumaMapGeneralDescriptorInfo.desc = sessionParams.lumaMap;
	lumaMapGeneralDescriptorInfo.info.image.imageLayout = IImage::LAYOUT::GENERAL;

	IGPUDescriptorSet::SDescriptorInfo lumaMapReadDescriptorInfo;
	lumaMapReadDescriptorInfo.desc = sessionParams.lumaMap;
	lumaMapReadDescriptorInfo.info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;

	IGPUDescriptorSet::SDescriptorInfo warpMapDescriptorInfo;
	warpMapDescriptorInfo.desc = sessionParams.warpMap;
	warpMapDescriptorInfo.info.image.imageLayout = IImage::LAYOUT::GENERAL;

	auto* dsPtr = sessionParams.descriptorSet.get();
	const IGPUDescriptorSet::SWriteDescriptorSet writes[] = {
		{
			.dstSet = dsPtr, .binding = 0, .count = 1, .info = &envMapDescriptorInfo
		},
		{
			.dstSet = dsPtr, .binding = 1, .count = 1, .info = &lumaMapGeneralDescriptorInfo
		},
		{
			.dstSet = dsPtr, .binding = 2, .count = 1, .info = &lumaMapReadDescriptorInfo
		},
		{
			.dstSet = dsPtr, .binding = 3, .count = 1, .info = &warpMapDescriptorInfo
		},
	};
	device->updateDescriptorSets(writes, {});

	sessionParams.layerCount = envmapLayers;

	return make_smart_refctd_ptr<SSession>(std::move(sessionParams));
}

void CEnvmapWarpGenerator::SSession::computeWarpMap(video::IGPUCommandBuffer* cmdBuf)
{
	const auto lumaMapImage = m_params.lumaMap->getCreationParameters().image.get();
	const auto lumaMapMipLevels = lumaMapImage->getCreationParameters().mipLevels;
	const auto lumaMapExtent = lumaMapImage->getCreationParameters().extent;

	const auto warpMapImage = m_params.warpMap->getCreationParameters().image.get();
	const auto warpMapExtent = warpMapImage->getCreationParameters().extent;

	const auto* genLumaPipeline = m_params.generator->getGenLumaPipeline();
	const auto* genWarpPipeline = m_params.generator->getGenWarpPipeline();

  cmdBuf->bindDescriptorSets(EPBP_COMPUTE, genLumaPipeline->getLayout(),
    0, 1, &m_params.descriptorSet.get());

	// Gen Luma Map
	{
		IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t barriers[] = {
			{
				.barrier = {
					.dep = {
						.srcStageMask = PIPELINE_STAGE_FLAGS::NONE,
						.srcAccessMask = ACCESS_FLAGS::NONE,
						.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
						.dstAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS
					}
				},
				.image = lumaMapImage,
				.subresourceRange = {
					.aspectMask = IImage::EAF_COLOR_BIT,
					.baseMipLevel = 0u,
					.levelCount = 1,
					.baseArrayLayer = 0u,
					.layerCount = IGPUImageView::remaining_array_layers
				},
				.oldLayout = IImage::LAYOUT::UNDEFINED,
				.newLayout = IImage::LAYOUT::GENERAL,
			},
		};

		SLumaGenPushConstants pcData = {};
		pcData.lumaMapWidth = lumaMapExtent.width;
		pcData.lumaMapHeight = lumaMapExtent.height;

		cmdBuf->bindComputePipeline(genLumaPipeline);
		cmdBuf->pushConstants(genLumaPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE,
			0, sizeof(SLumaGenPushConstants), &pcData);
		cmdBuf->dispatch(m_params.genLumaWorkgroupCount.x, m_params.genLumaWorkgroupCount.y, m_params.layerCount);
	}

  // Generate luminance mip map
	{
		IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t barriers[] = {
			{
				.barrier = {
					.dep = {
						.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
						.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
						.dstStageMask = PIPELINE_STAGE_FLAGS::BLIT_BIT,
						.dstAccessMask = ACCESS_FLAGS::TRANSFER_READ_BIT
					}
				},
				.image = lumaMapImage,
				.subresourceRange = {
					.aspectMask = IImage::EAF_COLOR_BIT,
					.baseMipLevel = 0u,
					.levelCount = 1,
					.baseArrayLayer = 0u,
					.layerCount = IGPUImageView::remaining_array_layers,
				},
				.oldLayout = IImage::LAYOUT::GENERAL,
				.newLayout = IImage::LAYOUT::TRANSFER_SRC_OPTIMAL,
			},
			{
				.barrier = {
					.dep = {
						.srcStageMask = PIPELINE_STAGE_FLAGS::NONE,
						.srcAccessMask = ACCESS_FLAGS::NONE,
						.dstStageMask = PIPELINE_STAGE_FLAGS::BLIT_BIT,
						.dstAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT
					}
				},
				.image = lumaMapImage,
				.subresourceRange = {
					.aspectMask = IImage::EAF_COLOR_BIT,
					.baseMipLevel = 1u,
					.levelCount = lumaMapMipLevels - 1,
					.baseArrayLayer = 0u,
					.layerCount = IGPUImageView::remaining_array_layers
				},
				.oldLayout = IImage::LAYOUT::UNDEFINED,
				.newLayout = IImage::LAYOUT::TRANSFER_DST_OPTIMAL,
			}
		};                
		cmdBuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = barriers });
		const auto mipLevels = lumaMapMipLevels;
		const auto extent = lumaMapExtent;
		auto* image = lumaMapImage;
		for (uint32_t srcMip_i = 0; srcMip_i < mipLevels-1; srcMip_i++)
		{
			const IGPUCommandBuffer::SImageBlit blit = {
				.srcMinCoord = {0, 0, 0},
				.srcMaxCoord = {extent.width >> (srcMip_i), extent.height >> (srcMip_i), 1},
				.dstMinCoord = {0, 0, 0},
				.dstMaxCoord = {extent.width >> srcMip_i + 1, extent.height >> (srcMip_i + 1), 1},
				.layerCount = IGPUImageView::remaining_array_layers,
				.srcBaseLayer = 0,
				.dstBaseLayer = 0,
				.srcMipLevel = srcMip_i,
				.dstMipLevel = srcMip_i + 1,
				.aspectMask = IGPUImage::E_ASPECT_FLAGS::EAF_COLOR_BIT,
			};
			cmdBuf->blitImage(image, IImage::LAYOUT::TRANSFER_SRC_OPTIMAL, image, IImage::LAYOUT::TRANSFER_DST_OPTIMAL, { &blit, 1 }, IGPUSampler::E_TEXTURE_FILTER::ETF_LINEAR);

			// last mip no need to transition
			if (srcMip_i + 1 == mipLevels - 1) break;
			
			IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t barrier = {
				.barrier = {
					.dep = {
						.srcStageMask = PIPELINE_STAGE_FLAGS::BLIT_BIT,
						.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT,
						.dstStageMask = PIPELINE_STAGE_FLAGS::BLIT_BIT,
						.dstAccessMask = ACCESS_FLAGS::TRANSFER_READ_BIT
					}
				},
				.image = image,
				.subresourceRange = {
					.aspectMask = IImage::EAF_COLOR_BIT,
					.baseMipLevel = srcMip_i + 1,
					.levelCount = 1,
					.baseArrayLayer = 0u,
					.layerCount = IGPUImageView::remaining_array_layers
				},
				.oldLayout = IImage::LAYOUT::TRANSFER_DST_OPTIMAL,
				.newLayout = IImage::LAYOUT::TRANSFER_SRC_OPTIMAL,
			};               
			cmdBuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = {&barrier, 1} });

		}
	}

	// Gen Warp Map
  {
		IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t barriers[] = {
			{
				.barrier = {
					.dep = {
						.srcStageMask = PIPELINE_STAGE_FLAGS::BLIT_BIT,
						.srcAccessMask = ACCESS_FLAGS::TRANSFER_READ_BIT,
						.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
						.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS
					}
				},
				.image = lumaMapImage,
				.subresourceRange = {
					.aspectMask = IImage::EAF_COLOR_BIT,
					.baseMipLevel = 0u,
					.levelCount = lumaMapMipLevels - 1,
					.baseArrayLayer = 0u,
					.layerCount = IGPUImageView::remaining_array_layers,
				},
				.oldLayout = IImage::LAYOUT::TRANSFER_SRC_OPTIMAL,
				.newLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL,
			},
			{
				.barrier = {
					.dep = {
						.srcStageMask = PIPELINE_STAGE_FLAGS::BLIT_BIT,
						.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT,
						.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
						.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS
					}
				},
				.image = lumaMapImage,
				.subresourceRange = {
					.aspectMask = IImage::EAF_COLOR_BIT,
					.baseMipLevel = lumaMapMipLevels - 1,
					.levelCount = 1,
					.baseArrayLayer = 0u,
					.layerCount = IGPUImageView::remaining_array_layers
				},
				.oldLayout = IImage::LAYOUT::TRANSFER_DST_OPTIMAL,
				.newLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL,
			},
			{
				.barrier = {
					.dep = {
						.srcStageMask = PIPELINE_STAGE_FLAGS::NONE,
						.srcAccessMask = ACCESS_FLAGS::NONE,
						.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
						.dstAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS
					}
				},
				.image = warpMapImage,
				.subresourceRange = {
					.aspectMask = IImage::EAF_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0u,
					.layerCount = IGPUImageView::remaining_array_layers
				},
				.oldLayout = IImage::LAYOUT::UNDEFINED,
				.newLayout = IImage::LAYOUT::GENERAL,
			}
		};                
		cmdBuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = barriers });

		const SWarpGenPushConstants pcData = {
		  .lumaMapWidth = lumaMapExtent.width,
			.lumaMapHeight = lumaMapExtent.height,
			.warpMapWidth = warpMapExtent.width,
			.warpMapHeight = warpMapExtent.height
		};
		cmdBuf->bindComputePipeline(genWarpPipeline);
		cmdBuf->pushConstants(genWarpPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE,
			0, sizeof(SWarpGenPushConstants), &pcData);
		cmdBuf->dispatch(m_params.genWarpWorkgroupCount.x, m_params.genWarpWorkgroupCount.y, m_params.layerCount);
	}
}

CEnvmapWarpGenerator::SSession::image_barrier_t CEnvmapWarpGenerator::SSession::getEnvMapPrevBarrier(core::bitflag<asset::PIPELINE_STAGE_FLAGS> srcStageMask, core::bitflag<asset::ACCESS_FLAGS> srcAccessMask, IGPUImage::LAYOUT oldLayout)
{
	return image_barrier_t{
    .barrier = {
      .dep = {
        .srcStageMask = srcStageMask,
        .srcAccessMask = srcAccessMask,
        .dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
        .dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS
      }
    },
    .image = m_params.envMap->getCreationParameters().image.get(),
    .subresourceRange = {
      .aspectMask = IImage::EAF_COLOR_BIT,
      .baseMipLevel = 0u,
      .levelCount = IImageViewBase::remaining_mip_levels,
      .baseArrayLayer = 0u,
      .layerCount = 1u
    },
    .oldLayout = oldLayout,
    .newLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL,
	};
}

CEnvmapWarpGenerator::SSession::image_barrier_t CEnvmapWarpGenerator::SSession::getEnvMapNextBarrier(core::bitflag<asset::PIPELINE_STAGE_FLAGS> dstStageMask, core::bitflag<asset::ACCESS_FLAGS> dstAccessMask, IGPUImage::LAYOUT newLayout)
{
	return image_barrier_t{
		.barrier = {
      .dep = {
        .srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
        .srcAccessMask = ACCESS_FLAGS::SHADER_READ_BITS,
        .dstStageMask = dstStageMask,
        .dstAccessMask = dstAccessMask      }
		},
    .image = m_params.envMap->getCreationParameters().image.get(),
    .subresourceRange = {
      .aspectMask = IImage::EAF_COLOR_BIT,
      .baseMipLevel = 0u,
      .levelCount = IImageViewBase::remaining_mip_levels,
      .baseArrayLayer = 0u,
      .layerCount = 1u
    },
    .oldLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL,
    .newLayout = newLayout,
	};
}

std::array<CEnvmapWarpGenerator::SSession::image_barrier_t, 2> CEnvmapWarpGenerator::SSession::getOutputMapNextBarrier(core::bitflag<asset::PIPELINE_STAGE_FLAGS> dstStageMask, core::bitflag<asset::ACCESS_FLAGS> dstAccessMask, IGPUImage::LAYOUT newLayout)
{
	return {
		image_barrier_t {
      .barrier = {
        .dep = {
          .srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
          .srcAccessMask = ACCESS_FLAGS::SHADER_READ_BITS,
          .dstStageMask = dstStageMask,
          .dstAccessMask = dstAccessMask      }
      },
      .image = m_params.lumaMap->getCreationParameters().image.get(),
      .subresourceRange = {
        .aspectMask = IImage::EAF_COLOR_BIT,
        .baseMipLevel = 0u,
        .levelCount = IImageViewBase::remaining_mip_levels,
        .baseArrayLayer = 0u,
        .layerCount = IImageViewBase::remaining_array_layers,
      },
      .oldLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL,
      .newLayout = newLayout,
		},
		image_barrier_t {
      .barrier = {
        .dep = {
          .srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
          .srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
          .dstStageMask = dstStageMask,
          .dstAccessMask = dstAccessMask      }
      },
      .image = m_params.warpMap->getCreationParameters().image.get(),
      .subresourceRange = {
        .aspectMask = IImage::EAF_COLOR_BIT,
        .baseMipLevel = 0u,
        .levelCount = IImageViewBase::remaining_mip_levels,
        .baseArrayLayer = 0u,
        .layerCount = IImageViewBase::remaining_array_layers,
      },
      .oldLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL,
      .newLayout = newLayout,
		}
	};
}

}
