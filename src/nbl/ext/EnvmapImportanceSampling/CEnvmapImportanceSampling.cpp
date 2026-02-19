#include "nbl/ext/EnvmapImportanceSampling/CEnvmapImportanceSampling.h"
#include "nbl/ext/EnvmapImportanceSampling/builtin/hlsl/common.hlsl"
#include "nlohmann/detail/input/parser.hpp"

using namespace nbl::hlsl::ext::envmap_importance_sampling;

#ifdef NBL_EMBED_BUILTIN_RESOURCES
#include "nbl/ext/debug_draw/builtin/build/CArchive.h"
#endif

using namespace nbl;
using namespace core;
using namespace video;
using namespace system;
using namespace asset;
using namespace hlsl;

namespace nbl::ext::envmap_importance_sampling
{

namespace
{
  constexpr std::string_view NBL_EXT_MOUNT_ENTRY = "nbl/ext/EnvmapImportanceSampling";

  // image must have the first mip layout set to transfer src, and the rest to dst
  void generateMipmap(video::IGPUCommandBuffer* cmdBuf, IGPUImage* image)
  {
    const auto mipLevels = image->getCreationParameters().mipLevels;
    const auto extent = image->getCreationParameters().extent;
    for (uint32_t srcMip_i = 0; srcMip_i < mipLevels-1; srcMip_i++)
    {
      
      const IGPUCommandBuffer::SImageBlit blit = {
        .srcMinCoord = {0, 0, 0},
        .srcMaxCoord = {extent.width >> (srcMip_i), extent.height >> (srcMip_i), 1},
        .dstMinCoord = {0, 0, 0},
        .dstMaxCoord = {extent.width >> srcMip_i + 1, extent.height >> srcMip_i + 1, 1},
        .layerCount = 1,
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
          .layerCount = 1u
        },
        .oldLayout = IImage::LAYOUT::TRANSFER_DST_OPTIMAL,
        .newLayout = IImage::LAYOUT::TRANSFER_SRC_OPTIMAL,
      };               
      cmdBuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = {&barrier, 1} });

    }
  }

  core::smart_refctd_ptr<IGPUImageView> createTexture(video::ILogicalDevice* device, const asset::VkExtent3D extent, E_FORMAT format, uint32_t mipLevels = 1u, uint32_t layers = 0u)
  {
    const auto real_layers = layers ? layers:1u;

    IGPUImage::SCreationParams imgParams;
    imgParams.extent = extent;
    imgParams.arrayLayers = real_layers;
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
    viewparams.subresourceRange.layerCount = real_layers;
    viewparams.subresourceRange.baseMipLevel = 0u;
    viewparams.subresourceRange.levelCount = mipLevels;

    return device->createImageView(std::move(viewparams));
  }

  core::smart_refctd_ptr<IShader> getShaderSource( asset::IAssetManager* assetManager, const char* filePath, system::ILogger* logger)
  {
    IAssetLoader::SAssetLoadParams lparams = {};
    lparams.logger = logger;
    lparams.workingDirectory = NBL_EXT_MOUNT_ENTRY;
    auto bundle = assetManager->getAsset(filePath, lparams);
    if (bundle.getContents().empty() || bundle.getAssetType()!=IAsset::ET_SHADER)
    {
      const auto assetType = bundle.getAssetType();
      logger->log("Shader %s not found!", ILogger::ELL_ERROR, filePath);
      exit(-1);
    }
    auto firstAssetInBundle = bundle.getContents()[0];
    return smart_refctd_ptr_static_cast<IShader>(firstAssetInBundle);
  }
}

core::smart_refctd_ptr<EnvmapImportanceSampling> EnvmapImportanceSampling::create(SCreationParameters&& params)
{
	auto* const logger = params.utilities->getLogger();

	if (!params.validate())
	{
		logger->log("Failed creation parameters validation!", ILogger::ELL_ERROR);
		return nullptr;
	}

	const auto EnvmapExtent = params.envMap->getCreationParameters().image->getCreationParameters().extent;
	// we don't need the 1x1 mip for anything
	const uint32_t MipCountLuminance = IImage::calculateFullMipPyramidLevelCount(EnvmapExtent,IImage::ET_2D)-1;
	const auto EnvMapPoTExtent = [MipCountLuminance]() -> asset::VkExtent3D
	{
		const uint32_t width = 0x1u<<MipCountLuminance;
		return { width,width>>1u,1u };
	}();
	auto calcWorkgroupSize = [](const asset::VkExtent3D extent, const uint32_t workgroupDimension) -> uint32_t2
	{
    return uint32_t2(extent.width - 1, extent.height - 1) / workgroupDimension + uint32_t2(1);
	};

  const auto device = params.utilities->getLogicalDevice();

  ConstructorParams constructorParams;
  
  constructorParams.lumaWorkgroupCount = calcWorkgroupSize(EnvMapPoTExtent, params.genLumaMapWorkgroupDimension);
  constructorParams.lumaMap = createLumaMap(device, EnvMapPoTExtent, MipCountLuminance);

	const auto upscale = 0;
	const asset::VkExtent3D WarpMapExtent = {EnvMapPoTExtent.width<<upscale,EnvMapPoTExtent.height<<upscale,EnvMapPoTExtent.depth};
  constructorParams.warpWorkgroupCount = calcWorkgroupSize(WarpMapExtent, params.genWarpMapWorkgroupDimension);
  constructorParams.warpMap = createWarpMap(device, WarpMapExtent);

  const auto genLumaPipelineLayout = createGenLumaPipelineLayout(device);
  constructorParams.genLumaPipeline = createGenLumaPipeline(params, genLumaPipelineLayout.get());
  const auto genLumaDescriptorPool = device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT, genLumaPipelineLayout->getDescriptorSetLayouts());
  const auto genLumaDescriptorSet = genLumaDescriptorPool->createDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(genLumaPipelineLayout->getDescriptorSetLayouts()[0]));

  const auto genWarpPipelineLayout = createGenWarpPipelineLayout(device);
  constructorParams.genWarpPipeline = createGenWarpPipeline(params, genWarpPipelineLayout.get());
  const auto genWarpDescriptorPool = device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT, genWarpPipelineLayout->getDescriptorSetLayouts());
  const auto genWarpDescriptorSet = genWarpDescriptorPool->createDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(genWarpPipelineLayout->getDescriptorSetLayouts()[0]));

  IGPUDescriptorSet::SDescriptorInfo envMapDescriptorInfo; 
  envMapDescriptorInfo.desc = params.envMap;
  envMapDescriptorInfo.info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;

  IGPUDescriptorSet::SDescriptorInfo lumaMapGeneralDescriptorInfo;
  lumaMapGeneralDescriptorInfo.desc = constructorParams.lumaMap;
  lumaMapGeneralDescriptorInfo.info.image.imageLayout = IImage::LAYOUT::GENERAL;

  IGPUDescriptorSet::SDescriptorInfo lumaMapReadDescriptorInfo;
  lumaMapReadDescriptorInfo.desc = constructorParams.lumaMap;
  lumaMapReadDescriptorInfo.info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;

  IGPUDescriptorSet::SDescriptorInfo warpMapDescriptorInfo;
  warpMapDescriptorInfo.desc = constructorParams.warpMap;
  warpMapDescriptorInfo.info.image.imageLayout = IImage::LAYOUT::GENERAL;

  const IGPUDescriptorSet::SWriteDescriptorSet writes[] = {
    {
      .dstSet = genLumaDescriptorSet.get(), .binding = 0, .count = 1, .info = &envMapDescriptorInfo
    },
    {
      .dstSet = genLumaDescriptorSet.get(), .binding = 1, .count = 1, .info = &lumaMapGeneralDescriptorInfo
    },
    {
      .dstSet = genWarpDescriptorSet.get(), .binding = 0, .count = 1, .info = &lumaMapReadDescriptorInfo
    },
    {
      .dstSet = genWarpDescriptorSet.get(), .binding = 1, .count = 1, .info = &warpMapDescriptorInfo
    },
  };

  device->updateDescriptorSets(writes, {});

  constructorParams.genLumaDescriptorSet = genLumaDescriptorSet;
  constructorParams.genWarpDescriptorSet = genWarpDescriptorSet;

  constructorParams.creationParams = std::move(params);

  return core::smart_refctd_ptr<EnvmapImportanceSampling>(new EnvmapImportanceSampling(std::move(constructorParams)));
}

core::smart_refctd_ptr<video::IGPUImageView> EnvmapImportanceSampling::createLumaMap(video::ILogicalDevice* device, asset::VkExtent3D extent, uint32_t mipCount, const std::string_view debugName)
{
  return createTexture(device, extent, EF_R32_SFLOAT, mipCount);
}

core::smart_refctd_ptr<video::IGPUImageView> EnvmapImportanceSampling::createWarpMap(video::ILogicalDevice* device, asset::VkExtent3D extent, const std::string_view debugName)
{
  return createTexture(device, extent, EF_R32G32_SFLOAT);
}

smart_refctd_ptr<IFileArchive> EnvmapImportanceSampling::mount(core::smart_refctd_ptr<ILogger> logger, ISystem* system, video::ILogicalDevice* device, const std::string_view archiveAlias)
{
  assert(system);

	if (!system)
		return nullptr;

	// extension should mount everything for you, regardless if content goes from virtual filesystem 
	// or disk directly - and you should never rely on application framework to expose extension data
	#ifdef NBL_EMBED_BUILTIN_RESOURCES
	auto archive = make_smart_refctd_ptr<builtin::build::CArchive>(smart_refctd_ptr(logger));
	#else
	auto archive = make_smart_refctd_ptr<nbl::system::CMountDirectoryArchive>(std::string_view(NBL_ENVMAP_IMPORTANCE_SAMPLING_HLSL_MOUNT_POINT), smart_refctd_ptr(logger), system);
	#endif

	system->mount(smart_refctd_ptr(archive), archiveAlias.data());
	return smart_refctd_ptr(archive);
}

core::smart_refctd_ptr<video::IGPUComputePipeline> EnvmapImportanceSampling::createGenLumaPipeline(const SCreationParameters& params, const video::IGPUPipelineLayout* pipelineLayout)
{
	system::logger_opt_ptr logger = params.utilities->getLogger();
	auto system = smart_refctd_ptr<ISystem>(params.assetManager->getSystem());
	auto* device = params.utilities->getLogicalDevice();
  mount(smart_refctd_ptr<ILogger>(params.utilities->getLogger()), system.get(), params.utilities->getLogicalDevice(), NBL_EXT_MOUNT_ENTRY);

  const auto shaderSource = getShaderSource(params.assetManager.get(), "gen_luma.comp.hlsl", logger.get());
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

  const auto workgroupDimStr = std::to_string(params.genLumaMapWorkgroupDimension);
  const IShaderCompiler::SMacroDefinition defines[] = {
    { "WORKGROUP_DIM", workgroupDimStr.data() },
  };

  options.preprocessorOptions.extraDefines = defines;

  const auto overridenUnspecialized = compiler->compileToSPIRV((const char*)shaderSource->getContent()->getPointer(), options);
  const auto shader = device->compileShader({ overridenUnspecialized.get() });
	if (!shader)
	{
		logger.log("Could not compile shaders!", ILogger::ELL_ERROR);
		return nullptr;
	}

  video::IGPUComputePipeline::SCreationParams pipelineParams[1] = {};
  pipelineParams[0].layout = pipelineLayout;
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

core::smart_refctd_ptr<video::IGPUComputePipeline> EnvmapImportanceSampling::createGenWarpPipeline(const SCreationParameters& params, const video::IGPUPipelineLayout* pipelineLayout)
{
	system::logger_opt_ptr logger = params.utilities->getLogger();
	auto system = smart_refctd_ptr<ISystem>(params.assetManager->getSystem());
	auto* device = params.utilities->getLogicalDevice();
  mount(smart_refctd_ptr<ILogger>(params.utilities->getLogger()), system.get(), params.utilities->getLogicalDevice(), NBL_EXT_MOUNT_ENTRY);

  const auto shaderSource = getShaderSource(params.assetManager.get(), "gen_warp.comp.hlsl", logger.get());
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

  const auto workgroupDimStr = std::to_string(params.genWarpMapWorkgroupDimension);
  const IShaderCompiler::SMacroDefinition defines[] = {
    { "WORKGROUP_DIM", workgroupDimStr.data() },
  };

  options.preprocessorOptions.extraDefines = defines;

  const auto overridenUnspecialized = compiler->compileToSPIRV((const char*)shaderSource->getContent()->getPointer(), options);
  const auto shader = device->compileShader({ overridenUnspecialized.get() });
	if (!shader)
	{
		logger.log("Could not compile shaders!", ILogger::ELL_ERROR);
		return nullptr;
	}

  video::IGPUComputePipeline::SCreationParams pipelineParams[1] = {};
  pipelineParams[0].layout = pipelineLayout;
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

core::smart_refctd_ptr < video::IGPUPipelineLayout> EnvmapImportanceSampling::createGenLumaPipelineLayout(video::ILogicalDevice* device)
{
  asset::SPushConstantRange pcRange = {
    .stageFlags = hlsl::ESS_COMPUTE,
    .offset = 0,
    .size = sizeof(SLumaGenPushConstants)
  };

  const IGPUDescriptorSetLayout::SBinding bindings[] = {
    {
      .binding = 0u,
      .type = nbl::asset::IDescriptor::E_TYPE::ET_SAMPLED_IMAGE,
      .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
      .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
      .count = 1u
    },
    {
      .binding = 1u,
      .type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
      .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
      .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
      .count = 1u
    }
  };

  const auto setLayout = device->createDescriptorSetLayout(bindings);
	return device->createPipelineLayout({ &pcRange, 1 }, setLayout);

}

core::smart_refctd_ptr<video::IGPUPipelineLayout> EnvmapImportanceSampling::createGenWarpPipelineLayout(video::ILogicalDevice* device)
{
  const IGPUDescriptorSetLayout::SBinding bindings[] = {
    {
      .binding = 0u,
      .type = nbl::asset::IDescriptor::E_TYPE::ET_SAMPLED_IMAGE,
      .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
      .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
      .count = 1u,
    },
    {
      .binding = 1u,
      .type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
      .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
      .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
      .count = 1u
    }
  };

  const auto setLayout = device->createDescriptorSetLayout(bindings);
	return device->createPipelineLayout({}, setLayout, nullptr, nullptr, nullptr);
}

void EnvmapImportanceSampling::computeWarpMap(video::IQueue* queue)
{
  const auto logicalDevice = m_cachedCreationParams.utilities->getLogicalDevice();

  core::smart_refctd_ptr<IGPUCommandBuffer> cmdBuf;
	{
		// commandbuffer should refcount the pool, so it should be 100% legal to drop at the end of the scope
		auto gpuCommandPool = logicalDevice->createCommandPool(queue->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
		if (!gpuCommandPool)
		{
			if (auto* logger = logicalDevice->getLogger())
				logger->log("Compute Warpmap: failed to create command pool.", system::ILogger::ELL_ERROR);
			return;
		}
		gpuCommandPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &cmdBuf);
		if (!cmdBuf)
		{
			if (auto* logger = logicalDevice->getLogger())
				logger->log("Compute Warpmap: failed to create command buffer.", system::ILogger::ELL_ERROR);
			return;
		}
	}

  if (!cmdBuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT))
	{
		if (auto* logger = logicalDevice->getLogger())
			logger->log("Compute Warpmap: failed to begin command buffer.", system::ILogger::ELL_ERROR);
		return;
	}

  const auto lumaMapImage = m_lumaMap->getCreationParameters().image.get();
  const auto lumaMapMipLevels = lumaMapImage->getCreationParameters().mipLevels;
  const auto lumaMapExtent = lumaMapImage->getCreationParameters().extent;

  const auto warpMapImage = m_warpMap->getCreationParameters().image.get();

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
          .levelCount = lumaMapMipLevels,
          .baseArrayLayer = 0u,
          .layerCount = 1u
        },
        .oldLayout = IImage::LAYOUT::UNDEFINED,
        .newLayout = IImage::LAYOUT::GENERAL,
      }
    };                
    cmdBuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = barriers });
  }

  // Gen Luma Map
  {
    SLumaGenPushConstants pcData = {};
    pcData.lumaRGBCoefficients = { 0.2126729f, 0.7151522f, 0.0721750f };
    pcData.lumaMapResolution = {lumaMapExtent.width, lumaMapExtent.height};

    cmdBuf->bindComputePipeline(m_genLumaPipeline.get());
    cmdBuf->pushConstants(m_genLumaPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE,
      0, sizeof(SLumaGenPushConstants), &pcData);
    cmdBuf->bindDescriptorSets(EPBP_COMPUTE, m_genLumaPipeline->getLayout(),
      0, 1, &m_genLumaDescriptorSet.get());
    cmdBuf->dispatch(m_lumaWorkgroupCount.x, m_lumaWorkgroupCount.y, 1);
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
          .layerCount = 1u
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
          .layerCount = 1u
        },
        .oldLayout = IImage::LAYOUT::GENERAL,
        .newLayout = IImage::LAYOUT::TRANSFER_DST_OPTIMAL,
      }
    };                
    cmdBuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = barriers });
    generateMipmap(cmdBuf.get(), lumaMapImage);
  }

  core::smart_refctd_ptr<IGPUBuffer> lumaTexelBuffer;
  const auto lumaMapLastMip = lumaMapMipLevels - 1;
  const auto lumaMapLastMipExtent = lumaMapImage->getMipSize(lumaMapLastMip);
  const auto lumaMapLastTexelCount = lumaMapLastMipExtent.x * lumaMapLastMipExtent.y * lumaMapLastMipExtent.z;
  {
    IGPUImage::SBufferCopy region = {};
    region.imageSubresource.aspectMask = IImage::EAF_COLOR_BIT;
    region.imageSubresource.mipLevel = lumaMapLastMip;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = { lumaMapLastMipExtent.x, lumaMapLastMipExtent.y, lumaMapLastMipExtent.z };

    IGPUBuffer::SCreationParams bufferCreationParams = {};
    bufferCreationParams.size = lumaMapLastTexelCount * getTexelOrBlockBytesize(EF_R32_SFLOAT);
    bufferCreationParams.usage = IGPUBuffer::EUF_TRANSFER_DST_BIT;
    lumaTexelBuffer = logicalDevice->createBuffer(std::move(bufferCreationParams));
    if (!lumaTexelBuffer)
    {
      if (auto* logger = logicalDevice->getLogger())
        logger->log("ScreenShot: failed to create GPU texel buffer.", system::ILogger::ELL_ERROR);
      return;
    }
    auto gpuTexelBufferMemReqs = lumaTexelBuffer->getMemoryReqs();
    gpuTexelBufferMemReqs.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDownStreamingMemoryTypeBits();
    if (!gpuTexelBufferMemReqs.memoryTypeBits)
    {
      if (auto* logger = logicalDevice->getLogger())
        logger->log("ScreenShot: no down-streaming memory type for texel buffer.", system::ILogger::ELL_ERROR);
      return;
    }
    auto gpuTexelBufferMem = logicalDevice->allocate(gpuTexelBufferMemReqs, lumaTexelBuffer.get());
    if (!gpuTexelBufferMem.isValid())
    {
      if (auto* logger = logicalDevice->getLogger())
        logger->log("ScreenShot: failed to allocate texel buffer memory.", system::ILogger::ELL_ERROR);
      return;
    }

    IGPUCommandBuffer::SPipelineBarrierDependencyInfo info = {};
		decltype(info)::image_barrier_t barrier = {};
		info.imgBarriers = { &barrier, &barrier + 1 };

		{
			barrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::BLIT_BIT;
			barrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT;
			barrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT;
			barrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::TRANSFER_READ_BIT;
			barrier.oldLayout = IImage::LAYOUT::TRANSFER_DST_OPTIMAL;
			barrier.newLayout = IImage::LAYOUT::TRANSFER_SRC_OPTIMAL;
			barrier.image = lumaMapImage;
			barrier.subresourceRange.aspectMask = IImage::EAF_COLOR_BIT;
			barrier.subresourceRange.baseMipLevel = lumaMapMipLevels - 1;
			barrier.subresourceRange.levelCount = 1u;
			barrier.subresourceRange.baseArrayLayer = 0;
			barrier.subresourceRange.layerCount = 1;
			cmdBuf->pipelineBarrier(EDF_NONE,info);
		}
    cmdBuf->copyImageToBuffer(lumaMapImage,IImage::LAYOUT::TRANSFER_SRC_OPTIMAL,lumaTexelBuffer.get(),1,&region);
  }

  {
    IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t barriers[] = {
      {
        .barrier = {
          .dep = {
            .srcStageMask = PIPELINE_STAGE_FLAGS::BLIT_BIT,
            .srcAccessMask = ACCESS_FLAGS::TRANSFER_READ_BIT,
            .dstStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS,
            .dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS
          }
        },
        .image = lumaMapImage,
        .subresourceRange = {
          .aspectMask = IImage::EAF_COLOR_BIT,
          .baseMipLevel = 0u,
          .levelCount = lumaMapMipLevels - 1,
          .baseArrayLayer = 0u,
          .layerCount = 1u
        },
        .oldLayout = IImage::LAYOUT::TRANSFER_SRC_OPTIMAL,
        .newLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL,
      },
      {
        .barrier = {
          .dep = {
            .srcStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT,
            .srcAccessMask = ACCESS_FLAGS::TRANSFER_READ_BIT,
            .dstStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS,
            .dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS
          }
        },
        .image = lumaMapImage,
        .subresourceRange = {
          .aspectMask = IImage::EAF_COLOR_BIT,
          .baseMipLevel = lumaMapMipLevels - 1,
          .levelCount = 1,
          .baseArrayLayer = 0u,
          .layerCount = 1u
        },
        .oldLayout = IImage::LAYOUT::TRANSFER_SRC_OPTIMAL,
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
          .layerCount = 1u
        },
        .oldLayout = IImage::LAYOUT::UNDEFINED,
        .newLayout = IImage::LAYOUT::GENERAL,
      }
    };                
    cmdBuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = barriers });
    cmdBuf->bindComputePipeline(m_genWarpPipeline.get());
    cmdBuf->bindDescriptorSets(EPBP_COMPUTE, m_genWarpPipeline->getLayout(),
      0, 1, &m_genWarpDescriptorSet.get());
    cmdBuf->dispatch(m_warpWorkgroupCount.x, m_warpWorkgroupCount.y, 1);
  }

  {
    IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t barriers[] = {
      {
        .barrier = {
          .dep = {
            .srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
            .srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
            .dstStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS,
            .dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS
          }
        },
        .image = warpMapImage,
        .subresourceRange = {
          .aspectMask = IImage::EAF_COLOR_BIT,
          .baseMipLevel = 0,
          .levelCount = 1,
          .baseArrayLayer = 0u,
          .layerCount = 1u
        },
        .oldLayout = IImage::LAYOUT::GENERAL,
        .newLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL,
      }
    };                
    cmdBuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = barriers });
  }

  if (!cmdBuf->end())
	{
		if (auto* logger = logicalDevice->getLogger())
			logger->log("ScreenShot: failed to end command buffer.", system::ILogger::ELL_ERROR);
		return;
	}

  {
    auto signalSemaphore = logicalDevice->createSemaphore(0);

    IQueue::SSubmitInfo info;
    IQueue::SSubmitInfo::SCommandBufferInfo cmdBufferInfo{ cmdBuf.get() };
    IQueue::SSubmitInfo::SSemaphoreInfo signalSemaphoreInfo;
    signalSemaphoreInfo.semaphore = signalSemaphore.get();
    signalSemaphoreInfo.value = 1;
    signalSemaphoreInfo.stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
    info.commandBuffers = { &cmdBufferInfo, &cmdBufferInfo + 1 };
    info.signalSemaphores = { &signalSemaphoreInfo, &signalSemaphoreInfo + 1 };

    if (auto* logger = logicalDevice->getLogger())
      logger->log("Compute Warpmap: submitting copy command buffer.", system::ILogger::ELL_INFO);
    if (queue->submit({ &info, &info + 1}) != IQueue::RESULT::SUCCESS)
    {
      if (auto* logger = logicalDevice->getLogger())
        logger->log("Compute Warpmap: failed to submit copy command buffer.", system::ILogger::ELL_ERROR);
      return;
    }

    ISemaphore::SWaitInfo waitInfo{ signalSemaphore.get(), 1u};

    if (auto* logger = logicalDevice->getLogger())
      logger->log("Compute Warpmap: waiting for copy completion.", system::ILogger::ELL_INFO);
    if (logicalDevice->blockForSemaphores({&waitInfo, &waitInfo + 1}) != ISemaphore::WAIT_RESULT::SUCCESS)
    {
      if (auto* logger = logicalDevice->getLogger())
        logger->log("Compute Warpmap: failed to wait for copy completion.", system::ILogger::ELL_ERROR);
      return;
    }

    auto* allocation = lumaTexelBuffer->getBoundMemory().memory;
    const IDeviceMemoryAllocation::MemoryRange range = { 0u, lumaTexelBuffer->getSize() };
    auto* ptr = reinterpret_cast<hlsl::float32_t*>(allocation->map(range, IDeviceMemoryAllocation::EMCAF_READ));

    m_avgLuma = std::reduce(ptr, ptr + lumaMapLastTexelCount) / float32_t(lumaMapLastTexelCount);
  }
}

nbl::video::IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t EnvmapImportanceSampling::getWarpMapBarrier(
  core::bitflag<nbl::asset::PIPELINE_STAGE_FLAGS> dstStageMask,
  core::bitflag<nbl::asset::ACCESS_FLAGS> dstAccessMask,
  nbl::video::IGPUImage::LAYOUT newLayout)
{
  const auto warpMapImage = m_warpMap->getCreationParameters().image.get();
  return {
    .barrier = {
      .dep = {
        .srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
        .srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
        .dstStageMask = dstStageMask,
        .dstAccessMask = dstAccessMask
      }
    },
    .image = warpMapImage,
    .subresourceRange = {
      .aspectMask = IImage::EAF_COLOR_BIT,
      .baseMipLevel = 0,
      .levelCount = 1,
      .baseArrayLayer = 0u,
      .layerCount = 1u
    },
    .oldLayout = IImage::LAYOUT::GENERAL,
    .newLayout = newLayout,
  };
}

nbl::video::IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t EnvmapImportanceSampling::getLumaMapBarrier(
  core::bitflag<nbl::asset::PIPELINE_STAGE_FLAGS> dstStageMask,
  core::bitflag<nbl::asset::ACCESS_FLAGS> dstAccessMask,
  nbl::video::IGPUImage::LAYOUT newLayout)
{
  const auto lumaMapImage = m_lumaMap->getCreationParameters().image.get();
  return {
    .barrier = {
      .dep = {
        .srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
        .srcAccessMask = ACCESS_FLAGS::SHADER_READ_BITS,
        .dstStageMask = dstStageMask,
        .dstAccessMask = dstAccessMask
      }
    },
    .image = lumaMapImage,
    .subresourceRange = {
      .aspectMask = IImage::EAF_COLOR_BIT,
      .baseMipLevel = 0,
      .levelCount = 1,
      .baseArrayLayer = 0u,
      .layerCount = 1u
    },
    .oldLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL,
    .newLayout = newLayout,
  };
}


}
