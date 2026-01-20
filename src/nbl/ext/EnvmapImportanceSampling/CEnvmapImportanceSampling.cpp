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

  void generateMipmap(video::IGPUCommandBuffer* cmdBuf, core::smart_refctd_ptr<IGPUImageView> textureView)
  {
    
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
    imgParams.usage = IImage::EUF_STORAGE_BIT;
    const auto image = device->createImage(std::move(imgParams));
    auto imageMemReqs = image->getMemoryReqs();
    imageMemReqs.memoryTypeBits &= device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
    device->allocate(imageMemReqs, image.get());

    IGPUImageView::SCreationParams viewparams;
    viewparams.subUsages = IImage::EUF_STORAGE_BIT;
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
  
  constructorParams.lumaWorkgroupSize = calcWorkgroupSize(EnvMapPoTExtent, params.genLumaMapWorkgroupDimension);

  constructorParams.lumaMap = createLumaMap(device, EnvMapPoTExtent, MipCountLuminance);

	const auto upscale = 0;
	const asset::VkExtent3D WarpMapExtent = {EnvMapPoTExtent.width<<upscale,EnvMapPoTExtent.height<<upscale,EnvMapPoTExtent.depth};
  constructorParams.warpMap = createWarpMap(device, WarpMapExtent);
  constructorParams.warpWorkgroupSize = calcWorkgroupSize(WarpMapExtent, params.genWarpMapWorkgroupDimension);

  const auto genLumaPipelineLayout = createGenLumaPipelineLayout(device);
  constructorParams.genLumaPipeline = createGenLumaPipeline(params, genLumaPipelineLayout.get());
  const auto genLumaDescriptorPool = device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT, genLumaPipelineLayout->getDescriptorSetLayouts());
  const auto genLumaDescriptorSet = genLumaDescriptorPool->createDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(genLumaPipelineLayout->getDescriptorSetLayouts()[0]));

  IGPUDescriptorSet::SDescriptorInfo envMapDescriptorInfo; 
  envMapDescriptorInfo.desc = params.envMap;
  envMapDescriptorInfo.info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;

  IGPUDescriptorSet::SDescriptorInfo lumaMapDescriptorInfo;
  lumaMapDescriptorInfo.desc = constructorParams.lumaMap;
  lumaMapDescriptorInfo.info.image.imageLayout = IImage::LAYOUT::GENERAL;

  const IGPUDescriptorSet::SWriteDescriptorSet writes[] = {
    {
      .dstSet = genLumaDescriptorSet.get(), .binding = 0, .count = 1, .info = &envMapDescriptorInfo
    },
    {
      .dstSet = genLumaDescriptorSet.get(), .binding = 1, .count = 1, .info = &lumaMapDescriptorInfo
    }
  };

  device->updateDescriptorSets(writes, {});

  constructorParams.genLumaDescriptorSet = genLumaDescriptorSet;

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

  const IShaderCompiler::SMacroDefinition defines[] = {
    { "WORKGROUP_DIM", "16" },
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
	return device->createPipelineLayout({ &pcRange, 1 }, setLayout, nullptr, nullptr, nullptr);

}

core::smart_refctd_ptr<video::IGPUPipelineLayout> EnvmapImportanceSampling::createMeasureLumaPipelineLayout(video::ILogicalDevice* device)
{
  asset::SPushConstantRange pcRange = {
    .stageFlags = hlsl::ESS_COMPUTE,
		.offset = 0,
		.size = sizeof(SLumaMeasurePushConstants)
  };

  const IGPUDescriptorSetLayout::SBinding bindings[] = {
    {
      .binding = 0u,
      .type = nbl::asset::IDescriptor::E_TYPE::ET_SAMPLED_IMAGE,
      .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
      .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
      .count = 1u,
    }
  };

  const auto setLayout = device->createDescriptorSetLayout(bindings);
	return device->createPipelineLayout({ &pcRange, 1 }, setLayout, nullptr, nullptr, nullptr);
}

core::smart_refctd_ptr<video::IGPUPipelineLayout> EnvmapImportanceSampling::createGenWarpMapPipelineLayout(video::ILogicalDevice* device)
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

bool EnvmapImportanceSampling::computeWarpMap(video::IGPUCommandBuffer* cmdBuf, const float envMapRegularizationFactor, float& pdfNormalizationFactor, float& maxEmittanceLuma)
{
  bool enableRIS = false;
  
  SLumaGenPushConstants pcData = {};
  pcData.luminanceScales = { 0.2126729f, 0.7151522f, 0.0721750f, 0.0f };
  {
    const auto imageExtent = m_lumaMap->getCreationParameters().image->getCreationParameters().extent;
    pcData.lumaMapResolution = {imageExtent.width, imageExtent.height};
  }

  IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t barrier;                
  barrier.barrier = {
    .dep = {
      .srcStageMask = PIPELINE_STAGE_FLAGS::NONE,
      .srcAccessMask = ACCESS_FLAGS::NONE,
      .dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
      .dstAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS
    }
  };
  barrier.image = m_lumaMap->getCreationParameters().image.get();
  barrier.subresourceRange = {
    .aspectMask = IImage::EAF_COLOR_BIT,
    .baseMipLevel = 0u,
    .levelCount = m_lumaMap->getCreationParameters().image->getCreationParameters().mipLevels,
    .baseArrayLayer = 0u,
    .layerCount = 1u
  };
  barrier.oldLayout = IImage::LAYOUT::UNDEFINED;
  barrier.newLayout = IImage::LAYOUT::GENERAL;
  cmdBuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = {&barrier, 1} });

  cmdBuf->bindComputePipeline(m_genLumaPipeline.get());
  cmdBuf->pushConstants(m_genLumaPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE,
    0, sizeof(SLumaGenPushConstants), &pcData);
  cmdBuf->bindDescriptorSets(EPBP_COMPUTE, m_genLumaPipeline->getLayout(),
    0, 1, &m_genLumaDescriptorSet.get());
  cmdBuf->dispatch(m_lumaWorkgroupSize.x, m_lumaWorkgroupSize.y, 1);
  
  return enableRIS;
  
}
}
