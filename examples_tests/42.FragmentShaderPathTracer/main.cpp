// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"
#include "../common/Camera.hpp"
#include "nbl/ext/ScreenShot/ScreenShot.h"
#include "nbl/video/utilities/CDumbPresentationOracle.h"

using namespace nbl;
using namespace core;
using namespace ui;


using namespace nbl;
using namespace core;
using namespace asset;
using namespace video;

smart_refctd_ptr<IGPUImageView> createHDRImageView(nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> device, asset::E_FORMAT colorFormat, uint32_t width, uint32_t height)
{
	smart_refctd_ptr<IGPUImageView> gpuImageViewColorBuffer;
	{
		IGPUImage::SCreationParams imgInfo;
		imgInfo.format = colorFormat;
		imgInfo.type = IGPUImage::ET_2D;
		imgInfo.extent.width = width;
		imgInfo.extent.height = height;
		imgInfo.extent.depth = 1u;
		imgInfo.mipLevels = 1u;
		imgInfo.arrayLayers = 1u;
		imgInfo.samples = asset::ICPUImage::ESCF_1_BIT;
		imgInfo.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
		imgInfo.usage = core::bitflag(asset::IImage::EUF_STORAGE_BIT) | asset::IImage::EUF_TRANSFER_SRC_BIT;

		auto image = device->createGPUImageOnDedMem(std::move(imgInfo),device->getDeviceLocalGPUMemoryReqs());

		IGPUImageView::SCreationParams imgViewInfo;
		imgViewInfo.image = std::move(image);
		imgViewInfo.format = colorFormat;
		imgViewInfo.viewType = IGPUImageView::ET_2D;
		imgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
		imgViewInfo.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
		imgViewInfo.subresourceRange.baseArrayLayer = 0u;
		imgViewInfo.subresourceRange.baseMipLevel = 0u;
		imgViewInfo.subresourceRange.layerCount = 1u;
		imgViewInfo.subresourceRange.levelCount = 1u;

		gpuImageViewColorBuffer = device->createGPUImageView(std::move(imgViewInfo));
	}

	return gpuImageViewColorBuffer;
}

struct ShaderParameters
{
	const uint32_t MaxDepthLog2 = 4; //5
	const uint32_t MaxSamplesLog2 = 10; //18
} kShaderParameters;

enum E_LIGHT_GEOMETRY
{
	ELG_SPHERE,
	ELG_TRIANGLE,
	ELG_RECTANGLE
};

struct DispatchInfo_t
{
	uint32_t workGroupCount[3];
};

_NBL_STATIC_INLINE_CONSTEXPR uint32_t DEFAULT_WORK_GROUP_SIZE = 16u;

DispatchInfo_t getDispatchInfo(uint32_t imgWidth, uint32_t imgHeight) {
	DispatchInfo_t ret = {};
	ret.workGroupCount[0] = (uint32_t)core::ceil<float>((float)imgWidth / (float)DEFAULT_WORK_GROUP_SIZE);
	ret.workGroupCount[1] = (uint32_t)core::ceil<float>((float)imgHeight / (float)DEFAULT_WORK_GROUP_SIZE);
	ret.workGroupCount[2] = 1;
	return ret;
}

int main()
{
	constexpr uint32_t WIN_W = 1280;
	constexpr uint32_t WIN_H = 720;
	constexpr uint32_t FBO_COUNT = 2u;
	constexpr uint32_t FRAMES_IN_FLIGHT = 5u;
	static_assert(FRAMES_IN_FLIGHT>FBO_COUNT);
	
	CommonAPI::SFeatureRequest<video::IAPIConnection::E_FEATURE> requiredInstanceFeatures = {};
	requiredInstanceFeatures.count = 1u;
	video::IAPIConnection::E_FEATURE requiredFeatures_Instance[] = { video::IAPIConnection::EF_SURFACE };
	requiredInstanceFeatures.features = requiredFeatures_Instance;

	CommonAPI::SFeatureRequest<video::IAPIConnection::E_FEATURE> optionalInstanceFeatures = {};

	CommonAPI::SFeatureRequest<video::ILogicalDevice::E_FEATURE> requiredDeviceFeatures = {};
	requiredDeviceFeatures.count = 1u;
	video::ILogicalDevice::E_FEATURE requiredFeatures_Device[] = { video::ILogicalDevice::EF_SWAPCHAIN };
	requiredDeviceFeatures.features = requiredFeatures_Device;

	CommonAPI::SFeatureRequest< video::ILogicalDevice::E_FEATURE> optionalDeviceFeatures = {};
	optionalDeviceFeatures.count = 2u;
	video::ILogicalDevice::E_FEATURE optionalFeatures_Device[] = { video::ILogicalDevice::EF_RAY_TRACING_PIPELINE, video::ILogicalDevice::EF_RAY_QUERY };
	optionalDeviceFeatures.features = optionalFeatures_Device;

	const auto swapchainImageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_COLOR_ATTACHMENT_BIT | asset::IImage::EUF_TRANSFER_DST_BIT);
	const video::ISurface::SFormat surfaceFormat;
	
	auto initOutput = CommonAPI::Init(
		video::EAT_VULKAN,
		"Compute Shader PathTracer",
		requiredInstanceFeatures,
		optionalInstanceFeatures,
		requiredDeviceFeatures,
		optionalDeviceFeatures,
		WIN_W, WIN_H,
		FBO_COUNT,
		swapchainImageUsage, 
		surfaceFormat,
		asset::EF_D32_SFLOAT);

	auto system = std::move(initOutput.system);
	auto window = std::move(initOutput.window);
	auto windowCb = std::move(initOutput.windowCb);
	auto gl = std::move(initOutput.apiConnection);
	auto surface = std::move(initOutput.surface);
	auto gpuPhysicalDevice = std::move(initOutput.physicalDevice);
	auto device = std::move(initOutput.logicalDevice);
	auto queues = std::move(initOutput.queues);
	auto graphicsQueue = queues[CommonAPI::InitOutput::EQT_GRAPHICS];
	auto transferUpQueue = queues[CommonAPI::InitOutput::EQT_TRANSFER_UP];
	auto computeQueue = queues[CommonAPI::InitOutput::EQT_COMPUTE];
	auto swapchain = std::move(initOutput.swapchain);
	auto renderpass = std::move(initOutput.renderpass);
	auto fbo = std::move(initOutput.fbo);
	auto assetManager = std::move(initOutput.assetManager);
	auto cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
	auto logger = std::move(initOutput.logger);
	auto inputSystem = std::move(initOutput.inputSystem);
	auto utilities = std::move(initOutput.utilities);
	auto graphicsCommandPool = std::move(initOutput.commandPools[CommonAPI::InitOutput::EQT_GRAPHICS]);
	auto computeCommandPool = std::move(initOutput.commandPools[CommonAPI::InitOutput::EQT_COMPUTE]);

	auto graphicsCmdPoolQueueFamIdx = graphicsQueue->getFamilyIndex();

	nbl::video::IGPUObjectFromAssetConverter CPU2GPU;
	
	core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdbuf[FRAMES_IN_FLIGHT];
	device->createCommandBuffers(graphicsCommandPool.get(), nbl::video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, cmdbuf);
	
	constexpr uint32_t maxDescriptorCount = 256u;
	constexpr uint32_t PoolSizesCount = 5u;
	nbl::video::IDescriptorPool::SDescriptorPoolSize poolSizes[PoolSizesCount] = {
		{ EDT_STORAGE_BUFFER, 1},
		{ EDT_STORAGE_IMAGE, 8},
		{ EDT_COMBINED_IMAGE_SAMPLER, 2},
		{ EDT_UNIFORM_TEXEL_BUFFER, 1},
		{ EDT_UNIFORM_BUFFER, 1},
	};

	auto descriptorPool = device->createDescriptorPool(static_cast<nbl::video::IDescriptorPool::E_CREATE_FLAGS>(0), maxDescriptorCount, PoolSizesCount, poolSizes);

	// Camera 
	core::vectorSIMDf cameraPosition(0, 5, -10);
	matrix4SIMD proj = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(60), float(WIN_W) / WIN_H, 0.01f, 500.0f);
	Camera cam = Camera(cameraPosition, core::vectorSIMDf(0, 0, 0), proj);

	IGPUDescriptorSetLayout::SBinding descriptorSet0Bindings[] = {
		{ 0u, EDT_STORAGE_IMAGE, 1u, IShader::ESS_COMPUTE, nullptr },
	};
	IGPUDescriptorSetLayout::SBinding uboBinding {0, EDT_UNIFORM_BUFFER, 1u, IShader::ESS_COMPUTE, nullptr};
	IGPUDescriptorSetLayout::SBinding descriptorSet3Bindings[] = {
		{ 0u, EDT_COMBINED_IMAGE_SAMPLER, 1u, IShader::ESS_COMPUTE, nullptr },
		{ 1u, EDT_UNIFORM_TEXEL_BUFFER, 1u, IShader::ESS_COMPUTE, nullptr },
		{ 2u, EDT_COMBINED_IMAGE_SAMPLER, 1u, IShader::ESS_COMPUTE, nullptr }
	};
	
	auto gpuDescriptorSetLayout0 = device->createGPUDescriptorSetLayout(descriptorSet0Bindings, descriptorSet0Bindings + 1u);
	auto gpuDescriptorSetLayout1 = device->createGPUDescriptorSetLayout(&uboBinding, &uboBinding + 1u);
	auto gpuDescriptorSetLayout2 = device->createGPUDescriptorSetLayout(descriptorSet3Bindings, descriptorSet3Bindings+3u);

	auto createGpuResources = [&](std::string pathToShader) -> core::smart_refctd_ptr<video::IGPUComputePipeline>
	{
		asset::IAssetLoader::SAssetLoadParams params{};
		params.logger = logger.get();
		//params.relativeDir = tmp.c_str();
		auto spec = assetManager->getAsset(pathToShader,params).getContents();
		
		if (spec.empty())
			assert(false);

		auto cpuComputeSpecializedShader = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*spec.begin());

		ISpecializedShader::SInfo info = cpuComputeSpecializedShader->getSpecializationInfo();
		info.m_backingBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(sizeof(ShaderParameters));
		memcpy(info.m_backingBuffer->getPointer(),&kShaderParameters,sizeof(ShaderParameters));
		info.m_entries = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ISpecializedShader::SInfo::SMapEntry>>(2u);
		for (uint32_t i=0; i<2; i++)
			info.m_entries->operator[](i) = {i,i*sizeof(uint32_t),sizeof(uint32_t)};


		cpuComputeSpecializedShader->setSpecializationInfo(std::move(info));

		auto gpuComputeSpecializedShader = CPU2GPU.getGPUObjectsFromAssets(&cpuComputeSpecializedShader, &cpuComputeSpecializedShader + 1, cpu2gpuParams)->front();

		auto gpuPipelineLayout = device->createGPUPipelineLayout(nullptr, nullptr, core::smart_refctd_ptr(gpuDescriptorSetLayout0), core::smart_refctd_ptr(gpuDescriptorSetLayout1), core::smart_refctd_ptr(gpuDescriptorSetLayout2), nullptr);

		auto gpuPipeline = device->createGPUComputePipeline(nullptr, std::move(gpuPipelineLayout), std::move(gpuComputeSpecializedShader));

		return gpuPipeline;
	};

	E_LIGHT_GEOMETRY lightGeom = ELG_SPHERE;
	constexpr const char* shaderPaths[] = {"../litBySphere.comp","../litByTriangle.comp","../litByRectangle.comp"};
	auto gpuComputePipeline = createGpuResources(shaderPaths[lightGeom]);

	DispatchInfo_t dispatchInfo = getDispatchInfo(WIN_W, WIN_H);

	auto createGPUImageView = [&](std::string pathToOpenEXRHDRIImage)
	{
		auto pathToTexture = pathToOpenEXRHDRIImage;
		IAssetLoader::SAssetLoadParams lp(0ull, nullptr, IAssetLoader::ECF_DONT_CACHE_REFERENCES);
		auto cpuTexture = assetManager->getAsset(pathToTexture, lp);
		auto cpuTextureContents = cpuTexture.getContents();
		assert(!cpuTextureContents.empty());
		auto cpuImage = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(*cpuTextureContents.begin());
		cpuImage->setImageUsageFlags(IImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT);

		ICPUImageView::SCreationParams viewParams;
		viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
		viewParams.image = cpuImage;
		viewParams.format = viewParams.image->getCreationParameters().format;
		viewParams.viewType = IImageView<ICPUImage>::ET_2D;
		viewParams.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
		viewParams.subresourceRange.baseArrayLayer = 0u;
		viewParams.subresourceRange.layerCount = 1u;
		viewParams.subresourceRange.baseMipLevel = 0u;
		viewParams.subresourceRange.levelCount = 1u;

		auto cpuImageView = ICPUImageView::create(std::move(viewParams));

		cpu2gpuParams.beginCommandBuffers();
		auto gpuImageView = CPU2GPU.getGPUObjectsFromAssets(&cpuImageView, &cpuImageView + 1u, cpu2gpuParams)->front();
		cpu2gpuParams.waitForCreationToComplete(false);

		return gpuImageView;
	};
	
	auto gpuEnvmapImageView = createGPUImageView("../../media/envmap/envmap_0.exr");

	smart_refctd_ptr<IGPUBufferView> gpuSequenceBufferView;
	{
		const uint32_t MaxDimensions = 3u<<kShaderParameters.MaxDepthLog2;
		const uint32_t MaxSamples = 1u<<kShaderParameters.MaxSamplesLog2;

		auto sampleSequence = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(uint32_t)*MaxDimensions*MaxSamples);
		
		core::OwenSampler sampler(MaxDimensions, 0xdeadbeefu);
		//core::SobolSampler sampler(MaxDimensions);

		auto out = reinterpret_cast<uint32_t*>(sampleSequence->getPointer());
		for (auto dim=0u; dim<MaxDimensions; dim++)
		for (uint32_t i=0; i<MaxSamples; i++)
		{
			out[i*MaxDimensions+dim] = sampler.sample(dim,i);
		}
		
		// TODO: Temp Fix because createFilledDeviceLocalGPUBufferOnDedMem doesn't take in params
		// auto gpuSequenceBuffer = utilities->createFilledDeviceLocalGPUBufferOnDedMem(graphicsQueue, sampleSequence->getSize(), sampleSequence->getPointer());
		core::smart_refctd_ptr<IGPUBuffer> gpuSequenceBuffer;
		{
			IGPUBuffer::SCreationParams params = {};
			const size_t size = sampleSequence->getSize();
			params.usage = core::bitflag(asset::IBuffer::EUF_TRANSFER_DST_BIT) | asset::IBuffer::EUF_UNIFORM_TEXEL_BUFFER_BIT; 
			gpuSequenceBuffer = device->createDeviceLocalGPUBufferOnDedMem(params, size);
			utilities->updateBufferRangeViaStagingBuffer(graphicsQueue, asset::SBufferRange<IGPUBuffer>{0u,size,gpuSequenceBuffer},sampleSequence->getPointer());
		}
		gpuSequenceBufferView = device->createGPUBufferView(gpuSequenceBuffer.get(), asset::EF_R32G32B32_UINT);
	}

	smart_refctd_ptr<IGPUImageView> gpuScrambleImageView;
	{
		IGPUImage::SCreationParams imgParams;
		imgParams.flags = static_cast<IImage::E_CREATE_FLAGS>(0u);
		imgParams.type = IImage::ET_2D;
		imgParams.format = EF_R32G32_UINT;
		imgParams.extent = {WIN_W, WIN_H,1u};
		imgParams.mipLevels = 1u;
		imgParams.arrayLayers = 1u;
		imgParams.samples = IImage::ESCF_1_BIT;
		imgParams.usage = core::bitflag(IImage::EUF_SAMPLED_BIT) | IImage::EUF_TRANSFER_DST_BIT;
		imgParams.initialLayout = asset::EIL_SHADER_READ_ONLY_OPTIMAL;

		IGPUImage::SBufferCopy region = {};
		region.bufferOffset = 0u;
		region.bufferRowLength = 0u;
		region.bufferImageHeight = 0u;
		region.imageExtent = imgParams.extent;
		region.imageOffset = {0u,0u,0u};
		region.imageSubresource.layerCount = 1u;
		region.imageSubresource.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;

		constexpr auto ScrambleStateChannels = 2u;
		const auto renderPixelCount = imgParams.extent.width*imgParams.extent.height;
		core::vector<uint32_t> random(renderPixelCount*ScrambleStateChannels);
		{
			core::RandomSampler rng(0xbadc0ffeu);
			for (auto& pixel : random)
				pixel = rng.nextSample();
		}

		// TODO: Temp Fix because createFilledDeviceLocalGPUBufferOnDedMem doesn't take in params
		// auto buffer = utilities->createFilledDeviceLocalGPUBufferOnDedMem(graphicsQueue, random.size()*sizeof(uint32_t), random.data());
		core::smart_refctd_ptr<IGPUBuffer> buffer;
		{
			IGPUBuffer::SCreationParams params = {};
			const size_t size = random.size() * sizeof(uint32_t);
			params.usage = core::bitflag(asset::IBuffer::EUF_TRANSFER_DST_BIT) | asset::IBuffer::EUF_TRANSFER_SRC_BIT; 
			buffer = device->createDeviceLocalGPUBufferOnDedMem(params, size);
			utilities->updateBufferRangeViaStagingBuffer(graphicsQueue, asset::SBufferRange<IGPUBuffer>{0u,size,buffer},random.data());
		}

		IGPUImageView::SCreationParams viewParams;
		viewParams.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
		viewParams.image = utilities->createFilledDeviceLocalGPUImageOnDedMem(graphicsQueue, std::move(imgParams), buffer.get(), 1u, &region);
		viewParams.viewType = IGPUImageView::ET_2D;
		viewParams.format = EF_R32G32_UINT;
		viewParams.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
		viewParams.subresourceRange.levelCount = 1u;
		viewParams.subresourceRange.layerCount = 1u;
		gpuScrambleImageView = device->createGPUImageView(std::move(viewParams));
	}
	
	// Create Out Image TODO
	smart_refctd_ptr<IGPUImageView> outHDRImageViews[FBO_COUNT] = {};
	for(uint32_t i = 0; i < FBO_COUNT; ++i) {
		outHDRImageViews[i] = createHDRImageView(device, asset::EF_R16G16B16A16_SFLOAT, WIN_W, WIN_H);
	}

	core::smart_refctd_ptr<IGPUDescriptorSet> descriptorSets0[FBO_COUNT] = {};
	for(uint32_t i = 0; i < FBO_COUNT; ++i)
	{
		auto & descSet = descriptorSets0[i];
		descSet = device->createGPUDescriptorSet(descriptorPool.get(), core::smart_refctd_ptr(gpuDescriptorSetLayout0));
		video::IGPUDescriptorSet::SWriteDescriptorSet writeDescriptorSet;
		writeDescriptorSet.dstSet = descSet.get();
		writeDescriptorSet.binding = 0;
		writeDescriptorSet.count = 1u;
		writeDescriptorSet.arrayElement = 0u;
		writeDescriptorSet.descriptorType = asset::EDT_STORAGE_IMAGE;
		video::IGPUDescriptorSet::SDescriptorInfo info;
		{
			info.desc = outHDRImageViews[i];
			info.image.sampler = nullptr;
			info.image.imageLayout = asset::E_IMAGE_LAYOUT::EIL_GENERAL;
		}
		writeDescriptorSet.info = &info;
		device->updateDescriptorSets(1u, &writeDescriptorSet, 0u, nullptr);
	}
	
	struct SBasicViewParametersAligned
	{
		SBasicViewParameters uboData;
	};

	IGPUBuffer::SCreationParams gpuuboParams = {};
	const size_t gpuuboParamsSize = sizeof(SBasicViewParametersAligned);
	gpuuboParams.usage = core::bitflag(IGPUBuffer::EUF_UNIFORM_BUFFER_BIT) | IGPUBuffer::EUF_TRANSFER_DST_BIT;
	auto gpuubo = device->createDeviceLocalGPUBufferOnDedMem(gpuuboParams, gpuuboParamsSize);
	auto uboDescriptorSet1 = device->createGPUDescriptorSet(descriptorPool.get(), core::smart_refctd_ptr(gpuDescriptorSetLayout1));
	{
		video::IGPUDescriptorSet::SWriteDescriptorSet uboWriteDescriptorSet;
		uboWriteDescriptorSet.dstSet = uboDescriptorSet1.get();
		uboWriteDescriptorSet.binding = 0;
		uboWriteDescriptorSet.count = 1u;
		uboWriteDescriptorSet.arrayElement = 0u;
		uboWriteDescriptorSet.descriptorType = asset::EDT_UNIFORM_BUFFER;
		video::IGPUDescriptorSet::SDescriptorInfo info;
		{
			info.desc = gpuubo;
			info.buffer.offset = 0ull;
			info.buffer.size = sizeof(SBasicViewParametersAligned);
		}
		uboWriteDescriptorSet.info = &info;
		device->updateDescriptorSets(1u, &uboWriteDescriptorSet, 0u, nullptr);
	}

	ISampler::SParams samplerParams0 = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_LINEAR, ISampler::ETF_LINEAR, ISampler::ESMM_LINEAR, 0u, false, ECO_ALWAYS };
	auto sampler0 = device->createGPUSampler(samplerParams0);
	ISampler::SParams samplerParams1 = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_INT_OPAQUE_BLACK, ISampler::ETF_NEAREST, ISampler::ETF_NEAREST, ISampler::ESMM_NEAREST, 0u, false, ECO_ALWAYS };
	auto sampler1 = device->createGPUSampler(samplerParams1);
	
	auto descriptorSet2 = device->createGPUDescriptorSet(descriptorPool.get(), core::smart_refctd_ptr(gpuDescriptorSetLayout2));
	{
		constexpr auto kDescriptorCount = 3;
		IGPUDescriptorSet::SWriteDescriptorSet samplerWriteDescriptorSet[kDescriptorCount];
		IGPUDescriptorSet::SDescriptorInfo samplerDescriptorInfo[kDescriptorCount];
		for (auto i=0; i<kDescriptorCount; i++)
		{
			samplerWriteDescriptorSet[i].dstSet = descriptorSet2.get();
			samplerWriteDescriptorSet[i].binding = i;
			samplerWriteDescriptorSet[i].arrayElement = 0u;
			samplerWriteDescriptorSet[i].count = 1u;
			samplerWriteDescriptorSet[i].info = samplerDescriptorInfo+i;
		}
		samplerWriteDescriptorSet[0].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
		samplerWriteDescriptorSet[1].descriptorType = EDT_UNIFORM_TEXEL_BUFFER;
		samplerWriteDescriptorSet[2].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;

		samplerDescriptorInfo[0].desc = gpuEnvmapImageView;
		{
			// ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_LINEAR, ISampler::ETF_LINEAR, ISampler::ESMM_LINEAR, 0u, false, ECO_ALWAYS };
			samplerDescriptorInfo[0].image.sampler = sampler0;
			samplerDescriptorInfo[0].image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;
		}
		samplerDescriptorInfo[1].desc = gpuSequenceBufferView;
		samplerDescriptorInfo[2].desc = gpuScrambleImageView;
		{
			// ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_INT_OPAQUE_BLACK, ISampler::ETF_NEAREST, ISampler::ETF_NEAREST, ISampler::ESMM_NEAREST, 0u, false, ECO_ALWAYS };
			samplerDescriptorInfo[2].image.sampler = sampler1;
			samplerDescriptorInfo[2].image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;
		}

		device->updateDescriptorSets(kDescriptorCount, samplerWriteDescriptorSet, 0u, nullptr);
	}

	constexpr uint32_t FRAME_COUNT = 500000u;

	core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
	for (uint32_t i=0u; i<FRAMES_IN_FLIGHT; i++)
	{
		imageAcquire[i] = device->createSemaphore();
		renderFinished[i] = device->createSemaphore();
	}
	
	CDumbPresentationOracle oracle;
	oracle.reportBeginFrameRecord();
	constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;
	
	// polling for events!
	CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> mouse;
	CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;
	
	uint32_t resourceIx = 0;
	while(windowCb->isWindowOpen())
	{
		resourceIx++;
		if(resourceIx >= FRAMES_IN_FLIGHT) {
			resourceIx = 0;
		}
		
		oracle.reportEndFrameRecord();
		double dt = oracle.getDeltaTimeInMicroSeconds() / 1000.0;
		auto nextPresentationTimeStamp = oracle.getNextPresentationTimeStamp();
		oracle.reportBeginFrameRecord();

		// Input 
		inputSystem->getDefaultMouse(&mouse);
		inputSystem->getDefaultKeyboard(&keyboard);

		cam.beginInputProcessing(nextPresentationTimeStamp);
		mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { cam.mouseProcess(events); }, logger.get());
		keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { cam.keyboardProcess(events); }, logger.get());
		cam.endInputProcessing(nextPresentationTimeStamp);
		
		auto& cb = cmdbuf[resourceIx];
		auto& fence = frameComplete[resourceIx];
		if (fence)
		while (device->waitForFences(1u,&fence.get(),false,MAX_TIMEOUT)==video::IGPUFence::ES_TIMEOUT)
		{
		}
		else
			fence = device->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));
		
		const auto viewMatrix = cam.getViewMatrix();
		const auto viewProjectionMatrix = cam.getConcatenatedMatrix();
				
		// safe to proceed
		cb->begin(0);

		// renderpass 
		uint32_t imgnum = 0u;
		swapchain->acquireNextImage(MAX_TIMEOUT,imageAcquire[resourceIx].get(),nullptr,&imgnum);
		{
			auto mv = viewMatrix;
			auto mvp = viewProjectionMatrix;
			core::matrix3x4SIMD normalMat;
			mv.getSub3x3InverseTranspose(normalMat);

			SBasicViewParametersAligned viewParams;
			memcpy(viewParams.uboData.MV, mv.pointer(), sizeof(mv));
			memcpy(viewParams.uboData.MVP, mvp.pointer(), sizeof(mvp));
			memcpy(viewParams.uboData.NormalMat, normalMat.pointer(), sizeof(normalMat));
			
			asset::SBufferRange<video::IGPUBuffer> range;
			range.buffer = gpuubo;
			range.offset = 0ull;
			range.size = sizeof(viewParams);
			utilities->updateBufferRangeViaStagingBuffer(graphicsQueue, range, &viewParams);
		}
				
		// TRANSITION outHDRImageViews[imgnum] to EIL_GENERAL (because of descriptorSets0 -> ComputeShader Writes into the image)
		{
			IGPUCommandBuffer::SImageMemoryBarrier imageBarriers[3u] = {};
			imageBarriers[0].barrier.srcAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0u);
			imageBarriers[0].barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(asset::EAF_SHADER_WRITE_BIT);
			imageBarriers[0].oldLayout = asset::EIL_UNDEFINED;
			imageBarriers[0].newLayout = asset::EIL_GENERAL;
			imageBarriers[0].srcQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
			imageBarriers[0].dstQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
			imageBarriers[0].image = outHDRImageViews[imgnum]->getCreationParameters().image;
			imageBarriers[0].subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			imageBarriers[0].subresourceRange.baseMipLevel = 0u;
			imageBarriers[0].subresourceRange.levelCount = 1;
			imageBarriers[0].subresourceRange.baseArrayLayer = 0u;
			imageBarriers[0].subresourceRange.layerCount = 1;

			imageBarriers[1].barrier.srcAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0u);
			imageBarriers[1].barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(asset::EAF_SHADER_READ_BIT);
			imageBarriers[1].oldLayout = asset::EIL_UNDEFINED;
			imageBarriers[1].newLayout = asset::EIL_SHADER_READ_ONLY_OPTIMAL;
			imageBarriers[1].srcQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
			imageBarriers[1].dstQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
			imageBarriers[1].image = gpuScrambleImageView->getCreationParameters().image;
			imageBarriers[1].subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			imageBarriers[1].subresourceRange.baseMipLevel = 0u;
			imageBarriers[1].subresourceRange.levelCount = 1;
			imageBarriers[1].subresourceRange.baseArrayLayer = 0u;
			imageBarriers[1].subresourceRange.layerCount = 1;

			 imageBarriers[2].barrier.srcAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0u);
			 imageBarriers[2].barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(asset::EAF_SHADER_READ_BIT);
			 imageBarriers[2].oldLayout = asset::EIL_UNDEFINED;
			 imageBarriers[2].newLayout = asset::EIL_SHADER_READ_ONLY_OPTIMAL;
			 imageBarriers[2].srcQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
			 imageBarriers[2].dstQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
			 imageBarriers[2].image = gpuEnvmapImageView->getCreationParameters().image;
			 imageBarriers[2].subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			 imageBarriers[2].subresourceRange.baseMipLevel = 0u;
			 imageBarriers[2].subresourceRange.levelCount = gpuEnvmapImageView->getCreationParameters().subresourceRange.levelCount;
			 imageBarriers[2].subresourceRange.baseArrayLayer = 0u;
			 imageBarriers[2].subresourceRange.layerCount = gpuEnvmapImageView->getCreationParameters().subresourceRange.layerCount;

			cb->pipelineBarrier(asset::EPSF_TOP_OF_PIPE_BIT, asset::EPSF_COMPUTE_SHADER_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 3u, imageBarriers);
		}

		// cube envmap handle
		{
			cb->bindComputePipeline(gpuComputePipeline.get());
			cb->bindDescriptorSets(EPBP_COMPUTE, gpuComputePipeline->getLayout(), 0u, 1u, &descriptorSets0[imgnum].get(), nullptr);
			cb->bindDescriptorSets(EPBP_COMPUTE, gpuComputePipeline->getLayout(), 1u, 1u, &uboDescriptorSet1.get(), nullptr);
			cb->bindDescriptorSets(EPBP_COMPUTE, gpuComputePipeline->getLayout(), 2u, 1u, &descriptorSet2.get(), nullptr);
			cb->dispatch(dispatchInfo.workGroupCount[0], dispatchInfo.workGroupCount[1], dispatchInfo.workGroupCount[2]);
		}
		// TODO: tone mapping and stuff

		// Copy HDR Image to SwapChain
		auto srcImgViewCreationParams = outHDRImageViews[imgnum]->getCreationParameters();
		auto dstImgViewCreationParams = fbo[imgnum]->getCreationParameters().attachments[0]->getCreationParameters();
		
		// Getting Ready for Blit
		// TRANSITION outHDRImageViews[imgnum] to EIL_TRANSFER_SRC_OPTIMAL
		// TRANSITION `fbo[imgnum]->getCreationParameters().attachments[0]` to EIL_TRANSFER_DST_OPTIMAL
		{
			IGPUCommandBuffer::SImageMemoryBarrier imageBarriers[2u] = {};
			imageBarriers[0].barrier.srcAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0u);
			imageBarriers[0].barrier.dstAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
			imageBarriers[0].oldLayout = asset::EIL_UNDEFINED;
			imageBarriers[0].newLayout = asset::EIL_TRANSFER_SRC_OPTIMAL;
			imageBarriers[0].srcQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
			imageBarriers[0].dstQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
			imageBarriers[0].image = srcImgViewCreationParams.image;
			imageBarriers[0].subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			imageBarriers[0].subresourceRange.baseMipLevel = 0u;
			imageBarriers[0].subresourceRange.levelCount = 1;
			imageBarriers[0].subresourceRange.baseArrayLayer = 0u;
			imageBarriers[0].subresourceRange.layerCount = 1;

			imageBarriers[1].barrier.srcAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0u);
			imageBarriers[1].barrier.dstAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
			imageBarriers[1].oldLayout = asset::EIL_UNDEFINED;
			imageBarriers[1].newLayout = asset::EIL_TRANSFER_DST_OPTIMAL;
			imageBarriers[1].srcQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
			imageBarriers[1].dstQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
			imageBarriers[1].image = dstImgViewCreationParams.image;
			imageBarriers[1].subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			imageBarriers[1].subresourceRange.baseMipLevel = 0u;
			imageBarriers[1].subresourceRange.levelCount = 1;
			imageBarriers[1].subresourceRange.baseArrayLayer = 0u;
			imageBarriers[1].subresourceRange.layerCount = 1;
			cb->pipelineBarrier(asset::EPSF_TRANSFER_BIT, asset::EPSF_TRANSFER_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 2u, imageBarriers);
		}

		// Blit Image
		{
			SImageBlit blit = {};
			blit.srcOffsets[0] = {0, 0, 0};
			blit.srcOffsets[1] = {WIN_W, WIN_H, 1};
		
			blit.srcSubresource.aspectMask = srcImgViewCreationParams.subresourceRange.aspectMask;
			blit.srcSubresource.mipLevel = srcImgViewCreationParams.subresourceRange.baseMipLevel;
			blit.srcSubresource.baseArrayLayer = srcImgViewCreationParams.subresourceRange.baseArrayLayer;
			blit.srcSubresource.layerCount = srcImgViewCreationParams.subresourceRange.layerCount;
			blit.dstOffsets[0] = {0, 0, 0};
			blit.dstOffsets[1] = {WIN_W, WIN_H, 1};
			blit.dstSubresource.aspectMask = dstImgViewCreationParams.subresourceRange.aspectMask;
			blit.dstSubresource.mipLevel = dstImgViewCreationParams.subresourceRange.baseMipLevel;
			blit.dstSubresource.baseArrayLayer = dstImgViewCreationParams.subresourceRange.baseArrayLayer;
			blit.dstSubresource.layerCount = dstImgViewCreationParams.subresourceRange.layerCount;

			auto srcImg = srcImgViewCreationParams.image;
			auto dstImg = dstImgViewCreationParams.image;

			cb->blitImage(srcImg.get(), EIL_TRANSFER_SRC_OPTIMAL, dstImg.get(), EIL_TRANSFER_DST_OPTIMAL, 1u, &blit , ISampler::ETF_NEAREST);
		}
		
		// TRANSITION `fbo[imgnum]->getCreationParameters().attachments[0]` to EIL_PRESENT
		{
			IGPUCommandBuffer::SImageMemoryBarrier imageBarriers[1u] = {};
			imageBarriers[0].barrier.srcAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
			imageBarriers[0].barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0u);
			imageBarriers[0].oldLayout = asset::EIL_TRANSFER_DST_OPTIMAL;
			imageBarriers[0].newLayout = asset::EIL_PRESENT_SRC_KHR;
			imageBarriers[0].srcQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
			imageBarriers[0].dstQueueFamilyIndex = graphicsCmdPoolQueueFamIdx;
			imageBarriers[0].image = dstImgViewCreationParams.image;
			imageBarriers[0].subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			imageBarriers[0].subresourceRange.baseMipLevel = 0u;
			imageBarriers[0].subresourceRange.levelCount = 1;
			imageBarriers[0].subresourceRange.baseArrayLayer = 0u;
			imageBarriers[0].subresourceRange.layerCount = 1;
			cb->pipelineBarrier(asset::EPSF_TRANSFER_BIT, asset::EPSF_TOP_OF_PIPE_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, imageBarriers);
		}

		cb->end();
		device->resetFences(1, &fence.get());
		CommonAPI::Submit(device.get(), swapchain.get(), cb.get(), graphicsQueue, imageAcquire[resourceIx].get(), renderFinished[resourceIx].get(), fence.get());
		CommonAPI::Present(device.get(), swapchain.get(), graphicsQueue, renderFinished[resourceIx].get(), imgnum);
	}
	
	const auto& fboCreationParams = fbo[0]->getCreationParameters();
	auto gpuSourceImageView = fboCreationParams.attachments[0];

	device->waitIdle();

	// bool status = ext::ScreenShot::createScreenShot(device.get(), queues[decltype(initOutput)::EQT_TRANSFER_UP], renderFinished[0].get(), gpuSourceImageView.get(), assetManager.get(), "ScreenShot.png");
	// assert(status);

	return 0;
}
