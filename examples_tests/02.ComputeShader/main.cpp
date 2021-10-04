#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"

using namespace nbl;

int main()
{
	constexpr uint32_t WIN_W = 768u;
	constexpr uint32_t WIN_H = 512u;
	constexpr uint32_t MAX_SWAPCHAIN_IMAGE_COUNT = 8u;
	constexpr uint32_t SWAPCHAIN_IMAGE_COUNT = 3u; // Temporary, this will be gone as soon as CommonAPI::Init won't take in SC_IMAGE_COUNT template param
	constexpr uint32_t FRAMES_IN_FLIGHT = 2u;
	// static_assert(FRAMES_IN_FLIGHT>FBO_COUNT);

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

	const auto swapchainImageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_COLOR_ATTACHMENT_BIT | asset::IImage::EUF_STORAGE_BIT);
	const video::ISurface::SFormat surfaceFormat(asset::EF_B8G8R8A8_UNORM, asset::ECP_COUNT, asset::EOTF_UNKNOWN);

	// This creates FBOs with swapchain images but I don't really need them
	auto initResult = CommonAPI::Init(
		video::EAT_OPENGL,
		"02.ComputeShader",
		requiredInstanceFeatures,
		optionalInstanceFeatures,
		requiredDeviceFeatures,
		optionalDeviceFeatures,
		WIN_W, WIN_H, SWAPCHAIN_IMAGE_COUNT,
		swapchainImageUsage,
		surfaceFormat);

	auto computeCommandPool = std::move(initResult.commandPools[CommonAPI::InitOutput::EQT_COMPUTE]);

#if 0
	// Todo(achal): Pending bug investigation, when both API connections are created at
	// the same time
	core::smart_refctd_ptr<video::COpenGLConnection> api =
		video::COpenGLConnection::create(core::smart_refctd_ptr(system), 0, "02.ComputeShader", video::COpenGLDebugCallback(core::smart_refctd_ptr(logger)));

	core::smart_refctd_ptr<video::CSurfaceGLWin32> surface =
		video::CSurfaceGLWin32::create(core::smart_refctd_ptr(api),
			core::smart_refctd_ptr<ui::IWindowWin32>(static_cast<ui::IWindowWin32*>(window.get())));
#endif

	const auto swapchainImages = initResult.swapchain->getImages();
	const uint32_t swapchainImageCount = initResult.swapchain->getImageCount();

	core::smart_refctd_ptr<video::IGPUImageView> swapchainImageViews[MAX_SWAPCHAIN_IMAGE_COUNT];
	for (uint32_t i = 0u; i < swapchainImageCount; ++i)
	{
		auto img = swapchainImages.begin()[i];
		{
			video::IGPUImageView::SCreationParams viewParams;
			viewParams.format = img->getCreationParameters().format;
			viewParams.viewType = asset::IImageView<video::IGPUImage>::ET_2D;
			viewParams.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			viewParams.subresourceRange.baseMipLevel = 0u;
			viewParams.subresourceRange.levelCount = 1u;
			viewParams.subresourceRange.baseArrayLayer = 0u;
			viewParams.subresourceRange.layerCount = 1u;
			viewParams.image = core::smart_refctd_ptr<video::IGPUImage>(img);

			swapchainImageViews[i] = initResult.logicalDevice->createGPUImageView(std::move(viewParams));
			assert(swapchainImageViews[i]);
		}
	}

	video::IGPUObjectFromAssetConverter CPU2GPU;

	const char* pathToShader = "../compute.comp";
	core::smart_refctd_ptr<video::IGPUSpecializedShader> specializedShader = nullptr;
	{
		asset::IAssetLoader::SAssetLoadParams params = {};
		params.logger = initResult.logger.get();
		auto spec = (initResult.assetManager->getAsset(pathToShader, params).getContents());
		auto specShader_cpu = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*initResult.assetManager->getAsset(pathToShader, params).getContents().begin());
		specializedShader = CPU2GPU.getGPUObjectsFromAssets(&specShader_cpu, &specShader_cpu + 1, initResult.cpu2gpuParams)->front();
	}
	assert(specializedShader);

	core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[MAX_SWAPCHAIN_IMAGE_COUNT];
	initResult.logicalDevice->createCommandBuffers(computeCommandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY,
		swapchainImageCount, commandBuffers);

	const uint32_t bindingCount = 2u;
	video::IGPUDescriptorSetLayout::SBinding bindings[bindingCount];
	{
		// image2D
		bindings[0].binding = 0u;
		bindings[0].type = asset::EDT_STORAGE_IMAGE;
		bindings[0].count = 1u;
		bindings[0].stageFlags = asset::ISpecializedShader::ESS_COMPUTE;
		bindings[0].samplers = nullptr;

		// ubo
		bindings[1].binding = 1u;
		bindings[1].type = asset::EDT_STORAGE_IMAGE;
		bindings[1].count = 1u;
		bindings[1].stageFlags = asset::ISpecializedShader::ESS_COMPUTE;
		bindings[1].samplers = nullptr;
	}
	core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> dsLayout =
		initResult.logicalDevice->createGPUDescriptorSetLayout(bindings, bindings + bindingCount);

	const uint32_t descriptorPoolSizeCount = 1u;
	video::IDescriptorPool::SDescriptorPoolSize poolSizes[descriptorPoolSizeCount];
	poolSizes[0].type = asset::EDT_STORAGE_IMAGE;
	poolSizes[0].count = swapchainImageCount + 1u;

	video::IDescriptorPool::E_CREATE_FLAGS descriptorPoolFlags =
		static_cast<video::IDescriptorPool::E_CREATE_FLAGS>(0);

	core::smart_refctd_ptr<video::IDescriptorPool> descriptorPool
		= initResult.logicalDevice->createDescriptorPool(descriptorPoolFlags, swapchainImageCount,
			descriptorPoolSizeCount, poolSizes);

	// For each swapchain image we have one descriptor set with two descriptors each
	core::smart_refctd_ptr<video::IGPUDescriptorSet> descriptorSets[MAX_SWAPCHAIN_IMAGE_COUNT];

	for (uint32_t i = 0u; i < swapchainImageCount; ++i)
	{
		descriptorSets[i] = initResult.logicalDevice->createGPUDescriptorSet(descriptorPool.get(),
			core::smart_refctd_ptr(dsLayout));
	}

	// Uncomment once the KTX loader works
#if 0
	constexpr auto cachingFlags = static_cast<asset::IAssetLoader::E_CACHING_FLAGS>(
		asset::IAssetLoader::ECF_DONT_CACHE_REFERENCES & asset::IAssetLoader::ECF_DONT_CACHE_TOP_LEVEL);

	const char* pathToImage = "../../media/color_space_test/kueken7_rgba8_unorm.ktx";
	
	asset::IAssetLoader::SAssetLoadParams loadParams(0ull, nullptr, cachingFlags);
	auto cpuImageBundle = assetManager->getAsset(pathToImage, loadParams);
	auto cpuImageContents = cpuImageBundle.getContents();
	if (cpuImageContents.empty())
	{
		logger->log("Failed to read image at path %s", nbl::system::ILogger::ELL_ERROR, pathToImage);
		exit(-1);
	}

	auto cpuImage = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(*cpuImageContents.begin());
#else
	const uint32_t imageWidth = WIN_W;
	const uint32_t imageHeight = WIN_H;
	const uint32_t imageChannelCount = 4u;
	const uint32_t mipLevels = 1u; // WILL NOT WORK FOR MORE THAN 1 MIPS, but doesn't matter since it is temporary until KTX loading works
	const size_t imageSize = imageWidth * imageHeight * imageChannelCount * sizeof(uint8_t);
	auto imagePixels = core::make_smart_refctd_ptr<asset::ICPUBuffer>(imageSize);

	uint32_t* dstPixel = (uint32_t*)imagePixels->getPointer();
	for (uint32_t y = 0u; y < imageHeight; ++y)
	{
		for (uint32_t x = 0u; x < imageWidth; ++x)
		{
			// Should be red in R8G8B8A8_UNORM
			*dstPixel++ = 0x000000FF;
		}
	}

	core::smart_refctd_ptr<asset::ICPUImage> inImage_CPU = nullptr;
	{
		asset::ICPUImage::SCreationParams creationParams = {};
		creationParams.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
		creationParams.type = asset::IImage::ET_2D;
		creationParams.format = asset::EF_R8G8B8A8_UNORM;
		creationParams.extent = { imageWidth, imageHeight, 1u };
		creationParams.mipLevels = mipLevels;
		creationParams.arrayLayers = 1u;
		creationParams.samples = asset::IImage::ESCF_1_BIT;
		creationParams.tiling = asset::IImage::ET_OPTIMAL;
		// This API check is temporary (or not?) since getFormatProperties is not
		// yet implemented on OpenGL
		// (Or this check should belong inside the engine wherever image usages
		// are validated?)
		if (initResult.apiConnection->getAPIType() == video::EAT_VULKAN)
		{
			const auto& formatProps = initResult.physicalDevice->getFormatProperties(creationParams.format);
			assert(formatProps.optimalTilingFeatures.operator&(asset::EFF_STORAGE_IMAGE_BIT).value);
			assert(formatProps.optimalTilingFeatures.operator&(asset::EFF_SAMPLED_IMAGE_FILTER_LINEAR_BIT).value);
		}
		creationParams.usage = core::bitflag(asset::IImage::EUF_STORAGE_BIT) | asset::IImage::EUF_TRANSFER_DST_BIT;
		creationParams.sharingMode = asset::ESM_EXCLUSIVE;
		creationParams.queueFamilyIndexCount = 1u;
		creationParams.queueFamilyIndices = nullptr;
		creationParams.initialLayout = asset::EIL_UNDEFINED;

		auto imageRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::ICPUImage::SBufferCopy>>(1ull);
		imageRegions->begin()->bufferOffset = 0ull;
		imageRegions->begin()->bufferRowLength = creationParams.extent.width;
		imageRegions->begin()->bufferImageHeight = 0u;
		imageRegions->begin()->imageSubresource = {};
		imageRegions->begin()->imageSubresource.aspectMask = asset::IImage::EAF_COLOR_BIT;
		imageRegions->begin()->imageSubresource.layerCount = 1u;
		imageRegions->begin()->imageOffset = { 0, 0, 0 };
		imageRegions->begin()->imageExtent = { creationParams.extent.width, creationParams.extent.height, 1u };

		inImage_CPU = asset::ICPUImage::create(std::move(creationParams));
		inImage_CPU->setBufferAndRegions(core::smart_refctd_ptr<asset::ICPUBuffer>(imagePixels), imageRegions);
	}
#endif	

	initResult.cpu2gpuParams.beginCommandBuffers();
	auto inImage = CPU2GPU.getGPUObjectsFromAssets(&inImage_CPU, &inImage_CPU + 1, initResult.cpu2gpuParams);
	initResult.cpu2gpuParams.waitForCreationToComplete(false);
	assert(inImage);

	// Create an image view for input image
	core::smart_refctd_ptr<video::IGPUImageView> inImageView = nullptr;
	{
		video::IGPUImageView::SCreationParams viewParams;
		viewParams.format = inImage_CPU->getCreationParameters().format;
		viewParams.viewType = asset::IImageView<video::IGPUImage>::ET_2D;
		viewParams.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
		viewParams.subresourceRange.baseMipLevel = 0u;
		viewParams.subresourceRange.levelCount = 1u;
		viewParams.subresourceRange.baseArrayLayer = 0u;
		viewParams.subresourceRange.layerCount = 1u;
		viewParams.image = inImage->begin()[0];

		inImageView = initResult.logicalDevice->createGPUImageView(std::move(viewParams));
	}
	assert(inImageView);

	for (uint32_t i = 0u; i < swapchainImageCount; ++i)
	{
		const uint32_t writeDescriptorCount = 2u;

		video::IGPUDescriptorSet::SDescriptorInfo descriptorInfos[writeDescriptorCount];
		video::IGPUDescriptorSet::SWriteDescriptorSet writeDescriptorSets[writeDescriptorCount] = {};

		// image2D -- swapchain image
		{
			descriptorInfos[0].image.imageLayout = asset::EIL_GENERAL;
			descriptorInfos[0].image.sampler = nullptr;
			descriptorInfos[0].desc = swapchainImageViews[i]; // shouldn't IGPUDescriptorSet hold a reference to the resources in its descriptors?

			writeDescriptorSets[0].dstSet = descriptorSets[i].get();
			writeDescriptorSets[0].binding = 0u;
			writeDescriptorSets[0].arrayElement = 0u;
			writeDescriptorSets[0].count = 1u;
			writeDescriptorSets[0].descriptorType = asset::EDT_STORAGE_IMAGE;
			writeDescriptorSets[0].info = &descriptorInfos[0];
		}

		// image2D -- my input
		{
			descriptorInfos[1].image.imageLayout = asset::EIL_GENERAL;
			descriptorInfos[1].image.sampler = nullptr;
			descriptorInfos[1].desc = inImageView;

			writeDescriptorSets[1].dstSet = descriptorSets[i].get();
			writeDescriptorSets[1].binding = 1u;
			writeDescriptorSets[1].arrayElement = 0u;
			writeDescriptorSets[1].count = 1u;
			writeDescriptorSets[1].descriptorType = asset::EDT_STORAGE_IMAGE;
			writeDescriptorSets[1].info = &descriptorInfos[1];
		}

		initResult.logicalDevice->updateDescriptorSets(writeDescriptorCount, writeDescriptorSets, 0u, nullptr);
	}

	asset::SPushConstantRange pcRange = {};
	pcRange.stageFlags = asset::ISpecializedShader::ESS_COMPUTE;
	pcRange.offset = 0u;
	pcRange.size = 2*sizeof(uint32_t);
	core::smart_refctd_ptr<video::IGPUPipelineLayout> pipelineLayout =
		initResult.logicalDevice->createGPUPipelineLayout(&pcRange, &pcRange + 1, core::smart_refctd_ptr(dsLayout));

	core::smart_refctd_ptr<video::IGPUComputePipeline> pipeline =
		initResult.logicalDevice->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(pipelineLayout),
			core::smart_refctd_ptr(specializedShader));

	core::smart_refctd_ptr<video::IGPUSemaphore> acquireSemaphores[FRAMES_IN_FLIGHT];
	core::smart_refctd_ptr<video::IGPUSemaphore> releaseSemaphores[FRAMES_IN_FLIGHT];
	core::smart_refctd_ptr<video::IGPUFence> frameFences[FRAMES_IN_FLIGHT];
	for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; ++i)
	{
		acquireSemaphores[i] = initResult.logicalDevice->createSemaphore();
		releaseSemaphores[i] = initResult.logicalDevice->createSemaphore();
		frameFences[i] = initResult.logicalDevice->createFence(video::IGPUFence::E_CREATE_FLAGS::ECF_SIGNALED_BIT);
	}

	const uint32_t windowDim[2] = { initResult.window->getWidth(), initResult.window->getHeight() };

	video::IGPUCommandBuffer::SImageMemoryBarrier layoutTransBarrier = {};
	layoutTransBarrier.srcQueueFamilyIndex = ~0u;
	layoutTransBarrier.dstQueueFamilyIndex = ~0u;
	layoutTransBarrier.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
	layoutTransBarrier.subresourceRange.baseMipLevel = 0u;
	layoutTransBarrier.subresourceRange.levelCount = 1u;
	layoutTransBarrier.subresourceRange.baseArrayLayer = 0u;
	layoutTransBarrier.subresourceRange.layerCount = 1u;

	for (uint32_t i = 0u; i < swapchainImageCount; ++i)
	{
		commandBuffers[i]->begin(0);

		layoutTransBarrier.barrier.srcAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0);
		layoutTransBarrier.barrier.dstAccessMask = asset::EAF_SHADER_READ_BIT;
		layoutTransBarrier.oldLayout = asset::EIL_UNDEFINED;
		layoutTransBarrier.newLayout = asset::EIL_GENERAL;
		layoutTransBarrier.image = *(swapchainImages.begin() + i);

		commandBuffers[i]->pipelineBarrier(asset::EPSF_TOP_OF_PIPE_BIT,
			asset::EPSF_COMPUTE_SHADER_BIT, static_cast<asset::E_DEPENDENCY_FLAGS>(0u), 0u,
			nullptr, 0u, nullptr, 1u, &layoutTransBarrier);

		const video::IGPUDescriptorSet* tmp[] = { descriptorSets[i].get() };
		commandBuffers[i]->bindDescriptorSets(asset::EPBP_COMPUTE, pipelineLayout.get(),
			0u, 1u, tmp);

		commandBuffers[i]->bindComputePipeline(pipeline.get());

		commandBuffers[i]->pushConstants(pipelineLayout.get(), pcRange.stageFlags, pcRange.offset, pcRange.size, windowDim);

		commandBuffers[i]->dispatch((WIN_W + 15u) / 16u, (WIN_H + 15u) / 16u, 1u);

		layoutTransBarrier.barrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
		layoutTransBarrier.barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0);
		layoutTransBarrier.oldLayout = asset::EIL_GENERAL;
		layoutTransBarrier.newLayout = asset::EIL_PRESENT_SRC_KHR;

		commandBuffers[i]->pipelineBarrier(asset::EPSF_COMPUTE_SHADER_BIT, asset::EPSF_BOTTOM_OF_PIPE_BIT,
			static_cast<asset::E_DEPENDENCY_FLAGS>(0u), 0u, nullptr, 0u, nullptr,
			1u, &layoutTransBarrier);

		commandBuffers[i]->end();
	}

	video::ISwapchain* rawPointerToSwapchain = initResult.swapchain.get();
	
	uint32_t currentFrameIndex = 0u;
	while (initResult.windowCb->isWindowOpen())
	{
		video::IGPUSemaphore* acquireSemaphore_frame = acquireSemaphores[currentFrameIndex].get();
		video::IGPUSemaphore* releaseSemaphore_frame = releaseSemaphores[currentFrameIndex].get();
		video::IGPUFence* fence_frame = frameFences[currentFrameIndex].get();

		video::IGPUFence::E_STATUS retval = initResult.logicalDevice->waitForFences(1u, &fence_frame, true, ~0ull);
		assert(retval == video::IGPUFence::ES_SUCCESS);

		uint32_t imageIndex;
		initResult.swapchain->acquireNextImage(~0ull, acquireSemaphores[currentFrameIndex].get(), nullptr,
			&imageIndex);

		initResult.logicalDevice->resetFences(1u, &fence_frame);

		CommonAPI::Submit(
			initResult.logicalDevice.get(),
			initResult.swapchain.get(),
			commandBuffers[imageIndex].get(),
			initResult.queues[CommonAPI::InitOutput<MAX_SWAPCHAIN_IMAGE_COUNT>::EQT_COMPUTE],
			acquireSemaphore_frame,
			releaseSemaphore_frame,
			fence_frame);

		CommonAPI::Present(
			initResult.logicalDevice.get(),
			initResult.swapchain.get(),
			initResult.queues[CommonAPI::InitOutput::EQT_COMPUTE],
			releaseSemaphore_frame, imageIndex);

		currentFrameIndex = (currentFrameIndex + 1) % FRAMES_IN_FLIGHT;
	}

	initResult.logicalDevice->waitIdle();

	return 0;
}