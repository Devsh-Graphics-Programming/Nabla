#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"

using namespace nbl;

class ComputeShaderSampleApp : public ApplicationBase
{
	constexpr static uint32_t WIN_W = 768u;
	constexpr static uint32_t WIN_H = 512u;
	constexpr static uint32_t SC_IMG_COUNT = 3u;
	constexpr static uint32_t FRAMES_IN_FLIGHT = 5u;
	static constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;
	static_assert(FRAMES_IN_FLIGHT>SC_IMG_COUNT);

	core::smart_refctd_ptr<nbl::ui::IWindowManager> windowManager;
	core::smart_refctd_ptr<nbl::ui::IWindow> window;
	core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCb;
	core::smart_refctd_ptr<nbl::video::IAPIConnection> apiConnection;
	core::smart_refctd_ptr<nbl::video::ISurface> surface;
	core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
	core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
	video::IPhysicalDevice* physicalDevice;
	std::array<video::IGPUQueue*, CommonAPI::InitOutput::MaxQueuesCount> queues;
	core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
	core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
	std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, CommonAPI::InitOutput::MaxSwapChainImageCount> fbo;
	std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, CommonAPI::InitOutput::MaxQueuesCount> commandPools;
	core::smart_refctd_ptr<nbl::system::ISystem> system;
	core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
	video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
	core::smart_refctd_ptr<nbl::system::ILogger> logger;
	core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;
	video::IGPUObjectFromAssetConverter cpu2gpu;

	int32_t m_resourceIx = -1;

	core::smart_refctd_ptr<video::IGPUSemaphore> m_imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> m_renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUFence> m_frameComplete[FRAMES_IN_FLIGHT] = { nullptr };

	core::smart_refctd_ptr<video::IGPUCommandBuffer> m_cmdbuf[FRAMES_IN_FLIGHT] = { nullptr };

	core::smart_refctd_ptr<video::IGPUComputePipeline> m_pipeline = nullptr;
	core::vector<core::smart_refctd_ptr<video::IGPUDescriptorSet>> m_descriptorSets;

	core::vector<core::smart_refctd_ptr<video::IGPUImageView>> m_swapchainImageViews;
	core::smart_refctd_ptr<video::IGPUImageView> m_inImageView;

public:
	void setWindow(core::smart_refctd_ptr<nbl::ui::IWindow>&& wnd) override
	{
		window = std::move(wnd);
	}
	nbl::ui::IWindow* getWindow() override
	{
		return window.get();
	}
	void setSystem(core::smart_refctd_ptr<nbl::system::ISystem>&& system) override
	{
		system = std::move(system);
	}

	APP_CONSTRUCTOR(ComputeShaderSampleApp);

	void onAppInitialized_impl() override
	{
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

		const auto swapchainImageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_COLOR_ATTACHMENT_BIT | asset::IImage::EUF_STORAGE_BIT);
		const video::ISurface::SFormat surfaceFormat(asset::EF_B8G8R8A8_UNORM, asset::ECP_COUNT, asset::EOTF_UNKNOWN);

		CommonAPI::InitOutput initOutput;
		initOutput.window = core::smart_refctd_ptr(window);
		CommonAPI::Init(
			initOutput,
			video::EAT_VULKAN,
			"02.ComputeShader",
			requiredInstanceFeatures,
			optionalInstanceFeatures,
			requiredDeviceFeatures,
			optionalDeviceFeatures,
			WIN_W, WIN_H, SC_IMG_COUNT,
			swapchainImageUsage,
			surfaceFormat);

		system = std::move(initOutput.system);
		window = std::move(initOutput.window);
		windowCb = std::move(initOutput.windowCb);
		apiConnection = std::move(initOutput.apiConnection);
		surface = std::move(initOutput.surface);
		physicalDevice = std::move(initOutput.physicalDevice);
		logicalDevice = std::move(initOutput.logicalDevice);
		utilities = std::move(initOutput.utilities);
		queues = std::move(initOutput.queues);
		swapchain = std::move(initOutput.swapchain);
		renderpass = std::move(initOutput.renderpass);
		fbo = std::move(initOutput.fbo);
		commandPools = std::move(initOutput.commandPools);
		assetManager = std::move(initOutput.assetManager);
		cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
		logger = std::move(initOutput.logger);
		inputSystem = std::move(initOutput.inputSystem);

		const auto& computeCommandPool = commandPools[CommonAPI::InitOutput::EQT_COMPUTE];

#if 0
		// Todo(achal): Pending bug investigation, when both API connections are created at
		// the same time
		core::smart_refctd_ptr<video::COpenGLConnection> api =
			video::COpenGLConnection::create(core::smart_refctd_ptr(system), 0, "02.ComputeShader", video::COpenGLDebugCallback(core::smart_refctd_ptr(logger)));

		core::smart_refctd_ptr<video::CSurfaceGLWin32> surface =
			video::CSurfaceGLWin32::create(core::smart_refctd_ptr(api),
				core::smart_refctd_ptr<ui::IWindowWin32>(static_cast<ui::IWindowWin32*>(window.get())));
#endif

		const auto swapchainImages = swapchain->getImages();
		const uint32_t swapchainImageCount = swapchain->getImageCount();

		m_swapchainImageViews.resize(swapchainImageCount);
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

				m_swapchainImageViews[i] = logicalDevice->createGPUImageView(std::move(viewParams));
				assert(m_swapchainImageViews[i]);
			}
		}

		video::IGPUObjectFromAssetConverter CPU2GPU;

		const char* pathToShader = "../compute.comp";
		core::smart_refctd_ptr<video::IGPUSpecializedShader> specializedShader = nullptr;
		{
			asset::IAssetLoader::SAssetLoadParams params = {};
			params.logger = logger.get();
			auto spec = (assetManager->getAsset(pathToShader, params).getContents());
			auto specShader_cpu = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*assetManager->getAsset(pathToShader, params).getContents().begin());
			specializedShader = CPU2GPU.getGPUObjectsFromAssets(&specShader_cpu, &specShader_cpu + 1, cpu2gpuParams)->front();
		}
		assert(specializedShader);

		logicalDevice->createCommandBuffers(
			computeCommandPool.get(),
			video::IGPUCommandBuffer::EL_PRIMARY,
			FRAMES_IN_FLIGHT,
			m_cmdbuf);

		const uint32_t bindingCount = 2u;
		video::IGPUDescriptorSetLayout::SBinding bindings[bindingCount];
		{
			// image2D
			bindings[0].binding = 0u;
			bindings[0].type = asset::EDT_STORAGE_IMAGE;
			bindings[0].count = 1u;
			bindings[0].stageFlags = asset::IShader::ESS_COMPUTE;
			bindings[0].samplers = nullptr;

			// ubo
			bindings[1].binding = 1u;
			bindings[1].type = asset::EDT_STORAGE_IMAGE;
			bindings[1].count = 1u;
			bindings[1].stageFlags = asset::IShader::ESS_COMPUTE;
			bindings[1].samplers = nullptr;
		}
		core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> dsLayout =
			logicalDevice->createGPUDescriptorSetLayout(bindings, bindings + bindingCount);

		const uint32_t descriptorPoolSizeCount = 1u;
		video::IDescriptorPool::SDescriptorPoolSize poolSizes[descriptorPoolSizeCount];
		poolSizes[0].type = asset::EDT_STORAGE_IMAGE;
		poolSizes[0].count = swapchainImageCount + 1u;

		video::IDescriptorPool::E_CREATE_FLAGS descriptorPoolFlags =
			static_cast<video::IDescriptorPool::E_CREATE_FLAGS>(0);

		core::smart_refctd_ptr<video::IDescriptorPool> descriptorPool
			= logicalDevice->createDescriptorPool(descriptorPoolFlags, swapchainImageCount,
				descriptorPoolSizeCount, poolSizes);

		m_descriptorSets.resize(swapchainImageCount);
		for (uint32_t i = 0u; i < swapchainImageCount; ++i)
		{
			m_descriptorSets[i] = logicalDevice->createGPUDescriptorSet(descriptorPool.get(),
				core::smart_refctd_ptr(dsLayout));
		}
		
		// Uncomment once the KTX loader works
		constexpr auto cachingFlags = static_cast<asset::IAssetLoader::E_CACHING_FLAGS>(
			asset::IAssetLoader::ECF_DONT_CACHE_REFERENCES & asset::IAssetLoader::ECF_DONT_CACHE_TOP_LEVEL);

		const char* pathToImage = "../../media/color_space_test/kueken7_rgba8_unorm.ktx";
		// const char* pathToImage = "../../media/color_space_test/R8G8B8_1.jpg";

		asset::IAssetLoader::SAssetLoadParams loadParams(0ull, nullptr, cachingFlags);
		auto cpuImageBundle = assetManager->getAsset(pathToImage, loadParams);
		auto cpuImageContents = cpuImageBundle.getContents();
		if (cpuImageContents.empty())
		{
			logger->log("Failed to read image at path %s", nbl::system::ILogger::ELL_ERROR, pathToImage);
			exit(-1);
		}
		__debugbreak();
#if 0

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
			if (apiConnection->getAPIType() == video::EAT_VULKAN)
			{
				const auto& formatProps = physicalDevice->getFormatProperties(creationParams.format);
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

		cpu2gpuParams.beginCommandBuffers();
		auto inImage = CPU2GPU.getGPUObjectsFromAssets(&inImage_CPU, &inImage_CPU + 1, cpu2gpuParams);
		cpu2gpuParams.waitForCreationToComplete(false);
		assert(inImage);

		// Create an image view for input image
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

			m_inImageView = logicalDevice->createGPUImageView(std::move(viewParams));
		}
		assert(m_inImageView);

		for (uint32_t i = 0u; i < swapchainImageCount; ++i)
		{
			const uint32_t writeDescriptorCount = 2u;

			video::IGPUDescriptorSet::SDescriptorInfo descriptorInfos[writeDescriptorCount];
			video::IGPUDescriptorSet::SWriteDescriptorSet writeDescriptorSets[writeDescriptorCount] = {};

			// image2D -- swapchain image
			{
				descriptorInfos[0].image.imageLayout = asset::EIL_GENERAL;
				descriptorInfos[0].image.sampler = nullptr;
				descriptorInfos[0].desc = m_swapchainImageViews[i]; // shouldn't IGPUDescriptorSet hold a reference to the resources in its descriptors?

				writeDescriptorSets[0].dstSet = m_descriptorSets[i].get();
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
				descriptorInfos[1].desc = m_inImageView;

				writeDescriptorSets[1].dstSet = m_descriptorSets[i].get();
				writeDescriptorSets[1].binding = 1u;
				writeDescriptorSets[1].arrayElement = 0u;
				writeDescriptorSets[1].count = 1u;
				writeDescriptorSets[1].descriptorType = asset::EDT_STORAGE_IMAGE;
				writeDescriptorSets[1].info = &descriptorInfos[1];
			}

			logicalDevice->updateDescriptorSets(writeDescriptorCount, writeDescriptorSets, 0u, nullptr);
		}

		asset::SPushConstantRange pcRange = {};
		pcRange.stageFlags = asset::IShader::ESS_COMPUTE;
		pcRange.offset = 0u;
		pcRange.size = 2 * sizeof(uint32_t);
		core::smart_refctd_ptr<video::IGPUPipelineLayout> pipelineLayout =
			logicalDevice->createGPUPipelineLayout(&pcRange, &pcRange + 1, core::smart_refctd_ptr(dsLayout));

		m_pipeline = logicalDevice->createGPUComputePipeline(nullptr,
			core::smart_refctd_ptr(pipelineLayout), core::smart_refctd_ptr(specializedShader));

		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
		{
			m_imageAcquire[i] = logicalDevice->createSemaphore();
			m_renderFinished[i] = logicalDevice->createSemaphore();
		}
	}

	void onAppTerminated_impl() override
	{
		logicalDevice->waitIdle();
	}

	void workLoopBody() override
	{
		m_resourceIx++;
		if (m_resourceIx >= FRAMES_IN_FLIGHT)
			m_resourceIx = 0;

		auto& cb = m_cmdbuf[m_resourceIx];
		auto& fence = m_frameComplete[m_resourceIx];
		if (fence)
		{
			while (logicalDevice->waitForFences(1u, &fence.get(), false, MAX_TIMEOUT) == video::IGPUFence::ES_TIMEOUT)
			{
			}
			logicalDevice->resetFences(1u, &fence.get());
		}
		else
			fence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

		// acquire image 
		uint32_t imgnum = 0u;
		swapchain->acquireNextImage(MAX_TIMEOUT, m_imageAcquire[m_resourceIx].get(), nullptr, &imgnum);

		// safe to proceed
		cb->begin(0);

		{
			asset::SViewport vp;
			vp.minDepth = 1.f;
			vp.maxDepth = 0.f;
			vp.x = 0u;
			vp.y = 0u;
			vp.width = WIN_W;
			vp.height = WIN_H;
			cb->setViewport(0u, 1u, &vp);

			VkRect2D scissor;
			scissor.extent = { WIN_W, WIN_H };
			scissor.offset = { 0, 0 };
			cb->setScissor(0u, 1u, &scissor);
		}

		video::IGPUCommandBuffer::SImageMemoryBarrier layoutTransBarrier = {};
		layoutTransBarrier.srcQueueFamilyIndex = ~0u;
		layoutTransBarrier.dstQueueFamilyIndex = ~0u;
		layoutTransBarrier.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
		layoutTransBarrier.subresourceRange.baseMipLevel = 0u;
		layoutTransBarrier.subresourceRange.levelCount = 1u;
		layoutTransBarrier.subresourceRange.baseArrayLayer = 0u;
		layoutTransBarrier.subresourceRange.layerCount = 1u;

		const uint32_t windowDim[2] = { window->getWidth(), window->getHeight() };

		layoutTransBarrier.barrier.srcAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0);
		layoutTransBarrier.barrier.dstAccessMask = asset::EAF_SHADER_READ_BIT;
		layoutTransBarrier.oldLayout = asset::EIL_UNDEFINED;
		layoutTransBarrier.newLayout = asset::EIL_GENERAL;
		layoutTransBarrier.image = *(swapchain->getImages().begin() + imgnum);

		cb->pipelineBarrier(
			asset::EPSF_TOP_OF_PIPE_BIT,
			asset::EPSF_COMPUTE_SHADER_BIT,
			static_cast<asset::E_DEPENDENCY_FLAGS>(0u),
			0u, nullptr,
			0u, nullptr,
			1u, &layoutTransBarrier);

		const video::IGPUDescriptorSet* tmp[] = { m_descriptorSets[imgnum].get() };
		cb->bindDescriptorSets(asset::EPBP_COMPUTE, m_pipeline->getLayout(), 0u, 1u, tmp);

		cb->bindComputePipeline(m_pipeline.get());
		
		const asset::SPushConstantRange& pcRange = m_pipeline->getLayout()->getPushConstantRanges().begin()[0];
		cb->pushConstants(m_pipeline->getLayout(), pcRange.stageFlags, pcRange.offset, pcRange.size, windowDim);

		cb->dispatch((WIN_W + 15u) / 16u, (WIN_H + 15u) / 16u, 1u);

		layoutTransBarrier.barrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
		layoutTransBarrier.barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0);
		layoutTransBarrier.oldLayout = asset::EIL_GENERAL;
		layoutTransBarrier.newLayout = asset::EIL_PRESENT_SRC_KHR;

		cb->pipelineBarrier(
			asset::EPSF_COMPUTE_SHADER_BIT,
			asset::EPSF_BOTTOM_OF_PIPE_BIT,
			static_cast<asset::E_DEPENDENCY_FLAGS>(0u),
			0u, nullptr,
			0u, nullptr,
			1u, &layoutTransBarrier);

		cb->end();

		CommonAPI::Submit(
			logicalDevice.get(),
			swapchain.get(),
			cb.get(),
			queues[CommonAPI::InitOutput::EQT_COMPUTE],
			m_imageAcquire[m_resourceIx].get(),
			m_renderFinished[m_resourceIx].get(),
			fence.get());

		CommonAPI::Present(
			logicalDevice.get(),
			swapchain.get(),
			queues[CommonAPI::InitOutput::EQT_COMPUTE],
			m_renderFinished[m_resourceIx].get(),
			imgnum);
	}

	bool keepRunning() override
	{
		return windowCb->isWindowOpen();
	}
};

NBL_COMMON_API_MAIN(ComputeShaderSampleApp)

extern "C" {  _declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001; }