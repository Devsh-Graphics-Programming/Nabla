// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

/**
This example just shows a screen which clears to red,
nothing fancy, just to show that Irrlicht links fine
**/
#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nabla.h>

#if defined(_NBL_PLATFORM_WINDOWS_)
#	include <nbl/system/CColoredStdoutLoggerWin32.h>
#endif // TODO more platforms
// TODO: make these include themselves via `nabla.h`

using namespace nbl;

#define LOG(...) printf(__VA_ARGS__); printf("\n");
class DemoEventCallback : public nbl::ui::IWindow::IEventCallback
{
	bool onWindowShown_impl() override
	{
		LOG("Window Shown");
		return true;
	}
	bool onWindowHidden_impl() override
	{
		LOG("Window hidden");
		return true;
	}
	bool onWindowMoved_impl(int32_t x, int32_t y) override
	{
		LOG("Window window moved to { %d, %d }", x, y);
		return true;
	}
	bool onWindowResized_impl(uint32_t w, uint32_t h) override
	{
		LOG("Window resized to { %u, %u }", w, h);
		return true;
	}
	bool onWindowMinimized_impl() override
	{
		LOG("Window minimized");
		return true;
	}
	bool onWindowMaximized_impl() override
	{
		LOG("Window maximized");
		return true;
	}
	void onGainedMouseFocus_impl() override
	{
		LOG("Window gained mouse focus");
	}
	void onLostMouseFocus_impl() override
	{
		LOG("Window lost mouse focus");
	}
	void onGainedKeyboardFocus_impl() override
	{
		LOG("Window gained keyboard focus");
	}
	void onLostKeyboardFocus_impl() override
	{
		LOG("Window lost keyboard focus");
	}
	bool onWindowClosed_impl() override
	{
		LOG("Window closed");
		m_gotWindowClosedMsg = true;
		return true;
	}
	void onMouseConnected_impl(core::smart_refctd_ptr<nbl::ui::IMouseEventChannel>&& mch) override
	{
		LOG("A mouse has been connected");
	}
	void onMouseDisconnected_impl(nbl::ui::IMouseEventChannel* mch) override
	{
		LOG("A mouse has been disconnected");
	}
	void onKeyboardConnected_impl(core::smart_refctd_ptr<nbl::ui::IKeyboardEventChannel>&& kbch) override
	{
		LOG("A keyboard has been connected");
	}
	void onKeyboardDisconnected_impl(nbl::ui::IKeyboardEventChannel* mch) override
	{
		LOG("A keyboard has been disconnected");
	}

public:
	bool m_gotWindowClosedMsg = false;
};

static core::smart_refctd_ptr<system::ISystem> createSystem()
{
	core::smart_refctd_ptr<system::ISystemCaller> caller = nullptr;
#ifdef _NBL_PLATFORM_WINDOWS_
	caller = core::make_smart_refctd_ptr<nbl::system::CSystemCallerWin32>();
#endif
	return make_smart_refctd_ptr<system::ISystem>(std::move(caller));
}

int main()
{
	const char* APP_NAME = "01.HelloWorld";
	constexpr uint32_t WIN_W = 800u;
	constexpr uint32_t WIN_H = 600u;
	constexpr uint32_t MAX_SWAPCHAIN_IMAGE_COUNT = 8u;
	constexpr uint32_t FRAMES_IN_FLIGHT = 2u;

	auto system = createSystem();
	auto logLevelMask = core::bitflag(system::ILogger::ELL_DEBUG) | system::ILogger::ELL_PERFORMANCE | system::ILogger::ELL_WARNING | system::ILogger::ELL_ERROR;
	auto logger = core::make_smart_refctd_ptr<system::CColoredStdoutLoggerWin32>(logLevelMask);
	auto winManager = core::make_smart_refctd_ptr<nbl::ui::CWindowManagerWin32>();
	auto eventCallback = core::make_smart_refctd_ptr<DemoEventCallback>();

	nbl::ui::IWindow::SCreationParams params;
	params.callback = nullptr;
	params.width = WIN_W;
	params.height = WIN_H;
	params.x = 100;
	params.y = 100;
	params.system = core::smart_refctd_ptr(system);
	params.flags = nbl::ui::IWindow::ECF_NONE;
	params.windowCaption = APP_NAME;
	params.callback = eventCallback;
	auto window = winManager->createWindow(std::move(params));

	video::IAPIConnection::E_FEATURE requiredFeatures_Instance[] = { video::IAPIConnection::EF_SURFACE };

	std::cout <<
		R"(
Choose Graphics API:
0) Vulkan
1) OpenGL
2) OpenGL ES
)" << std::endl;

	int apiType;
	std::cin >> apiType;

	core::smart_refctd_ptr<video::IAPIConnection> api = nullptr;
	core::smart_refctd_ptr<video::ISurface> surface = nullptr;
	switch (apiType)
	{
		case 0:
		{
			api = video::CVulkanConnection::create(
				core::smart_refctd_ptr(system),
				0,
				APP_NAME,
				1u, requiredFeatures_Instance,
				0u, nullptr,
				core::smart_refctd_ptr(logger),
				true);

			surface = video::CSurfaceVulkanWin32::create(
				core::smart_refctd_ptr<video::CVulkanConnection>(static_cast<video::CVulkanConnection*>(api.get())),
				core::smart_refctd_ptr<ui::IWindowWin32>(static_cast<ui::IWindowWin32*>(window.get())));
		} break;

		case 1:
		{
			api = video::COpenGLConnection::create(core::smart_refctd_ptr(system), 0, APP_NAME, video::COpenGLDebugCallback(core::smart_refctd_ptr(logger)));

			surface = video::CSurfaceGLWin32::create(
				core::smart_refctd_ptr<video::COpenGLConnection>(static_cast<video::COpenGLConnection*>(api.get())),
				core::smart_refctd_ptr<ui::IWindowWin32>(static_cast<ui::IWindowWin32*>(window.get())));
		} break;

		case 2:
		{
			api = video::COpenGLESConnection::create(core::smart_refctd_ptr(system), 0, APP_NAME, video::COpenGLDebugCallback(core::smart_refctd_ptr(logger)));

			surface = video::CSurfaceGLWin32::create(
				core::smart_refctd_ptr<video::COpenGLESConnection>(static_cast<video::COpenGLESConnection*>(api.get())),
				core::smart_refctd_ptr<ui::IWindowWin32>(static_cast<ui::IWindowWin32*>(window.get())));
		} break;
	}
	assert(api);

	auto gpus = api->getPhysicalDevices();
	assert(!gpus.empty());
	
	// Find a suitable gpu
	uint32_t graphicsFamilyIndex(~0u);
	uint32_t presentFamilyIndex(~0u);

	// Todo(achal): Probably want to put these into some struct
	uint32_t minSwapchainImageCount(~0u);
	video::ISurface::SFormat surfaceFormat;
	video::ISurface::E_PRESENT_MODE presentMode;
	asset::E_SHARING_MODE imageSharingMode;
	VkExtent2D swapchainExtent;

	// Todo(achal): Look at this:
	// https://github.com/Devsh-Graphics-Programming/Nabla/blob/6bd5061abe0a2020142efda827269ea6c07f0f2f/examples_tests/common/CommonAPI.h

	video::IPhysicalDevice* gpu = nullptr;
	for (size_t i = 0ull; i < gpus.size(); ++i)
	{
		gpu = gpus.begin()[i];

		bool isGPUSuitable = false;

		// Todo(achal): Abstract out
		// Find required queue family indices
		{
			const auto& queueFamilyProperties = gpu->getQueueFamilyProperties();

			for (uint32_t familyIndex = 0u; familyIndex < queueFamilyProperties.size(); ++familyIndex)
			{
				const auto& familyProperty = queueFamilyProperties.begin() + familyIndex;

				if ((familyProperty->queueFlags & video::IPhysicalDevice::E_QUEUE_FLAGS::EQF_GRAPHICS_BIT).value)
					graphicsFamilyIndex = familyIndex;

				if (surface->isSupportedForPhysicalDevice(gpu, familyIndex))
					presentFamilyIndex = familyIndex;

				if ((graphicsFamilyIndex != ~0u) && (presentFamilyIndex != ~0u))
				{
					isGPUSuitable = true;
					break;
				}
			}
		}

		// Since our workload is not headless compute, a swapchain is mandatory
		if (!gpu->isSwapchainSupported())
			isGPUSuitable = false;

		// Todo(achal): Abstract it out
		// Check if the surface is adequate
		{
			uint32_t surfaceFormatCount;
			surface->getAvailableFormatsForPhysicalDevice(gpu, surfaceFormatCount, nullptr);
			std::vector<video::ISurface::SFormat> surfaceFormats(surfaceFormatCount);
			surface->getAvailableFormatsForPhysicalDevice(gpu, surfaceFormatCount, surfaceFormats.data());

			video::ISurface::E_PRESENT_MODE availablePresentModes =
				surface->getAvailablePresentModesForPhysicalDevice(gpu);

			video::ISurface::SCapabilities surfaceCapabilities = {};
			if (!surface->getSurfaceCapabilitiesForPhysicalDevice(gpu, surfaceCapabilities))
				isGPUSuitable = false;

			printf("Min swapchain image count: %d\n", surfaceCapabilities.minImageCount);
			printf("Max swapchain image count: %d\n", surfaceCapabilities.maxImageCount);

			if ((surfaceFormats.empty()) || (availablePresentModes == video::ISurface::EPM_UNKNOWN))
				isGPUSuitable = false;

			// Todo(achal): Probably a more sophisticated way to choose these
			minSwapchainImageCount = core::min(surfaceCapabilities.minImageCount + 1u, MAX_SWAPCHAIN_IMAGE_COUNT);
			if ((surfaceCapabilities.maxImageCount != 0u) && (minSwapchainImageCount > surfaceCapabilities.maxImageCount))
				minSwapchainImageCount = surfaceCapabilities.maxImageCount;

			surfaceFormat = surfaceFormats[0];
			presentMode = static_cast<video::ISurface::E_PRESENT_MODE>(availablePresentModes & (1 << 0));
			swapchainExtent = surfaceCapabilities.currentExtent;
		}

		if (isGPUSuitable) // find the first suitable GPU
			break;
	}
	assert((graphicsFamilyIndex != ~0u) && (presentFamilyIndex != ~0u));

	video::ILogicalDevice::SCreationParams deviceCreationParams = {};
	if (graphicsFamilyIndex == presentFamilyIndex)
	{
		deviceCreationParams.queueParamsCount = 1u;
		imageSharingMode = asset::ESM_EXCLUSIVE;
	}
	else
	{
		deviceCreationParams.queueParamsCount = 2u;
		imageSharingMode = asset::ESM_CONCURRENT;
	}

	std::vector<uint32_t> queueFamilyIndices(deviceCreationParams.queueParamsCount);
	{
		const uint32_t temp[] = { graphicsFamilyIndex, presentFamilyIndex };
		for (uint32_t i = 0u; i < deviceCreationParams.queueParamsCount; ++i)
			queueFamilyIndices[i] = temp[i];
	}

	const float priority = video::IGPUQueue::DEFAULT_QUEUE_PRIORITY;
	std::vector<video::ILogicalDevice::SQueueCreationParams> queueCreationParams(deviceCreationParams.queueParamsCount);
	for (uint32_t i = 0u; i < deviceCreationParams.queueParamsCount; ++i)
	{
		queueCreationParams[i].familyIndex = queueFamilyIndices[i];
		queueCreationParams[i].count = 1u;
		queueCreationParams[i].flags = static_cast<video::IGPUQueue::E_CREATE_FLAGS>(0);
		queueCreationParams[i].priorities = &priority;
	}
	deviceCreationParams.queueParams = queueCreationParams.data();
	deviceCreationParams.requiredFeatureCount = 1u;
	video::ILogicalDevice::E_FEATURE requiredFeatures_Device[] = { video::ILogicalDevice::EF_SWAPCHAIN };
	deviceCreationParams.requiredFeatures = requiredFeatures_Device;

	core::smart_refctd_ptr<video::ILogicalDevice> device = gpu->createLogicalDevice(std::move(deviceCreationParams));

	video::IGPUQueue* graphicsQueue = device->getQueue(graphicsFamilyIndex, 0u);
	video::IGPUQueue* presentQueue = device->getQueue(presentFamilyIndex, 0u);

	video::ISwapchain::SCreationParams sc_params = {};
	sc_params.surface = surface;
	sc_params.minImageCount = minSwapchainImageCount;
	sc_params.surfaceFormat = surfaceFormat;
	sc_params.presentMode = presentMode;
	sc_params.width = WIN_W;
	sc_params.height = WIN_H;
	sc_params.queueFamilyIndexCount = static_cast<uint32_t>(queueFamilyIndices.size());
	sc_params.queueFamilyIndices = queueFamilyIndices.data();
	sc_params.imageSharingMode = imageSharingMode;
	sc_params.preTransform = video::ISurface::EST_IDENTITY_BIT;
	sc_params.compositeAlpha = video::ISurface::ECA_OPAQUE_BIT;
	sc_params.imageUsage = asset::IImage::EUF_COLOR_ATTACHMENT_BIT;
	sc_params.oldSwapchain = nullptr;

	core::smart_refctd_ptr<video::ISwapchain> swapchain = device->createSwapchain(std::move(sc_params));

	// Create render pass
	video::IGPURenderpass::SCreationParams::SAttachmentDescription attachmentDescription = {};
	attachmentDescription.format = surfaceFormat.format; // this should be same as the imageview used for this attachment
	attachmentDescription.samples = asset::IImage::ESCF_1_BIT;
	attachmentDescription.loadOp = video::IGPURenderpass::ELO_CLEAR; // when the first subpass begins with this attachment, clear its color and depth components
	attachmentDescription.storeOp = video::IGPURenderpass::ESO_STORE; // when the last subpass ends with this attachment, store its results
	attachmentDescription.initialLayout = asset::EIL_UNDEFINED;
	attachmentDescription.finalLayout = asset::EIL_PRESENT_SRC_KHR;

	video::IGPURenderpass::SCreationParams::SSubpassDescription subpassDescription = {};
	subpassDescription.flags = video::IGPURenderpass::ESDF_NONE;
	subpassDescription.pipelineBindPoint = asset::EPBP_GRAPHICS;
	subpassDescription.inputAttachmentCount = 0u;
	subpassDescription.inputAttachments = nullptr;

	video::IGPURenderpass::SCreationParams::SSubpassDescription::SAttachmentRef colorAttRef;
	{
		colorAttRef.attachment = 0u;
		colorAttRef.layout = asset::EIL_COLOR_ATTACHMENT_OPTIMAL;
	}
	subpassDescription.colorAttachmentCount = 1u;
	subpassDescription.colorAttachments = &colorAttRef;
	subpassDescription.resolveAttachments = nullptr;
	subpassDescription.depthStencilAttachment = nullptr;
	subpassDescription.preserveAttachmentCount = 0u;
	subpassDescription.preserveAttachments = nullptr;

	video::IGPURenderpass::SCreationParams renderPassParams;
	renderPassParams.attachmentCount = 1u;
	renderPassParams.attachments = &attachmentDescription;
	renderPassParams.dependencies = nullptr;
	renderPassParams.dependencyCount = 0u;
	renderPassParams.subpasses = &subpassDescription;
	renderPassParams.subpassCount = 1u;

	core::smart_refctd_ptr<video::IGPURenderpass> renderPass = device->createGPURenderpass(renderPassParams);

	const auto swapchainImages = swapchain->getImages();
	const uint32_t swapchainImageCount = swapchain->getImageCount();

	core::smart_refctd_ptr<video::IGPUFramebuffer> fbos[MAX_SWAPCHAIN_IMAGE_COUNT];
	for (uint32_t i = 0u; i < swapchainImageCount; ++i)
	{
		auto img = swapchainImages.begin()[i];
		core::smart_refctd_ptr<video::IGPUImageView> imageView;
		{
			video::IGPUImageView::SCreationParams viewParams;
			viewParams.format = img->getCreationParameters().format;
			viewParams.viewType = asset::IImageView<video::IGPUImage>::ET_2D;
			viewParams.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			viewParams.subresourceRange.baseMipLevel = 0u;
			viewParams.subresourceRange.levelCount = 1u;
			viewParams.subresourceRange.baseArrayLayer = 0u;
			viewParams.subresourceRange.layerCount = 1u;
			viewParams.image = std::move(img);

			imageView = device->createGPUImageView(std::move(viewParams));
			assert(imageView);
		}

		video::IGPUFramebuffer::SCreationParams fbParams;
		fbParams.width = WIN_W;
		fbParams.height = WIN_H;
		fbParams.layers = 1u;
		fbParams.renderpass = renderPass;
		fbParams.flags = static_cast<video::IGPUFramebuffer::E_CREATE_FLAGS>(0);
		fbParams.attachmentCount = renderPass->getAttachments().size();
		fbParams.attachments = &imageView;

		fbos[i] = device->createGPUFramebuffer(std::move(fbParams));
	}

	core::smart_refctd_ptr<video::IGPUSemaphore> acquireSemaphores[FRAMES_IN_FLIGHT];
	core::smart_refctd_ptr<video::IGPUSemaphore> releaseSemaphores[FRAMES_IN_FLIGHT];
	core::smart_refctd_ptr<video::IGPUFence> frameFences[FRAMES_IN_FLIGHT];

	for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; ++i)
	{
		acquireSemaphores[i] = device->createSemaphore();
		releaseSemaphores[i] = device->createSemaphore();
		frameFences[i] = device->createFence(video::IGPUFence::E_CREATE_FLAGS::ECF_SIGNALED_BIT);
	}

	core::smart_refctd_ptr<video::IGPUCommandPool> commandPool =
		device->createCommandPool(graphicsFamilyIndex, video::IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);

	core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[MAX_SWAPCHAIN_IMAGE_COUNT];
	device->createCommandBuffers(commandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY,
		swapchainImageCount, commandBuffers);

	asset::SClearValue clearColor = { 1.f, 0.f, 0.f, 1.f };

	// Record commands in commandBuffers here
	for (uint32_t i = 0u; i < swapchainImageCount; ++i)
	{
		commandBuffers[i]->begin(0u);

		video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo = {};
		beginInfo.renderpass = renderPass;
		beginInfo.framebuffer = fbos[i];
		beginInfo.renderArea.offset = { 0, 0 };
		beginInfo.renderArea.extent = swapchainExtent;
		beginInfo.clearValueCount = 1u;
		beginInfo.clearValues = &clearColor;
		commandBuffers[i]->beginRenderPass(&beginInfo, asset::ESC_INLINE);

		// Do nothing

		commandBuffers[i]->endRenderPass();
		commandBuffers[i]->end();
	}

	video::ISwapchain* rawPointerToSwapchain = swapchain.get();

	uint32_t currentFrameIndex = 0u;
	while (!eventCallback->m_gotWindowClosedMsg)
	{
		video::IGPUSemaphore* acquireSemaphore = acquireSemaphores[currentFrameIndex].get();
		video::IGPUSemaphore* releaseSemaphore = releaseSemaphores[currentFrameIndex].get();
		video::IGPUFence* fence = frameFences[currentFrameIndex].get();

		device->waitForFences(1u, &fence, true, ~0ull);

		uint32_t imageIndex;
		swapchain->acquireNextImage(~0ull, acquireSemaphores[currentFrameIndex].get(), nullptr,
			&imageIndex);

		device->resetFences(1u, &fence);

		asset::E_PIPELINE_STAGE_FLAGS waitDstStageFlags = asset::E_PIPELINE_STAGE_FLAGS::EPSF_COLOR_ATTACHMENT_OUTPUT_BIT;

		video::IGPUQueue::SSubmitInfo submitInfo = {};
		submitInfo.waitSemaphoreCount = 1u;
		submitInfo.pWaitSemaphores = &acquireSemaphore;
		submitInfo.pWaitDstStageMask = &waitDstStageFlags;
		submitInfo.signalSemaphoreCount = 1u;
		submitInfo.pSignalSemaphores = &releaseSemaphore;
		submitInfo.commandBufferCount = 1u;
		submitInfo.commandBuffers = &commandBuffers[imageIndex].get();

		graphicsQueue->submit(1u, &submitInfo, fence);

		video::IGPUQueue::SPresentInfo presentInfo;
		presentInfo.waitSemaphoreCount = 1u;
		presentInfo.waitSemaphores = &releaseSemaphore;
		presentInfo.swapchainCount = 1u;
		presentInfo.swapchains = &rawPointerToSwapchain;
		presentInfo.imgIndices = &imageIndex;

		presentQueue->present(presentInfo);

		currentFrameIndex = (currentFrameIndex + 1) % FRAMES_IN_FLIGHT;
	}

	device->waitIdle();
    
	return 0;
}
