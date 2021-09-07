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


// Temporary
#include "../../src/nbl/video/CVulkanConnection.h"
#include <volk/volk.h>

#include <nbl/ui/CWindowManagerWin32.h>
#include "../common/CommonAPI.h"

using namespace nbl;

// This probably a TODO for @sadiuk
static bool windowShouldClose_Global = false;

// Don't wanna use Printer::log
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
		windowShouldClose_Global = true;
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
};

int main()
{
	constexpr uint32_t WIN_W = 800u;
	constexpr uint32_t WIN_H = 600u;
	constexpr uint32_t MAX_SWAPCHAIN_IMAGE_COUNT = 16u;
	
	auto system = CommonAPI::createSystem(); // Todo(achal): Need to get rid of this

	auto winManager = core::make_smart_refctd_ptr<nbl::ui::CWindowManagerWin32>();

	nbl::ui::IWindow::SCreationParams params;
	params.callback = nullptr;
	params.width = WIN_W;
	params.height = WIN_H;
	params.x = 0;
	params.y = 0;
	params.system = core::smart_refctd_ptr(system);
	params.flags = nbl::ui::IWindow::ECF_NONE;
	params.windowCaption = "01.HelloWorld";
	params.callback = core::make_smart_refctd_ptr<DemoEventCallback>();
	auto window = winManager->createWindow(std::move(params));

	core::smart_refctd_ptr<video::CVulkanConnection> apiConnection =
		video::CVulkanConnection::create(core::smart_refctd_ptr(system), 0, "01.HelloWorld", true);

	core::smart_refctd_ptr<video::CSurfaceVulkanWin32> surface =
		video::CSurfaceVulkanWin32::create(core::smart_refctd_ptr(apiConnection),
			core::smart_refctd_ptr<ui::IWindowWin32>(static_cast<ui::IWindowWin32*>(window.get())));

	auto gpus = apiConnection->getPhysicalDevices();
	assert(!gpus.empty());
	
	// Find a suitable gpu
	uint32_t graphicsFamilyIndex(~0u);
	uint32_t presentFamilyIndex(~0u);

	// Todo(achal): Probably want to put these into some struct
	uint32_t minSwapchainImageCount(~0u);
	nbl::video::ISurface::SFormat surfaceFormat;
	nbl::video::ISurface::E_PRESENT_MODE presentMode;
	// nbl::video::ISurface::E_SURFACE_TRANSFORM_FLAGS preTransform; // Todo(achal)
	nbl::asset::E_SHARING_MODE imageSharingMode;
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

				if (familyProperty->queueFlags & video::IPhysicalDevice::E_QUEUE_FLAGS::EQF_GRAPHICS_BIT)
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

			printf("Min swapchain image count (as returned by Vulkan): %d\n", surfaceCapabilities.minImageCount);
			printf("Max swapchain image count (as returned by Vulkan): %d\n", surfaceCapabilities.maxImageCount);

			if ((surfaceFormats.empty()) || (availablePresentModes == video::ISurface::EPM_UNKNOWN))
				isGPUSuitable = false;

			// Todo(achal): Probably a more sophisticated way to choose these
			minSwapchainImageCount = core::min(surfaceCapabilities.minImageCount + 1u, MAX_SWAPCHAIN_IMAGE_COUNT);
			if ((surfaceCapabilities.maxImageCount != 0u) && (minSwapchainImageCount > surfaceCapabilities.maxImageCount))
				minSwapchainImageCount = surfaceCapabilities.maxImageCount;

			surfaceFormat = surfaceFormats[0];
			presentMode = static_cast<video::ISurface::E_PRESENT_MODE>(availablePresentModes & (1 << 0));
			// preTransform = static_cast<nbl::video::ISurface::E_SURFACE_TRANSFORM_FLAGS>(surfaceCapabilities.currentTransform);
			swapchainExtent = surfaceCapabilities.currentExtent;
		}

		if (isGPUSuitable) // find the first suitable GPU
			break;
	}
	assert((graphicsFamilyIndex != ~0u) && (presentFamilyIndex != ~0u));

	video::ILogicalDevice::SCreationParams deviceCreationParams;
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

	core::smart_refctd_ptr<video::ILogicalDevice> device = gpu->createLogicalDevice(std::move(deviceCreationParams));

	video::IGPUQueue* graphicsQueue = device->getQueue(graphicsFamilyIndex, 0u);
	video::IGPUQueue* presentQueue = device->getQueue(presentFamilyIndex, 0u);

	nbl::video::ISwapchain::SCreationParams sc_params = {};
	sc_params.surface = surface;
	sc_params.minImageCount = minSwapchainImageCount;
	sc_params.surfaceFormat = surfaceFormat;
	sc_params.presentMode = presentMode;
	sc_params.width = WIN_W;
	sc_params.height = WIN_H;
	sc_params.queueFamilyIndexCount = static_cast<uint32_t>(queueFamilyIndices.size());
	sc_params.queueFamilyIndices = queueFamilyIndices.data();
	sc_params.imageSharingMode = imageSharingMode;
	// sc_params.preTransform = preTransform;
	sc_params.imageUsage = asset::IImage::EUF_COLOR_ATTACHMENT_BIT;
	sc_params.oldSwapchain = nullptr;

	core::smart_refctd_ptr<video::ISwapchain> swapchain = device->createSwapchain(std::move(sc_params));

	const auto swapchainImages = swapchain->getImages();
	const uint32_t swapchainImageCount = swapchain->getImageCount();

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

#if 0
	std::array<core::smart_refctd_ptr<video::IGPUFramebuffer>, SC_IMG_COUNT> fbos;
	const auto swapchainImages = swapchain->getImages();
	const uint32_t swapchainImageCount = swapchain->getImageCount();
	assert(swapchainImageCount == SC_IMG_COUNT);

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
			viewParams.image = std::move(img); // this might create problems

			imageView = device->createGPUImageView(std::move(viewParams));
			assert(imageView);
		}

		video::IGPUFramebuffer::SCreationParams fbParams;
		fbParams.width = WIN_W;
		fbParams.height = WIN_H;
		fbParams.layers = 1u;
		fbParams.renderpass = renderPass;
		fbParams.flags = static_cast<video::IGPUFramebuffer::E_CREATE_FLAGS>(0);
		fbParams.attachmentCount = 1u; // Todo(achal): must be equal to the corresponding value in render pass
		fbParams.attachments = &imageView;

		fbos[i] = device->createGPUFramebuffer(std::move(fbParams));
	}

	const uint32_t FRAMES_IN_FLIGHT = 2u;

	// acquireSemaphore will be signalled once you acquire the image to render
	// releaseSeamphore will be signalled once you rendered to the image and you're ready to release it to present it
	core::smart_refctd_ptr<video::IGPUSemaphore> acquireSemaphores[FRAMES_IN_FLIGHT];
	core::smart_refctd_ptr<video::IGPUSemaphore> releaseSemaphores[FRAMES_IN_FLIGHT];
	core::smart_refctd_ptr<video::IGPUFence> frameFences[FRAMES_IN_FLIGHT];
	for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; ++i)
	{
		acquireSemaphores[i] = device->createSemaphore();
		releaseSemaphores[i] = device->createSemaphore();
		frameFences[i] = device->createFence(video::IGPUFence::E_CREATE_FLAGS::ECF_SIGNALED_BIT);
	}

	video::IGPUQueue* graphicsQueue = device->getQueue(graphicsFamilyIndex, 0u);
	video::IGPUQueue* presentQueue = device->getQueue(presentFamilyIndex, 0u);

	// Todo(achal): Hacky stuff begin, get rid
	// Get handles to existing Vulkan stuff
	VkDevice vk_device = reinterpret_cast<video::CVKLogicalDevice*>(device.get())->getInternalObject();
	VkSwapchainKHR vk_swapchain = reinterpret_cast<video::CVKSwapchain*>(swapchain.get())->m_swapchain;
	VkRenderPass vk_renderPass = reinterpret_cast<video::CVulkanRenderpass*>(renderPass.get())->getInternalObject();

	VkFramebuffer vk_framebuffers[SC_IMG_COUNT];
	for (uint32_t i = 0u; i < SC_IMG_COUNT; ++i)
		vk_framebuffers[i] = reinterpret_cast<video::CVulkanFramebuffer*>(fbos[i].get())->m_vkfbo;

	VkImage vk_swapchainImages[SC_IMG_COUNT];
	uint32_t i = 0u;
	for (auto image : swapchainImages)
		vk_swapchainImages[i++] = reinterpret_cast<video::CVulkanImage*>(image.get())->getInternalObject();

	VkSemaphore vk_acquireSemaphores[FRAMES_IN_FLIGHT], vk_releaseSemaphores[FRAMES_IN_FLIGHT];
	VkFence vk_frameFences[FRAMES_IN_FLIGHT];
	{
		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; ++i)
		{
			vk_acquireSemaphores[i] = reinterpret_cast<video::CVulkanSemaphore*>(acquireSemaphores[i].get())->getInternalObject();
			vk_releaseSemaphores[i] = reinterpret_cast<video::CVulkanSemaphore*>(releaseSemaphores[i].get())->getInternalObject();
			vk_frameFences[i] = reinterpret_cast<video::CVulkanFence*>(frameFences[i].get())->getInternalObject();
		}
	}

	VkQueue vk_graphicsQueue = reinterpret_cast<video::CVulkanQueue*>(graphicsQueue)->getInternalObject();

	// Pure Vulkan begins here, don't even have API code for them yet!
	VkCommandPool commandPool = VK_NULL_HANDLE;
	{
		VkCommandPoolCreateInfo createInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
		createInfo.queueFamilyIndex = graphicsFamilyIndex;

		assert(vkCreateCommandPool(vk_device, &createInfo, nullptr, &commandPool) == VK_SUCCESS);
		assert(commandPool);
	}

	VkCommandBuffer commandBuffers[SC_IMG_COUNT];

	VkCommandBufferAllocateInfo allocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	allocateInfo.commandPool = commandPool;
	allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocateInfo.commandBufferCount = SC_IMG_COUNT;

	assert(vkAllocateCommandBuffers(vk_device, &allocateInfo, commandBuffers) == VK_SUCCESS);

	VkClearValue clearColor = { 0.2f, 0.2f, 0.3f, 1.f };

	// Record commands in commandBuffers here
	for (uint32_t i = 0u; i < SC_IMG_COUNT; ++i)
	{
		VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };

		assert(vkBeginCommandBuffer(commandBuffers[i], &beginInfo) == VK_SUCCESS);

		VkRenderPassBeginInfo renderPassBeginInfo = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
		renderPassBeginInfo.renderPass = vk_renderPass;
		renderPassBeginInfo.framebuffer = vk_framebuffers[i];
		renderPassBeginInfo.renderArea.offset = { 0, 0 };
		renderPassBeginInfo.renderArea.extent = swapchainExtent;
		renderPassBeginInfo.clearValueCount = 1u;
		renderPassBeginInfo.pClearValues = &clearColor;

		vkCmdBeginRenderPass(commandBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

		// Do nothing

		vkCmdEndRenderPass(commandBuffers[i]);

		assert(vkEndCommandBuffer(commandBuffers[i]) == VK_SUCCESS);
	}

	video::ISwapchain* rawPointerToSwapchain = swapchain.get();

	uint32_t currentFrameIndex = 0u;
	while (!windowShouldClose_Global)
	{
		video::IGPUSemaphore* acquireSemaphore_frame = acquireSemaphores[currentFrameIndex].get();
		video::IGPUSemaphore* releaseSemaphore_frame = releaseSemaphores[currentFrameIndex].get();
		video::IGPUFence* fence_frame = frameFences[currentFrameIndex].get();

		assert(device->waitForFences(1u, &fence_frame, true, ~0ull) == video::IGPUFence::ES_SUCCESS);

		uint32_t imageIndex;
		swapchain->acquireNextImage(~0ull, acquireSemaphores[currentFrameIndex].get(), nullptr,
			&imageIndex);

		// At this stage the final color values are output from the pipeline
		// Todo(achal): Not really sure why are waiting at this pipeline stage for
		// acquiring the image to render
		VkPipelineStageFlags pipelineStageFlags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
		submitInfo.waitSemaphoreCount = 1u;
		submitInfo.pWaitSemaphores = &vk_acquireSemaphores[currentFrameIndex];
		submitInfo.pWaitDstStageMask = &pipelineStageFlags;
		submitInfo.commandBufferCount = 1u;
		submitInfo.pCommandBuffers = &commandBuffers[imageIndex];
		submitInfo.signalSemaphoreCount = 1u;
		submitInfo.pSignalSemaphores = &vk_releaseSemaphores[currentFrameIndex];

		// Make sure you unsignal the fence before expecting vkQueueSubmit to signal it
		// once it finishes its execution
		device->resetFences(1u, &fence_frame);

		VkResult result = vkQueueSubmit(vk_graphicsQueue, 1u, &submitInfo, vk_frameFences[currentFrameIndex]);
		assert(result == VK_SUCCESS);

		// asset::E_PIPELINE_STAGE_FLAGS waitDstStageFlags = asset::E_PIPELINE_STAGE_FLAGS::EPSF_COLOR_ATTACHMENT_OUTPUT_BIT;

		// video::IGPUQueue::SSubmitInfo submitInfo = {};
		// submitInfo.waitSemaphoreCount = 1u;
		// submitInfo.pWaitSemaphores = &acquireSemaphore_frame;
		// submitInfo.pWaitDstStageMask = &waitDstStageFlags;
		// submitInfo.signalSemaphoreCount = 1u;
		// submitInfo.pSignalSemaphores = &releaseSemaphore_frame;
		// submitInfo.commandBufferCount = 1u;
		// submitInfo.commandBuffers = ;
		// assert(graphicsQueue->submit(1u, &submitInfo, fence_frame));

		video::IGPUQueue::SPresentInfo presentInfo;
		presentInfo.waitSemaphoreCount = 1u;
		presentInfo.waitSemaphores = &releaseSemaphore_frame;
		presentInfo.swapchainCount = 1u;
		presentInfo.swapchains = &rawPointerToSwapchain;
		presentInfo.imgIndices = &imageIndex;
		assert(presentQueue->present(presentInfo));

		currentFrameIndex = (currentFrameIndex + 1) % FRAMES_IN_FLIGHT;
	}

	device->waitIdle();

	vkDestroyCommandPool(vk_device, commandPool, nullptr);
#endif

#if 0
	auto gl = video::IAPIConnection::create(video::EAT_OPENGL, 0, "New API Test", dbgcb);
	auto surface = gl->createSurface(win.get());
    
	auto gpus = gl->getPhysicalDevices();
	assert(!gpus.empty());
	auto gpu = gpus.begin()[0];
    
	assert(surface->isSupported(gpu.get(), 0u));
    
	video::ILogicalDevice::SCreationParams dev_params;
	dev_params.queueParamsCount = 1u;
	video::ILogicalDevice::SQueueCreationParams q_params;
	q_params.familyIndex = 0u;
	q_params.count = 4u;
	q_params.flags = static_cast<video::IGPUQueue::E_CREATE_FLAGS>(0);
	float priority[4] = { 1.f,1.f,1.f,1.f };
	q_params.priorities = priority;
	dev_params.queueCreateInfos = &q_params;
	auto device = gpu->createLogicalDevice(dev_params);
    
	auto* queue = device->getQueue(0u, 0u);
    
	core::smart_refctd_ptr<video::ISwapchain> sc  = CommonAPI::createSwapchain(WIN_W, WIN_H, SC_IMG_COUNT, device, surface, video::ISurface::EPM_FIFO_RELAXED);
	assert(sc);
    
	core::smart_refctd_ptr<video::IGPURenderpass> renderpass = CommonAPI::createRenderpass(device);
    
	auto fbo = CommonAPI::createFBOWithSwapchainImages<SC_IMG_COUNT, WIN_W, WIN_H>(device, sc, renderpass);
    
	auto cmdpool = device->createCommandPool(0u, static_cast<video::IGPUCommandPool::E_CREATE_FLAGS>(0));
	assert(cmdpool);
    
    
	{
		video::IDriverMemoryBacked::SDriverMemoryRequirements mreq;
        
        
		core::smart_refctd_ptr<video::IGPUCommandBuffer> cb;
		device->createCommandBuffers(cmdpool.get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &cb);
		assert(cb);
        
		cb->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
        
		asset::SViewport vp;
		vp.minDepth = 1.f;
		vp.maxDepth = 0.f;
		vp.x = 0u;
		vp.y = 0u;
		vp.width = WIN_W;
		vp.height = WIN_H;
		cb->setViewport(0u, 1u, &vp);
        
		cb->end();
        
		video::IGPUQueue::SSubmitInfo info;
		auto* cb_ = cb.get();
		info.commandBufferCount = 1u;
		info.commandBuffers = &cb_;
		info.pSignalSemaphores = nullptr;
		info.signalSemaphoreCount = 0u;
		info.pWaitSemaphores = nullptr;
		info.waitSemaphoreCount = 0u;
		info.pWaitDstStageMask = nullptr;
		queue->submit(1u, &info, nullptr);
	}
    
	core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf[SC_IMG_COUNT];
	device->createCommandBuffers(cmdpool.get(), video::IGPUCommandBuffer::EL_PRIMARY, SC_IMG_COUNT, cmdbuf);
	for (uint32_t i = 0u; i < SC_IMG_COUNT; ++i)
	{
		auto& cb = cmdbuf[i];
		auto& fb = fbo[i];
        
		cb->begin(0);
        
		size_t offset = 0u;
		video::IGPUCommandBuffer::SRenderpassBeginInfo info;
		asset::SClearValue clear;
		asset::VkRect2D area;
		area.offset = { 0, 0 };
		area.extent = { WIN_W, WIN_H };
		clear.color.float32[0] = 1.f;
		clear.color.float32[1] = 0.f;
		clear.color.float32[2] = 0.f;
		clear.color.float32[3] = 1.f;
		info.renderpass = renderpass;
		info.framebuffer = fb;
		info.clearValueCount = 1u;
		info.clearValues = &clear;
		info.renderArea = area;
		cb->beginRenderPass(&info, asset::ESC_INLINE);
		cb->endRenderPass();
        
		cb->end();
	}
    
	constexpr uint32_t FRAME_COUNT = 500000u;
	constexpr uint64_t MAX_TIMEOUT = 99999999999999ull; //ns
    for (uint32_t i = 0u; i < FRAME_COUNT; ++i)
	{
		auto img_acq_sem = device->createSemaphore();
		auto render1_finished_sem = device->createSemaphore();
        
		uint32_t imgnum = 0u;
		sc->acquireNextImage(MAX_TIMEOUT, img_acq_sem.get(), nullptr, &imgnum);
        
		CommonAPI::Submit(device.get(), sc.get(), cmdbuf, queue, img_acq_sem.get(), render1_finished_sem.get(), SC_IMG_COUNT, imgnum);
        
		CommonAPI::Present(device.get(), sc.get(), queue, render1_finished_sem.get(), imgnum);
	}
    
	device->waitIdle();
#endif
    
	return 0;
}
