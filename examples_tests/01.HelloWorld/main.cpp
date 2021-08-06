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
#include <volk/volk.h>
#include "../../src/nbl/video/CVulkanPhysicalDevice.h"
#include "../../include/nbl/video/surface/ISurfaceVK.h"
#include "../../src/nbl/video/CVulkanImage.h"
#include "../../src/nbl/video/CVulkanSemaphore.h"

#include <nbl/ui/CWindowManagerWin32.h>
#include "../common/CommonAPI.h"

using namespace nbl;

// This probably a TODO for @sadiuk
static bool windowShouldClose_Global = false;


inline void debugCallback(nbl::video::E_DEBUG_MESSAGE_SEVERITY severity, nbl::video::E_DEBUG_MESSAGE_TYPE type, const char* msg, void* userData)
{
	using namespace nbl;
	const char* sev = nullptr;
	switch (severity)
	{
        case video::EDMS_VERBOSE:
		sev = "verbose"; break;
        case video::EDMS_INFO:
		sev = "info"; break;
        case video::EDMS_WARNING:
		sev = "warning"; break;
        case video::EDMS_ERROR:
		sev = "error"; break;
	}
	std::cout << "OpenGL " << sev << ": " << msg << std::endl;
}

// Don't wanna use Printer::log
#define LOG(...) printf(__VA_ARGS__); printf("\n");
class DemoEventCallback : public nbl::ui::IWindow::IEventCallback
{
	void onWindowShown_impl() override
	{
		LOG("Window Shown");
	}
	void onWindowHidden_impl() override
	{
		LOG("Window hidden");
	}
	void onWindowMoved_impl(int32_t x, int32_t y) override
	{
		LOG("Window window moved to { %d, %d }", x, y);
	}
	void onWindowResized_impl(uint32_t w, uint32_t h) override
	{
		LOG("Window resized to { %u, %u }", w, h);
	}
	void onWindowMinimized_impl() override
	{
		LOG("Window minimized");
	}
	void onWindowMaximized_impl() override
	{
		LOG("Window maximized");
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
	constexpr uint32_t SC_IMG_COUNT = 3u; // problematic, shouldn't fix the number of swapchain images at compile time, since Vulkan is under no obligation to return you the exact number of images you requested
	
	// Note(achal): This is unused, for now
	video::SDebugCallback dbgcb;
	dbgcb.callback = &debugCallback;
	dbgcb.userData = nullptr;
	
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
	params.windowCaption = "Test Window";
	params.callback = core::make_smart_refctd_ptr<DemoEventCallback>();
	auto window = winManager->createWindow(std::move(params));

	core::smart_refctd_ptr<video::IAPIConnection> vk = video::IAPIConnection::create(std::move(system), video::EAT_VULKAN, 0, "New API Test", &dbgcb);
	core::smart_refctd_ptr<video::ISurface> surface = vk->createSurface(window.get());

	auto gpus = vk->getPhysicalDevices();
	assert(!gpus.empty());
	
	// Find a suitable gpu
	// 
	// Todo(achal): This process gets quite involved in Vulkan, do we want to delegate some of it to the engine.
	// For example, the user just specifies if the application is a headless compute one and we could check the
	// availability of the required extensions on the backend and report which physical device is suitable or
	// some thing like that

	// Todo(achal): Look at this:
	// https://github.com/Devsh-Graphics-Programming/Nabla/blob/6bd5061abe0a2020142efda827269ea6c07f0f2f/examples_tests/common/CommonAPI.h
	core::smart_refctd_ptr<video::CVulkanPhysicalDevice> gpu = nullptr;

	uint32_t graphicsFamilyIndex(~0u);
	uint32_t presentFamilyIndex(~0u);

	// Todo(achal): Probably want to put these into some struct
	nbl::video::ISurface::SFormat surfaceFormat;
	nbl::video::ISurface::E_PRESENT_MODE presentMode;
	nbl::video::ISurface::E_SURFACE_TRANSFORM_FLAGS preTransform;
	nbl::asset::E_SHARING_MODE imageSharingMode;
	VkExtent2D swapchainExtent;
	using QueueFamilyIndicesArrayType = core::smart_refctd_dynamic_array<uint32_t>;
	QueueFamilyIndicesArrayType queueFamilyIndices;

	for (size_t i = 0ull; i < gpus.size(); ++i)
	{
		// Todo(achal): Hacks, get rid
		gpu = core::smart_refctd_ptr_static_cast<video::CVulkanPhysicalDevice>(*(gpus.begin() + i));
		auto vk_surface = core::smart_refctd_ptr_static_cast<video::ISurfaceVK>(surface);

		bool isGPUSuitable = false;

		// Check if the physical device has queue families which support both graphics and presenting, not necessarily overlapping
		{
			const auto& queueFamilyProperties = gpu->getQueueFamilyProperties();

			for (uint32_t familyIndex = 0u; familyIndex < queueFamilyProperties.size(); ++familyIndex)
			{
				const auto& familyProperty = queueFamilyProperties.begin() + familyIndex;
				if (familyProperty->queueFlags & video::IPhysicalDevice::E_QUEUE_FLAGS::EQF_GRAPHICS_BIT)
					graphicsFamilyIndex = familyIndex;

				if (surface->isSupported(gpu.get(), familyIndex))
					presentFamilyIndex = familyIndex;

				if ((graphicsFamilyIndex != ~0u) && (presentFamilyIndex != ~0u))
				{
					isGPUSuitable = true;
					break;
				}
			}
		}

		// Check if this physical device supports the swapchain extension
		// Todo(achal): Eventually move this to CommonAPI.h
		{
			// Todo(achal): Get this from the user
			const uint32_t requiredDeviceExtensionCount = 1u;
			const char* requiredDeviceExtensionNames[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

			uint32_t availableExtensionCount;
			vkEnumerateDeviceExtensionProperties(gpu->getInternalObject(), NULL, &availableExtensionCount, NULL);
			std::vector<VkExtensionProperties> availableExtensions(availableExtensionCount);
			vkEnumerateDeviceExtensionProperties(gpu->getInternalObject(), NULL, &availableExtensionCount, availableExtensions.data());

			bool requiredDeviceExtensionsAvailable = false;
			for (uint32_t i = 0u; i < availableExtensionCount; ++i)
			{
				if (strcmp(availableExtensions[i].extensionName, requiredDeviceExtensionNames[0]) == 0)
				{
					requiredDeviceExtensionsAvailable = true;
					break;
				}
			}

			if (!requiredDeviceExtensionsAvailable)
				isGPUSuitable = false;
		}

		// Check if the surface is adequate
		{
			uint32_t surfaceFormatCount;
			gpu->getAvailableFormatsForSurface(surface.get(), surfaceFormatCount, nullptr);
			std::vector<video::ISurface::SFormat> surfaceFormats(surfaceFormatCount);
			gpu->getAvailableFormatsForSurface(surface.get(), surfaceFormatCount, surfaceFormats.data());

			video::ISurface::E_PRESENT_MODE presentModes =
				gpu->getAvailablePresentModesForSurface(surface.get());

			// Todo(achal): Probably should make a ISurface::SCapabilities
			// struct for this as a wrapper for VkSurfaceCapabilitiesKHR
			// nbl::video::ISurface::SCapabilities surfaceCapabilities = ;
			VkSurfaceCapabilitiesKHR surfaceCapabilities;
			vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu->getInternalObject(), 
				vk_surface->m_surface, &surfaceCapabilities);

			printf("Min swapchain image count: %d\n", surfaceCapabilities.minImageCount);
			printf("Max swapchain image count: %d\n", surfaceCapabilities.maxImageCount);

			if ((surfaceCapabilities.maxImageCount != 0) && (SC_IMG_COUNT > surfaceCapabilities.maxImageCount)
				|| (surfaceFormats.empty()) || (presentModes == static_cast<video::ISurface::E_PRESENT_MODE>(0)))
			{
				isGPUSuitable = false;
			}

			// Todo(achal): Probably a more sophisticated way to choose these
			surfaceFormat = surfaceFormats[0];
			presentMode = static_cast<video::ISurface::E_PRESENT_MODE>(presentModes & (1 << 0));
			preTransform = static_cast<nbl::video::ISurface::E_SURFACE_TRANSFORM_FLAGS>(surfaceCapabilities.currentTransform);
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

	queueFamilyIndices = core::make_refctd_dynamic_array<QueueFamilyIndicesArrayType>(deviceCreationParams.queueParamsCount);
	{
		const uint32_t temp[] = { graphicsFamilyIndex, presentFamilyIndex };
		for (uint32_t i = 0u; i < deviceCreationParams.queueParamsCount; ++i)
			(*queueFamilyIndices)[i] = temp[i];
	}

	const float priority = 1.f;
	std::vector<video::ILogicalDevice::SQueueCreationParams> queueCreationParams(deviceCreationParams.queueParamsCount);
	for (uint32_t i = 0u; i < deviceCreationParams.queueParamsCount; ++i)
	{
		queueCreationParams[i].familyIndex = (*queueFamilyIndices)[i];
		queueCreationParams[i].count = 1u;
		queueCreationParams[i].flags = static_cast<video::IGPUQueue::E_CREATE_FLAGS>(0);
		queueCreationParams[i].priorities = &priority;
	}
	deviceCreationParams.queueCreateInfos = queueCreationParams.data();
	core::smart_refctd_ptr<video::ILogicalDevice> device = gpu->createLogicalDevice(std::move(deviceCreationParams));

	nbl::video::ISwapchain::SCreationParams sc_params = {};
	sc_params.surface = surface;
	sc_params.minImageCount = SC_IMG_COUNT;
	sc_params.surfaceFormat = surfaceFormat;
	sc_params.presentMode = presentMode;
	sc_params.width = WIN_W;
	sc_params.height = WIN_H;
	sc_params.queueFamilyIndices = queueFamilyIndices;
	sc_params.imageSharingMode = imageSharingMode;
	sc_params.preTransform = preTransform;

	core::smart_refctd_ptr<video::ISwapchain> swapchain = device->createSwapchain(std::move(sc_params));

	// Create render pass
	video::IGPURenderpass::SCreationParams::SAttachmentDescription attachmentDescription;
	attachmentDescription.format = surfaceFormat.format; // this should be same as the imageview used for this attachment
	attachmentDescription.samples = asset::IImage::ESCF_1_BIT;
	attachmentDescription.loadOp = video::IGPURenderpass::ELO_CLEAR; // when the first subpass begins with this attachment, clear its color and depth components
	attachmentDescription.storeOp = video::IGPURenderpass::ESO_STORE; // when the last subpass ends with this attachment, store its results
	attachmentDescription.initialLayout = asset::EIL_UNDEFINED;
	attachmentDescription.finalLayout = asset::EIL_PRESENT_SRC_KHR;

	video::IGPURenderpass::SCreationParams::SSubpassDescription subpassDescription;
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
	renderPassParams.dependencies = nullptr; // Todo(achal): Probably need this very soon
	renderPassParams.dependencyCount = 0u;
	renderPassParams.subpasses = &subpassDescription;
	renderPassParams.subpassCount = 1u;

	core::smart_refctd_ptr<video::IGPURenderpass> renderPass = device->createGPURenderpass(renderPassParams);

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

	core::smart_refctd_ptr<video::IGPUSemaphore> acquireSemaphores[FRAMES_IN_FLIGHT];
	core::smart_refctd_ptr<video::IGPUSemaphore> releaseSemaphores[FRAMES_IN_FLIGHT];
	for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; ++i)
	{
		acquireSemaphores[i] = device->createSemaphore();
		releaseSemaphores[i] = device->createSemaphore();
	}

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

	// acquireSemaphore will be signalled once you acquire the image to render
	// releaseSeamphore will be signalled once you rendered to the image and you're ready to release it to present it
	VkSemaphore vk_acquireSemaphores[FRAMES_IN_FLIGHT], vk_releaseSemaphores[FRAMES_IN_FLIGHT];
	{

		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; ++i)
		{
			vk_acquireSemaphores[i] = reinterpret_cast<video::CVulkanSemaphore*>(acquireSemaphores[i].get())->getInternalObject();
			vk_releaseSemaphores[i] = reinterpret_cast<video::CVulkanSemaphore*>(releaseSemaphores[i].get())->getInternalObject();
		}
	}

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

	VkFence frameFences[FRAMES_IN_FLIGHT];
	{
		VkFenceCreateInfo createInfo = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
		createInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; ++i)
			assert(vkCreateFence(vk_device, &createInfo, nullptr, &frameFences[i]) == VK_SUCCESS);
	}

	VkQueue graphicsQueue, presentQueue;
	vkGetDeviceQueue(vk_device, graphicsFamilyIndex, 0, &graphicsQueue);
	vkGetDeviceQueue(vk_device, presentFamilyIndex, 0, &presentQueue);

	uint32_t currentFrameIndex = 0u;
	while (!windowShouldClose_Global)
	{
		// The purpose of this call is to ensure that there is free space in the "batch"
		// to incorporate this new frame
		assert(vkWaitForFences(vk_device, 1, &frameFences[currentFrameIndex], VK_TRUE, ~0ull) == VK_SUCCESS);

		uint32_t imageIndex;
		assert(vkAcquireNextImageKHR(vk_device, vk_swapchain, UINT64_MAX,
			vk_acquireSemaphores[currentFrameIndex], VK_NULL_HANDLE, &imageIndex) == VK_SUCCESS);

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
		assert(vkResetFences(vk_device, 1, &frameFences[currentFrameIndex]) == VK_SUCCESS);

		VkResult result = vkQueueSubmit(graphicsQueue, 1u, &submitInfo, frameFences[currentFrameIndex]);
		assert(result == VK_SUCCESS);

		VkPresentInfoKHR presentInfo = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
		presentInfo.waitSemaphoreCount = 1u;
		presentInfo.pWaitSemaphores = &vk_releaseSemaphores[currentFrameIndex];
		presentInfo.swapchainCount = 1u;
		presentInfo.pSwapchains = &vk_swapchain;
		presentInfo.pImageIndices = &imageIndex;
		vkQueuePresentKHR(presentQueue, &presentInfo);

		currentFrameIndex = (currentFrameIndex + 1) % FRAMES_IN_FLIGHT;
	}

	vkDeviceWaitIdle(vk_device);

	for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; ++i)
	{
		vkDestroyFence(vk_device, frameFences[i], nullptr);
	}
	vkDestroyCommandPool(vk_device, commandPool, nullptr);

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
    
	// return 0;
	exit(0);
}
