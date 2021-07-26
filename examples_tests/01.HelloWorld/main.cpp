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

#include <nbl/ui/CWindowManagerWin32.h>
#include <nbl/system/ISystem.h>
#include "../common/CommonAPI.h"

using namespace nbl;

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
	constexpr uint32_t SC_IMG_COUNT = 3u;
	
	// Note(achal): This is unused, for now
	video::SDebugCallback dbgcb;
	dbgcb.callback = &debugCallback;
	dbgcb.userData = nullptr;
	
	auto system = CommonAPI::createSystem();

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
		// Todo(achal): Eventually move this to somewhere inside the engine
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
			std::vector<video::ISurface::SFormat> surfaceFormats =
				gpu->getAvailableFormatsForSurface(surface.get());

			std::vector<video::ISurface::E_PRESENT_MODE> presentModes =
				gpu->getAvailablePresentModesForSurface(surface.get());

			// Todo(achal): Probably should make a ISurface::SCapabilities
			// struct for this as a wrapper for VkSurfaceCapabilitiesKHR
			// nbl::video::ISurface::SCapabilities surfaceCapabilities = ;
			VkSurfaceCapabilitiesKHR surfaceCapabilities;
			vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu->getInternalObject(),
				vk_surface->m_surface, &surfaceCapabilities);

			if ((surfaceCapabilities.maxImageCount != 0) && (SC_IMG_COUNT > surfaceCapabilities.maxImageCount)
				|| (surfaceFormats.empty()) || (presentModes.empty()))
			{
				isGPUSuitable = false;
			}

			// Todo(achal): Probably a more sophisticated way to choose these
			surfaceFormat = surfaceFormats[0];
			presentMode = presentModes[0];
			preTransform = static_cast<nbl::video::ISurface::E_SURFACE_TRANSFORM_FLAGS>(surfaceCapabilities.currentTransform);
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

	std::vector<video::ILogicalDevice::SQueueCreationParams> queueCreationParams(deviceCreationParams.queueParamsCount);
	for (uint32_t i = 0u; i < deviceCreationParams.queueParamsCount; ++i)
	{
		queueCreationParams[i].familyIndex = (*queueFamilyIndices)[i];
		queueCreationParams[i].count = 1u;
		queueCreationParams[i].flags = static_cast<video::IGPUQueue::E_CREATE_FLAGS>(0);
		const float priority = 1.f;
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

	while (true)
	{
	}


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
