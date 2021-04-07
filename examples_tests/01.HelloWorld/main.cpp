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
#include <nbl/system/CWindowWin32.h>
using CWindowT = nbl::system::CWindowWin32;
#elif defined(_NBL_PLATFORM_LINUX_)
#include <nbl/system/CWindowLinux.h>
using CWindowT = nbl::system::CWindowLinux;
#endif
using namespace nbl;

static void debugCallback(video::E_DEBUG_MESSAGE_SEVERITY severity, video::E_DEBUG_MESSAGE_TYPE type, const char* msg, void* userData)
{
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

int main()
{
	constexpr uint32_t WIN_W = 800u;
	constexpr uint32_t WIN_H = 600u;
	constexpr uint32_t SC_IMG_COUNT = 3u;

	auto win = CWindowT::create(WIN_W, WIN_H, system::IWindow::ECF_NONE);

	video::SDebugCallback dbgcb;
	dbgcb.callback = &debugCallback;
	dbgcb.userData = nullptr;
	auto gl = video::IAPIConnection::create(video::EAT_OPENGL, 0, "New API Test", &dbgcb);
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

	core::smart_refctd_ptr<video::ISwapchain> sc;
	{
		video::ISwapchain::SCreationParams sc_params;
		sc_params.width = WIN_W;
		sc_params.height = WIN_H;
		sc_params.arrayLayers = 1u;
		sc_params.minImageCount = SC_IMG_COUNT;
		sc_params.presentMode = video::ISurface::EPM_FIFO_RELAXED;
		sc_params.surface = surface;
		sc_params.surfaceFormat.format = asset::EF_R8G8B8A8_SRGB;
		sc_params.surfaceFormat.colorSpace.eotf = asset::EOTF_sRGB;
		sc_params.surfaceFormat.colorSpace.primary = asset::ECP_SRGB;

		sc = device->createSwapchain(std::move(sc_params));
		assert(sc);
	}

	core::smart_refctd_ptr<video::IGPURenderpass> renderpass;
	{
		video::IGPURenderpass::SCreationParams::SAttachmentDescription a;
		a.initialLayout = asset::EIL_UNDEFINED;
		a.finalLayout = asset::EIL_UNDEFINED;
		a.format = asset::EF_R8G8B8A8_SRGB;
		a.samples = asset::IImage::ESCF_1_BIT;
		a.loadOp = video::IGPURenderpass::ELO_CLEAR;
		a.storeOp = video::IGPURenderpass::ESO_STORE;

		video::IGPURenderpass::SCreationParams::SSubpassDescription::SAttachmentRef colorAttRef;
		colorAttRef.attachment = 0u;
		colorAttRef.layout = asset::EIL_UNDEFINED;
		video::IGPURenderpass::SCreationParams::SSubpassDescription sp;
		sp.colorAttachmentCount = 1u;
		sp.colorAttachments = &colorAttRef;
		sp.depthStencilAttachment = nullptr;
		sp.flags = video::IGPURenderpass::ESDF_NONE;
		sp.inputAttachmentCount = 0u;
		sp.inputAttachments = nullptr;
		sp.preserveAttachmentCount = 0u;
		sp.preserveAttachments = nullptr;
		sp.resolveAttachments = nullptr;

		video::IGPURenderpass::SCreationParams rp_params;
		rp_params.attachmentCount = 1u;
		rp_params.attachments = &a;
		rp_params.dependencies = nullptr;
		rp_params.dependencyCount = 0u;
		rp_params.subpasses = &sp;
		rp_params.subpassCount = 1u;

		renderpass = device->createGPURenderpass(rp_params);
	}

	auto sc_images = sc->getImages();
	assert(sc_images.size() == SC_IMG_COUNT);

	core::smart_refctd_ptr<video::IGPUFramebuffer> fbo[SC_IMG_COUNT];
	for (uint32_t i = 0u; i < sc_images.size(); ++i)
	{
		auto img = sc_images.begin()[i];
		core::smart_refctd_ptr<video::IGPUImageView> view;
		{
			video::IGPUImageView::SCreationParams view_params;
			view_params.format = img->getCreationParameters().format;
			view_params.viewType = asset::IImageView<video::IGPUImage>::ET_2D;
			view_params.subresourceRange.baseMipLevel = 0u;
			view_params.subresourceRange.levelCount = 1u;
			view_params.subresourceRange.baseArrayLayer = 0u;
			view_params.subresourceRange.layerCount = 1u;
			view_params.image = std::move(img);

			view = device->createGPUImageView(std::move(view_params));
			assert(view);
		}

		video::IGPUFramebuffer::SCreationParams fb_params;
		fb_params.width = WIN_W;
		fb_params.height = WIN_H;
		fb_params.layers = 1u;
		fb_params.renderpass = renderpass;
		fb_params.flags = static_cast<video::IGPUFramebuffer::E_CREATE_FLAGS>(0);
		fb_params.attachmentCount = 1u;
		fb_params.attachments = &view;

		fbo[i] = device->createGPUFramebuffer(std::move(fb_params));
		assert(fbo[i]);
	}

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
		auto render_finished_sem = device->createSemaphore();

		uint32_t imgnum = 0u;
		sc->acquireNextImage(MAX_TIMEOUT, img_acq_sem.get(), nullptr, &imgnum);

		video::IGPUQueue::SSubmitInfo submit;
		{
			auto* cb = cmdbuf[imgnum].get();
			submit.commandBufferCount = 1u;
			submit.commandBuffers = &cb;
			video::IGPUSemaphore* signalsem = render_finished_sem.get();
			submit.signalSemaphoreCount = 1u;
			submit.pSignalSemaphores = &signalsem;
			video::IGPUSemaphore* waitsem = img_acq_sem.get();
			asset::E_PIPELINE_STAGE_FLAGS dstWait = asset::EPSF_COLOR_ATTACHMENT_OUTPUT_BIT;
			submit.waitSemaphoreCount = 1u;
			submit.pWaitSemaphores = &waitsem;
			submit.pWaitDstStageMask = &dstWait;

			queue->submit(1u, &submit, nullptr);
		}

		video::IGPUQueue::SPresentInfo present;
		{
			present.swapchainCount = 1u;
			present.imgIndices = &imgnum;
			video::ISwapchain* swapchain = sc.get();
			present.swapchains = &swapchain;
			video::IGPUSemaphore* waitsem = render_finished_sem.get();
			present.waitSemaphoreCount = 1u;
			present.waitSemaphores = &waitsem;

			queue->present(present);
		}
		
	}

	device->waitIdle();

	return 0;
}
