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

#include "../common/CommonAPI.h"

using namespace nbl;

// TODO: replace with CFileLogger logs to .txt files, derived from ILogger
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


int main()
{
	constexpr uint32_t WIN_W = 800u;
	constexpr uint32_t WIN_H = 600u;
	constexpr uint32_t SC_IMG_COUNT = 3u;

	auto win = CWindowT::create(WIN_W, WIN_H, ui::IWindow::ECF_NONE);

	// TODO: API choice once its figured out in example 49, then copy that code here and remove it there
	video::SDebugCallback dbgcb;
	dbgcb.callback = &debugCallback;
	dbgcb.userData = nullptr;
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

	return 0;
}
