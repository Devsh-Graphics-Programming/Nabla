// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "nbl/ext/DebugDraw/CDraw3DLine.h"
#include "../common/CommonAPI.h"


using namespace nbl;
using namespace core;

int main()
{
	constexpr uint32_t WIN_W = 1280;
	constexpr uint32_t WIN_H = 720;
	constexpr uint32_t SC_IMG_COUNT = 3u;
	constexpr uint32_t FRAMES_IN_FLIGHT = 2u;

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
	const asset::E_FORMAT depthFormat = asset::EF_UNKNOWN;

	auto initOutp = CommonAPI::Init<WIN_W, WIN_H, SC_IMG_COUNT>(
		video::EAT_VULKAN,
		"33.Draw3DLine",
		requiredInstanceFeatures,
		optionalInstanceFeatures,
		requiredDeviceFeatures,
		optionalDeviceFeatures,
		swapchainImageUsage,
		surfaceFormat,
		depthFormat,
		true);

	auto win = std::move(initOutp.window);
	auto windowCb = std::move(initOutp.windowCb);
	auto api = std::move(initOutp.apiConnection);
	auto surface = std::move(initOutp.surface);
	auto device = std::move(initOutp.logicalDevice);
	auto gpu = std::move(initOutp.physicalDevice);
	auto queue = std::move(initOutp.queues[decltype(initOutp)::EQT_GRAPHICS]);
	auto sc = std::move(initOutp.swapchain);
	auto renderpass = std::move(initOutp.renderpass);
	auto fbo = std::move(initOutp.fbo);
	auto cmdpool = std::move(initOutp.commandPool);
	auto assetManager = std::move(initOutp.assetManager);
	auto filesystem = std::move(initOutp.system);
	auto cpu2gpuParams = std::move(initOutp.cpu2gpuParams);
	auto utils = std::move(initOutp.utilities);

	auto draw3DLine = ext::DebugDraw::CDraw3DLine::create(device);

	core::vector<std::pair<ext::DebugDraw::S3DLineVertex, ext::DebugDraw::S3DLineVertex>> lines;

	for (int i = 0; i < 100; ++i)
	{
		lines.push_back({
		{
			{ 0.f, 0.f, 0.f },     // start origin
			{ 1.f, 0.f, 0.f, 1.f } // start color
		}, {
			{ i % 2 ? float(i) : float(-i), 50.f, 10.f}, // end origin
			{ 1.f, 0.f, 0.f, 1.f }         // end color
		}
			});
	}

	constexpr uint64_t MAX_TIMEOUT = 99999999999999ull; //ns
	matrix4SIMD proj = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(90), float(WIN_W) / WIN_H, 0.01, 100);
	matrix3x4SIMD view = matrix3x4SIMD::buildCameraLookAtMatrixRH(core::vectorSIMDf(0, 0, -10), core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 1, 0));
	auto viewProj = matrix4SIMD::concatenateBFollowedByA(proj, matrix4SIMD(view));
	core::smart_refctd_ptr<video::IGPUGraphicsPipeline> pipeline;
	draw3DLine->setData(viewProj, lines);
	core::smart_refctd_ptr<video::IGPUFence> fence;
	draw3DLine->updateVertexBuffer(utils.get(), queue, &fence);
	device->waitForFences(1, const_cast<video::IGPUFence**>(&fence.get()), false, MAX_TIMEOUT);

	{
		auto* rpIndependentPipeline = draw3DLine->getRenderpassIndependentPipeline();
		video::IGPUGraphicsPipeline::SCreationParams gp_params;
		gp_params.rasterizationSamplesHint = asset::IImage::ESCF_1_BIT;
		gp_params.renderpass = core::smart_refctd_ptr<video::IGPURenderpass>(renderpass);
		gp_params.renderpassIndependent = core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline>(rpIndependentPipeline);
		gp_params.subpassIx = 0u;

		pipeline = device->createGPUGraphicsPipeline(nullptr, std::move(gp_params));
	}

	// Command buffers
	const uint32_t swapchainImageCount = sc->getImageCount();
	assert(swapchainImageCount <= SC_IMG_COUNT);
	core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbufs[SC_IMG_COUNT];
	device->createCommandBuffers(cmdpool.get(), video::IGPUCommandBuffer::EL_PRIMARY, swapchainImageCount, cmdbufs);

	for (uint32_t i = 0u; i < swapchainImageCount; ++i)
	{
		auto& cb = cmdbufs[i];
		auto& fb = fbo[i];

		cb->begin(0);

		asset::SViewport vp;
		vp.minDepth = 1.f;
		vp.maxDepth = 0.f;
		vp.x = 0u;
		vp.y = 0u;
		vp.width = WIN_W;
		vp.height = WIN_H;
		cb->setViewport(0u, 1u, &vp);

		VkRect2D scissor;
		scissor.offset = { 0, 0 };
		scissor.extent = { WIN_W, WIN_H };
		cb->setScissor(0u, 1u, &scissor);

		size_t offset = 0u;
		video::IGPUCommandBuffer::SRenderpassBeginInfo info;
		asset::SClearValue clear;
		VkRect2D area;
		area.offset = { 0, 0 };
		area.extent = { WIN_W, WIN_H };
		clear.color.float32[0] = 0.f;
		clear.color.float32[1] = 1.f;
		clear.color.float32[2] = 1.f;
		clear.color.float32[3] = 1.f;
		info.renderpass = renderpass;
		info.framebuffer = fb;
		info.clearValueCount = 1u;
		info.clearValues = &clear;
		info.renderArea = area;
		cb->beginRenderPass(&info, asset::ESC_INLINE);
		draw3DLine->recordToCommandBuffer(cb.get(), pipeline.get());
		cb->endRenderPass();

		cb->end();
	}

	// Sync primitives
	core::smart_refctd_ptr<video::IGPUFence> frameFences[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> acquireSemaphores[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> releaseSemaphores[FRAMES_IN_FLIGHT] = { nullptr };

	for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
	{
		acquireSemaphores[i] = device->createSemaphore();
		releaseSemaphores[i] = device->createSemaphore();
		frameFences[i] = device->createFence(video::IGPUFence::E_CREATE_FLAGS::ECF_SIGNALED_BIT);
	}

	uint32_t currentFrameIndex = 0u;
	while (windowCb->isWindowOpen())
	{
		video::IGPUSemaphore* acquireSemaphore_frame = acquireSemaphores[currentFrameIndex].get();
		video::IGPUSemaphore* releaseSemaphore_frame = releaseSemaphores[currentFrameIndex].get();
		video::IGPUFence* fence_frame = frameFences[currentFrameIndex].get();

		video::IGPUFence::E_STATUS retval = device->waitForFences(1u, &fence_frame, true, ~0ull);
		assert(retval == video::IGPUFence::ES_SUCCESS);

		uint32_t imgnum = 0u;
		auto acquireStatus = sc->acquireNextImage(MAX_TIMEOUT, acquireSemaphore_frame, nullptr, &imgnum);

		if (acquireStatus == video::ISwapchain::EAIR_SUCCESS)
		{
			device->resetFences(1u, &fence_frame);

			CommonAPI::Submit(
				device.get(),
				sc.get(),
				cmdbufs[imgnum].get(),
				queue,
				acquireSemaphore_frame,
				releaseSemaphore_frame,
				fence_frame);

			CommonAPI::Present(
				device.get(),
				sc.get(),
				queue,
				releaseSemaphore_frame,
				imgnum);

			currentFrameIndex = (currentFrameIndex + 1) % FRAMES_IN_FLIGHT;
		}
	}

	device->waitIdle();

	return 0;
}