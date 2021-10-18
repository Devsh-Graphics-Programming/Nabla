// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "nbl/ext/DebugDraw/CDraw3DLine.h"
#include "../common/CommonAPI.h"

using namespace nbl;
using namespace core;

class MeshLoadersApp : public ApplicationBase
{
	static constexpr uint32_t WIN_W = 1280;
	static constexpr uint32_t WIN_H = 720;
	static constexpr uint32_t SC_IMG_COUNT = 3u;
	static constexpr uint32_t FRAME_COUNT = 500000u;
	static constexpr uint64_t MAX_TIMEOUT = 99999999999999ull; //ns

public:
	struct Nabla : IUserData
	{
		nbl::core::smart_refctd_ptr<nbl::ui::IWindowManager> windowManager;
		nbl::core::smart_refctd_ptr<nbl::ui::IWindow> window;
		nbl::core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCb;
		nbl::core::smart_refctd_ptr<nbl::video::IAPIConnection> gl;
		nbl::core::smart_refctd_ptr<nbl::video::ISurface> surface;
		nbl::core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
		nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> device;
		nbl::video::IPhysicalDevice* gpuPhysicalDevice;
		std::array<nbl::video::IGPUQueue*, CommonAPI::InitOutput<SC_IMG_COUNT>::EQT_COUNT> queues = { nullptr, nullptr, nullptr, nullptr };
		nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> sc;
		nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
		std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, SC_IMG_COUNT> fbos;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool;
		nbl::core::smart_refctd_ptr<nbl::system::ISystem> system;
		nbl::core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
		nbl::video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
		nbl::core::smart_refctd_ptr<nbl::system::ILogger> logger;
		nbl::core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;

		nbl::core::smart_refctd_ptr<video::IGPUFence> gpuTransferFence;
		nbl::core::smart_refctd_ptr<video::IGPUFence> gpuComputeFence;
		nbl::video::IGPUObjectFromAssetConverter cpu2gpu;

		uint32_t i = 0u;

		void setWindow(core::smart_refctd_ptr<nbl::ui::IWindow>&& wnd) override
		{
			window = std::move(wnd);
		}
	};

APP_CONSTRUCTOR(MeshLoadersApp)

	void onAppInitialized_impl(void* data) override
	{
		Nabla* engine = static_cast<Nabla*>(data);

		CommonAPI::InitOutput<SC_IMG_COUNT> initOutput;
		initOutput.window = core::smart_refctd_ptr(engine->window);
		CommonAPI::Init<WIN_W, WIN_H, SC_IMG_COUNT>(initOutput, video::EAT_OPENGL, "Draw3DLine", nbl::asset::EF_D32_SFLOAT);
		engine->window = std::move(initOutput.window);
		engine->gl = std::move(initOutput.apiConnection);
		engine->surface = std::move(initOutput.surface);
		engine->device = std::move(initOutput.logicalDevice);
		engine->queues = std::move(initOutput.queues);
		engine->sc = std::move(initOutput.swapchain);
		engine->renderpass = std::move(initOutput.renderpass);
		engine->fbos = std::move(initOutput.fbo);
		engine->cmdpool = std::move(initOutput.commandPool);
		{
			video::IDriverMemoryBacked::SDriverMemoryRequirements mreq;
			core::smart_refctd_ptr<video::IGPUCommandBuffer> cb;
			engine->device->createCommandBuffers(engine->cmdpool.get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &cb);
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
			engine->queues->submit(1u, &info, nullptr);
		}
		core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf[SC_IMG_COUNT];
		engine->device->createCommandBuffers(engine->cmdpool.get(), video::IGPUCommandBuffer::EL_PRIMARY, SC_IMG_COUNT, cmdbuf);

		auto draw3DLine = ext::DebugDraw::CDraw3DLine::create(engine->device);

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

		matrix4SIMD proj = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(90), float(WIN_W) / WIN_H, 0.01, 100);
		matrix3x4SIMD view = matrix3x4SIMD::buildCameraLookAtMatrixRH(core::vectorSIMDf(0, 0, -10), core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 1, 0));
		auto viewProj = matrix4SIMD::concatenateBFollowedByA(proj, matrix4SIMD(view));
		core::smart_refctd_ptr<video::IGPUGraphicsPipeline> pipeline;
		draw3DLine->setData(viewProj, lines);
		core::smart_refctd_ptr<video::IGPUFence> fence;
		draw3DLine->updateVertexBuffer(engine->queues, &fence);
		engine->device->waitForFences(1, const_cast<video::IGPUFence**>(&fence.get()), false, 9999999999ull);
		{
			auto* rpIndependentPipeline = draw3DLine->getRenderpassIndependentPipeline();
			video::IGPUGraphicsPipeline::SCreationParams gp_params;
			gp_params.rasterizationSamplesHint = asset::IImage::ESCF_1_BIT;
			gp_params.renderpass = core::smart_refctd_ptr<video::IGPURenderpass>(engine->renderpass);
			gp_params.renderpassIndependent = core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline>(rpIndependentPipeline);
			gp_params.subpassIx = 0u;

			pipeline = engine->device->createGPUGraphicsPipeline(nullptr, std::move(gp_params));
		}

		for (uint32_t i = 0u; i < SC_IMG_COUNT; ++i)
		{
			auto& cb = cmdbuf[i];
			auto& fb = fbos[i];

			cb->begin(0);

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
			info.renderpass = engine->renderpass;
			info.framebuffer = fb;
			info.clearValueCount = 1u;
			info.clearValues = &clear;
			info.renderArea = area;
			cb->beginRenderPass(&info, asset::ESC_INLINE);
			draw3DLine->recordToCommandBuffer(cb.get(), pipeline.get());
			cb->endRenderPass();

			cb->end();
		}
	}

	void onAppTerminated_impl(void* data) override
	{
		Nabla* engine = static_cast<Nabla*>(data);
		engine->device->waitIdle();
	}

	void workLoopBody(void* data) override
	{
		Nabla* engine = static_cast<Nabla*>(data);

		auto img_acq_sem = engine->device->createSemaphore();
		auto render1_finished_sem = engine->device->createSemaphore();

		uint32_t imgnum = 0u;
		engine->sc->acquireNextImage(MAX_TIMEOUT, img_acq_sem.get(), nullptr, &imgnum);

		CommonAPI::Submit(engine->device.get(), engine->sc.get(), engine->cmdbuf, engine->queues, img_acq_sem.get(), render1_finished_sem.get(), SC_IMG_COUNT, imgnum);

		CommonAPI::Present(engine->device.get(), engine->sc.get(), engine->queues, render1_finished_sem.get(), imgnum);

		i++;
	}

	bool keepRunning(void* params) override
	{
		Nabla* engine = static_cast<Nabla*>(params);
		return engine->i < FRAME_COUNT;
	}
};

NBL_COMMON_API_MAIN(MeshLoadersApp, MeshLoadersApp::Nabla)

int main()
{

	
	
	

	

	return 0;
}