// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "nbl/ext/DebugDraw/CDraw3DLine.h"
#include "../common/CommonAPI.h"

using namespace nbl;
using namespace core;
using namespace ui;

class Draw3DLineSampleApp : public ApplicationBase
{
	constexpr static uint32_t WIN_W = 1280;
	constexpr static uint32_t WIN_H = 720;
	constexpr static uint32_t SC_IMG_COUNT = 3u;
	constexpr static uint32_t FRAMES_IN_FLIGHT = 5u;
	static constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;
	static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

	core::smart_refctd_ptr<nbl::ui::IWindow> window;
    core::smart_refctd_ptr<nbl::system::ISystem> system;
	core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCb;
	core::smart_refctd_ptr<nbl::video::IAPIConnection> api;
	core::smart_refctd_ptr<nbl::video::ISurface> surface;
	core::smart_refctd_ptr<nbl::video::ILogicalDevice> device;
	video::IPhysicalDevice* gpu;
	std::array<video::IGPUQueue*, CommonAPI::InitOutput::MaxQueuesCount> queues;
	core::smart_refctd_ptr<nbl::video::ISwapchain> sc;
	core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
	std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, CommonAPI::InitOutput::MaxSwapChainImageCount> fbo;
	std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, CommonAPI::InitOutput::MaxQueuesCount> commandPools;
	core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
	core::smart_refctd_ptr<nbl::system::ISystem> filesystem;
	video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
	core::smart_refctd_ptr<nbl::video::IUtilities> utils;

	int32_t m_resourceIx = -1;

	core::smart_refctd_ptr<ext::DebugDraw::CDraw3DLine> m_draw3DLine;
	core::smart_refctd_ptr<video::IGPUGraphicsPipeline> m_pipeline;

	core::smart_refctd_ptr<video::IGPUCommandBuffer> m_cmdbufs[FRAMES_IN_FLIGHT];
	core::smart_refctd_ptr<video::IGPUFence> m_frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> m_imageAcquired[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> m_renderFinished[FRAMES_IN_FLIGHT] = { nullptr };

public:

	void setWindow(core::smart_refctd_ptr<nbl::ui::IWindow>&& wnd) override
	{
		window = std::move(wnd);
	}
	void setSystem(core::smart_refctd_ptr<nbl::system::ISystem>&& s) override
	{
		system = std::move(s);
	}
	nbl::ui::IWindow* getWindow() override
	{
		return window.get();
	}
	video::IAPIConnection* getAPIConnection() override
	{
		return api.get();
	}
	video::ILogicalDevice* getLogicalDevice()  override
	{
		return device.get();
	}
	video::IGPURenderpass* getRenderpass() override
	{
		return renderpass.get();
	}
	void setSurface(core::smart_refctd_ptr<video::ISurface>&& s) override
	{
		surface = std::move(s);
	}
	void setFBOs(std::vector<core::smart_refctd_ptr<video::IGPUFramebuffer>>& f) override
	{
		for (int i = 0; i < f.size(); i++)
		{
			fbo[i] = core::smart_refctd_ptr(f[i]);
		}
	}
	void setSwapchain(core::smart_refctd_ptr<video::ISwapchain>&& s) override
	{
		sc = std::move(s);
	}
	uint32_t getSwapchainImageCount() override
	{
		return SC_IMG_COUNT;
	}
	virtual nbl::asset::E_FORMAT getDepthFormat() override
	{
		return nbl::asset::EF_UNKNOWN;
	}

	APP_CONSTRUCTOR(Draw3DLineSampleApp);

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
		const asset::E_FORMAT depthFormat = asset::EF_UNKNOWN;

		CommonAPI::InitOutput initOutp;
		
		initOutp.window = window;
		initOutp.system = system;

		CommonAPI::Init(
			initOutp,
			video::EAT_VULKAN,
			"33.Draw3DLine",
			requiredInstanceFeatures,
			optionalInstanceFeatures,
			requiredDeviceFeatures,
			optionalDeviceFeatures,
			WIN_W, WIN_H, SC_IMG_COUNT,
			swapchainImageUsage,
			surfaceFormat,
			depthFormat);

		window = std::move(initOutp.window);
		windowCb = std::move(initOutp.windowCb);
		api = std::move(initOutp.apiConnection);
		surface = std::move(initOutp.surface);
		device = std::move(initOutp.logicalDevice);
		gpu = std::move(initOutp.physicalDevice);
		queues = std::move(initOutp.queues);
		sc = std::move(initOutp.swapchain);
		renderpass = std::move(initOutp.renderpass);
		fbo = std::move(initOutp.fbo);
		commandPools = std::move(initOutp.commandPools);
		assetManager = std::move(initOutp.assetManager);
		filesystem = std::move(initOutp.system);
		cpu2gpuParams = std::move(initOutp.cpu2gpuParams);
		utils = std::move(initOutp.utilities);

		m_draw3DLine = ext::DebugDraw::CDraw3DLine::create(device);

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

		matrix4SIMD proj = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(90.0f), float(WIN_W) / WIN_H, 0.01, 100);
		matrix3x4SIMD view = matrix3x4SIMD::buildCameraLookAtMatrixRH(core::vectorSIMDf(0, 0, -10), core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 1, 0));
		auto viewProj = matrix4SIMD::concatenateBFollowedByA(proj, matrix4SIMD(view));
		
		m_draw3DLine->setData(viewProj, lines);
		core::smart_refctd_ptr<video::IGPUFence> fence;
		m_draw3DLine->updateVertexBuffer(utils.get(), queues[CommonAPI::InitOutput::EQT_GRAPHICS], &fence);
		device->waitForFences(1, const_cast<video::IGPUFence**>(&fence.get()), false, MAX_TIMEOUT);

		{
			auto* rpIndependentPipeline = m_draw3DLine->getRenderpassIndependentPipeline();
			video::IGPUGraphicsPipeline::SCreationParams gp_params;
			gp_params.rasterizationSamplesHint = asset::IImage::ESCF_1_BIT;
			gp_params.renderpass = core::smart_refctd_ptr<video::IGPURenderpass>(renderpass);
			gp_params.renderpassIndependent = core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline>(rpIndependentPipeline);
			gp_params.subpassIx = 0u;

			m_pipeline = device->createGraphicsPipeline(nullptr, std::move(gp_params));
		}

		device->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_GRAPHICS].get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, m_cmdbufs);

		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
		{
			m_imageAcquired[i] = device->createSemaphore();
			m_renderFinished[i] = device->createSemaphore();
		}
	}

	void onAppTerminated_impl() override
	{
		device->waitIdle();
	}

	void workLoopBody() override
	{
		m_resourceIx++;
		if (m_resourceIx >= FRAMES_IN_FLIGHT)
			m_resourceIx = 0;

		auto& cb = m_cmdbufs[m_resourceIx];
		auto& fence = m_frameComplete[m_resourceIx];
		if (fence)
		{
			while (device->waitForFences(1u, &fence.get(), false, MAX_TIMEOUT) == video::IGPUFence::ES_TIMEOUT)
			{
			}
			device->resetFences(1u, &fence.get());
		}
		else
			fence = device->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

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

		// acquire image 
		uint32_t imgnum = 0u;
		sc->acquireNextImage(MAX_TIMEOUT, m_imageAcquired[m_resourceIx].get(), nullptr, &imgnum);

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
		info.framebuffer = fbo[imgnum];
		info.clearValueCount = 1u;
		info.clearValues = &clear;
		info.renderArea = area;
		cb->beginRenderPass(&info, asset::ESC_INLINE);
		m_draw3DLine->recordToCommandBuffer(cb.get(), m_pipeline.get());
		cb->endRenderPass();

		cb->end();

		CommonAPI::Submit(
			device.get(),
			sc.get(),
			m_cmdbufs[m_resourceIx].get(),
			queues[CommonAPI::InitOutput::EQT_GRAPHICS],
			m_imageAcquired[m_resourceIx].get(),
			m_renderFinished[m_resourceIx].get(),
			fence.get());

		CommonAPI::Present(
			device.get(),
			sc.get(),
			queues[CommonAPI::InitOutput::EQT_GRAPHICS],
			m_renderFinished[m_resourceIx].get(),
			imgnum);
	}

	bool keepRunning() override
	{
		return windowCb->isWindowOpen();
	}
};

NBL_COMMON_API_MAIN(Draw3DLineSampleApp)

extern "C" {  _declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001; }
