// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>
#include "nbl/video/utilities/CSimpleResizeSurface.h"

#include "../common/SimpleWindowedApplication.hpp"
#include "../common/InputSystem.hpp"

#include "nbl/ext/ImGui/ImGui.h"
#include "nbl/ui/ICursorControl.h"

// Test Engine
#include "imgui_test_suite_imconfig.h"
#include "imgui_test_suite.h"
#include "imgui_te_engine.h"
#include "imgui_te_ui.h"
#include "imgui_te_utils.h"

// Argparse
#include <argparse/argparse.hpp>

using namespace nbl;
using namespace core;
using namespace hlsl;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;

class CEventCallback : public ISimpleManagedSurface::ICallback
{
public:
	CEventCallback(nbl::core::smart_refctd_ptr<InputSystem>&& m_inputSystem, nbl::system::logger_opt_smart_ptr&& logger) : m_inputSystem(std::move(m_inputSystem)), m_logger(std::move(logger)) {}
	CEventCallback() {}

	void setLogger(nbl::system::logger_opt_smart_ptr& logger)
	{
		m_logger = logger;
	}
	void setInputSystem(nbl::core::smart_refctd_ptr<InputSystem>&& m_inputSystem)
	{
		m_inputSystem = std::move(m_inputSystem);
	}
private:

	void onMouseConnected_impl(nbl::core::smart_refctd_ptr<nbl::ui::IMouseEventChannel>&& mch) override
	{
		m_logger.log("A mouse %p has been connected", nbl::system::ILogger::ELL_INFO, mch.get());
		m_inputSystem.get()->add(m_inputSystem.get()->m_mouse, std::move(mch));
	}
	void onMouseDisconnected_impl(nbl::ui::IMouseEventChannel* mch) override
	{
		m_logger.log("A mouse %p has been disconnected", nbl::system::ILogger::ELL_INFO, mch);
		m_inputSystem.get()->remove(m_inputSystem.get()->m_mouse, mch);
	}
	void onKeyboardConnected_impl(nbl::core::smart_refctd_ptr<nbl::ui::IKeyboardEventChannel>&& kbch) override
	{
		m_logger.log("A keyboard %p has been connected", nbl::system::ILogger::ELL_INFO, kbch.get());
		m_inputSystem.get()->add(m_inputSystem.get()->m_keyboard, std::move(kbch));
	}
	void onKeyboardDisconnected_impl(nbl::ui::IKeyboardEventChannel* kbch) override
	{
		m_logger.log("A keyboard %p has been disconnected", nbl::system::ILogger::ELL_INFO, kbch);
		m_inputSystem.get()->remove(m_inputSystem.get()->m_keyboard, kbch);
	}

private:
	nbl::core::smart_refctd_ptr<InputSystem> m_inputSystem = nullptr;
	nbl::system::logger_opt_smart_ptr m_logger = nullptr;
};

class NITETool final : public examples::SimpleWindowedApplication
{
	using device_base_t = examples::SimpleWindowedApplication;
	using clock_t = std::chrono::steady_clock;

	_NBL_STATIC_INLINE_CONSTEXPR uint32_t WIN_W = 1280, WIN_H = 720, SC_IMG_COUNT = 3u, FRAMES_IN_FLIGHT = 5u;
	static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

	constexpr static inline clock_t::duration DisplayImageDuration = std::chrono::milliseconds(900);

public:
	inline NITETool(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
		: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	inline core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
	{
		if (!m_surface)
		{
			{
				auto windowCallback = core::make_smart_refctd_ptr<CEventCallback>(smart_refctd_ptr(m_inputSystem), smart_refctd_ptr(m_logger));
				IWindow::SCreationParams params = {};
				params.callback = core::make_smart_refctd_ptr<nbl::video::ISimpleManagedSurface::ICallback>();
				params.width = WIN_W;
				params.height = WIN_H;
				params.x = 32;
				params.y = 32;
				params.flags = ui::IWindow::ECF_HIDDEN | IWindow::ECF_BORDERLESS | IWindow::ECF_RESIZABLE;
				params.windowCaption = "NITETool";
				params.callback = windowCallback;
				const_cast<std::remove_const_t<decltype(m_window)>&>(m_window) = m_winMgr->createWindow(std::move(params));
			}

			auto surface = CSurfaceVulkanWin32::create(smart_refctd_ptr(m_api), smart_refctd_ptr_static_cast<IWindowWin32>(m_window));
			const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = nbl::video::CSimpleResizeSurface<nbl::video::CDefaultSwapchainFramebuffers>::create(std::move(surface));
		}

		if (m_surface)
			return { {m_surface->getSurface()/*,EQF_NONE*/} };

		return {};
	}

	inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{

		_NBL_STATIC_INLINE_CONSTEXPR std::string_view NBL_QUEUE_ARG = "--queued";	// flag
		_NBL_STATIC_INLINE_CONSTEXPR std::string_view NBL_MODE_ARG = "--mode";		// "cmd" || "gui" value
		_NBL_STATIC_INLINE_CONSTEXPR std::string_view NBL_GROUP_ARG = "--group";	// "test" || "perf" value 

		argparse::ArgumentParser program("[NITE]: Performs Suite Test for Nabla IMGUI backend");

		program.add_argument(NBL_QUEUE_ARG.data())
			.flag()
			.help("use this argument to queue execution of tests depending on --group argument, otherwise (default) you can browse GUI freely");

		program.add_argument(NBL_MODE_ARG.data())
			.default_value("gui")
			.help("use \"cmd\" for running from command line and \"gui\" for GUI (default)");

		program.add_argument(NBL_GROUP_ARG.data())
			.default_value("test")
			.help("use \"test\" (default) for running basic tests and \"perf\" for performance tests");

		try
		{
			program.parse_args({ argv.data(), argv.data() + argv.size() });
		}
		catch (const std::exception& err)
		{
			std::cerr << err.what() << std::endl << program;
			return 1;
		}

		const auto pQueueArg = program.get<bool>(NBL_QUEUE_ARG.data());
		const auto pModeArg = program.get<std::string>(NBL_MODE_ARG.data());
		const auto pGroupArg = program.get<std::string>(NBL_GROUP_ARG.data());

		m_inputSystem = make_smart_refctd_ptr<InputSystem>(logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));

		if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;

		m_semaphore = m_device->createSemaphore(m_submitIx);
		if (!m_semaphore)
			return logFail("Failed to Create a Semaphore!");

		ISwapchain::SCreationParams swapchainParams = { .surface = m_surface->getSurface() };
		if (!swapchainParams.deduceFormat(m_physicalDevice))
			return logFail("Could not choose a Surface Format for the Swapchain!");

		const static IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] =
		{
			{
				.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
				.dstSubpass = 0,
				.memoryBarrier =
				{
					.srcStageMask = asset::PIPELINE_STAGE_FLAGS::COPY_BIT,
					.srcAccessMask = asset::ACCESS_FLAGS::TRANSFER_WRITE_BIT,
					.dstStageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
					.dstAccessMask = asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
				}
			},
			{
				.srcSubpass = 0,
				.dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
				.memoryBarrier =
				{
					.srcStageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
					.srcAccessMask = asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
				}
			},
			IGPURenderpass::SCreationParams::DependenciesEnd
		};

		auto scResources = std::make_unique<CDefaultSwapchainFramebuffers>(m_device.get(), swapchainParams.surfaceFormat.format, dependencies);
		auto* renderpass = scResources->getRenderpass();

		if (!renderpass)
			return logFail("Failed to create Renderpass!");

		auto gQueue = getGraphicsQueue();
		if (!m_surface || !m_surface->init(gQueue, std::move(scResources), swapchainParams.sharedParams))
			return logFail("Could not create Window & Surface or initialize the Surface!");

		m_maxFramesInFlight = m_surface->getMaxFramesInFlight();
		if (FRAMES_IN_FLIGHT < m_maxFramesInFlight)
		{
			m_logger->log("Lowering frames in flight!", ILogger::ELL_WARNING);
			m_maxFramesInFlight = FRAMES_IN_FLIGHT;
		}

		m_cmdPool = m_device->createCommandPool(gQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);

		for (auto i = 0u; i < m_maxFramesInFlight; i++)
		{
			if (!m_cmdPool)
				return logFail("Couldn't create Command Pool!");
			if (!m_cmdPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data() + i, 1 }))
				return logFail("Couldn't create Command Buffer!");
		}

		ui = core::make_smart_refctd_ptr<nbl::ext::imgui::UI>(smart_refctd_ptr(m_device), (int)m_maxFramesInFlight, renderpass, nullptr, smart_refctd_ptr(m_window));
		auto* ctx = reinterpret_cast<ImGuiContext*>(ui->getContext());
		ImGui::SetCurrentContext(ctx);
	
		// Initialize Test Engine
		engine = ImGuiTestEngine_CreateContext();
		ImGuiTestEngineIO& test_io = ImGuiTestEngine_GetIO(engine);
		test_io.ConfigVerboseLevel = ImGuiTestVerboseLevel_Info;
		test_io.ConfigVerboseLevelOnError = ImGuiTestVerboseLevel_Debug;
		test_io.ConfigSavedSettings = false;

		// Register tests
		RegisterTests_All(engine);

		// Start engine
		ImGuiTestEngine_Start(engine, ctx);
		ImGuiTestEngine_InstallDefaultCrashHandler();

		if (pQueueArg)
		{
			ImGuiTestGroup group = ImGuiTestGroup_Unknown;
			{
				if (pGroupArg == "test")
					group = ImGuiTestGroup_Tests;
				else if (pGroupArg == "perf")
					group = ImGuiTestGroup_Perfs;
			}
			ImGuiTestRunFlags flags = ImGuiTestRunFlags_None;
			{
				if (pModeArg == "cmd")
					flags = ImGuiTestRunFlags_RunFromCommandLine;
				else if (pModeArg == "gui")
					flags = ImGuiTestRunFlags_RunFromGui;
			}
			ImGuiTestEngine_QueueTests(engine, group, nullptr, flags);
		}

		ui->registerListener([this]() -> void
			{
				ImGuiTestEngine_ShowTestEngineWindows(engine, nullptr);
			}
		);

		m_winMgr->setWindowSize(m_window.get(), WIN_W, WIN_H);
		m_surface->recreateSwapchain();
		m_winMgr->show(m_window.get());

		return true;
	}

	inline void workLoopBody() override
	{
		static std::chrono::microseconds previousEventTimestamp{};
		// TODO: Use real deltaTime instead
		float deltaTimeInSec = 0.1f;

		struct
		{
			std::vector<SMouseEvent> mouse{};
		} capturedEvents;

		m_inputSystem->getDefaultMouse(&mouse);
		m_inputSystem->getDefaultKeyboard(&keyboard);

		mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void
			{
				for (auto event : events)
				{
					if (event.timeStamp < previousEventTimestamp)
						continue;

					previousEventTimestamp = event.timeStamp;
					capturedEvents.mouse.push_back(event);
				}
			}, m_logger.get());

		keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void
			{
				// TOOD
			}, m_logger.get());

		const auto mousePosition = m_window->getCursorControl()->getPosition();
		ui->update(deltaTimeInSec, static_cast<float>(mousePosition.x), static_cast<float>(mousePosition.y), capturedEvents.mouse.size(), capturedEvents.mouse.data());

		const auto resourceIx = m_realFrameIx % m_maxFramesInFlight;

		if (m_realFrameIx >= m_maxFramesInFlight)
		{
			const ISemaphore::SWaitInfo cbDonePending[] =
			{
				{
					.semaphore = m_semaphore.get(),
					.value = m_realFrameIx + 1 - m_maxFramesInFlight
				}
			};
			if (m_device->blockForSemaphores(cbDonePending) != ISemaphore::WAIT_RESULT::SUCCESS)
				return;
		}

		m_currentImageAcquire = m_surface->acquireNextImage();
		if (!m_currentImageAcquire)
			return;

		auto* const cb = m_cmdBufs.data()[resourceIx].get();
		cb->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
		cb->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
		cb->beginDebugMarker("NITE Tool Frame");

		auto* queue = getGraphicsQueue();

		asset::SViewport viewport;
		{
			viewport.minDepth = 1.f;
			viewport.maxDepth = 0.f;
			viewport.x = 0u;
			viewport.y = 0u;
			viewport.width = WIN_W;
			viewport.height = WIN_H;
		}
		cb->setViewport(0u, 1u, &viewport);
		{
			const VkRect2D currentRenderArea =
			{
				.offset = {0,0},
				.extent = {m_window->getWidth(),m_window->getHeight()}
			};

			const IGPUCommandBuffer::SClearColorValue clearValue = { .float32 = {0.f,0.f,0.f,1.f} };
			auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
			const IGPUCommandBuffer::SRenderpassBeginInfo info =
			{
				.framebuffer = scRes->getFramebuffer(m_currentImageAcquire.imageIndex),
				.colorClearValues = &clearValue,
				.depthStencilClearValues = nullptr,
				.renderArea = currentRenderArea
			};
			cb->beginRenderPass(info, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
		}

		ui->render(cb, resourceIx);
		cb->endRenderPass();
		cb->end();
		{
			const IQueue::SSubmitInfo::SSemaphoreInfo rendered[] =
			{
				{
					.semaphore = m_semaphore.get(),
					.value = ++m_submitIx,
					.stageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT
				}
			};
			{
				{
					const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[] =
					{
						{.cmdbuf = cb }
					};

					const IQueue::SSubmitInfo::SSemaphoreInfo acquired[] =
					{
						{
							.semaphore = m_currentImageAcquire.semaphore,
							.value = m_currentImageAcquire.acquireCount,
							.stageMask = PIPELINE_STAGE_FLAGS::NONE
						}
					};
					const IQueue::SSubmitInfo infos[] =
					{
						{
							.waitSemaphores = acquired,
							.commandBuffers = commandBuffers,
							.signalSemaphores = rendered
						}
					};

					if (queue->submit(infos) != IQueue::RESULT::SUCCESS)
						m_submitIx--;
				}
			}

			m_window->setCaption("[Nabla IMGUI Test Engine]");
			m_surface->present(m_currentImageAcquire.imageIndex, rendered);

			// Post swap Test Engine
			ImGuiTestEngine_PostSwap(engine);
		}
	}

	inline bool keepRunning() override
	{
		if (m_surface->irrecoverable())
			return false;

		return true;
	}

	inline bool onAppTerminated() override
	{
		int tested = 0, successed = 0;
		ImGuiTestEngine_GetResult(engine, tested, successed);
		const bool good = tested == successed;

		ImVector<ImGuiTest*>* tests = _NBL_NEW(ImVector<ImGuiTest*>);
		ImGuiTestEngine_GetTestList(engine, tests);

		if (successed < tested)
		{
			m_logger->log("Failing Tests: ", ILogger::ELL_ERROR);

			for (auto* test : *tests)
				if (test->Output.Status == ImGuiTestStatus_Error)
					m_logger->log("- " + std::string(test->Name), ILogger::ELL_ERROR);
		}

		m_logger->log(std::string("Tests Result: ") + (good ? "PASSING" : "FAILING"), (good ? ILogger::ELL_PERFORMANCE : ILogger::ELL_ERROR));
		m_logger->log(std::string("(") + std::to_string(successed) + "/" + std::to_string(tested) + " tests passed)", ILogger::ELL_PERFORMANCE);

		_NBL_DELETE(tests);
		ImGuiTestEngine_Stop(engine);

		return device_base_t::onAppTerminated() && good;
	}

private:
	smart_refctd_ptr<IWindow> m_window;
	smart_refctd_ptr<CSimpleResizeSurface<CDefaultSwapchainFramebuffers>> m_surface;
	smart_refctd_ptr<IGPUGraphicsPipeline> m_pipeline;
	smart_refctd_ptr<ISemaphore> m_semaphore;
	smart_refctd_ptr<IGPUCommandPool> m_cmdPool;
	uint64_t m_realFrameIx : 59 = 0;
	uint64_t m_submitIx : 59 = 0;
	uint64_t m_maxFramesInFlight : 5;
	std::array<smart_refctd_ptr<IGPUCommandBuffer>, ISwapchain::MaxImages> m_cmdBufs;
	ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};

	nbl::core::smart_refctd_ptr<nbl::ext::imgui::UI> ui;
	core::smart_refctd_ptr<InputSystem> m_inputSystem;
	InputSystem::ChannelReader<IMouseEventChannel> mouse;
	InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

	// Test engine
	ImGuiTestEngine* engine = nullptr;
};

NBL_MAIN_FUNC(NITETool)