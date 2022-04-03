// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nabla.h>

#include "../common/Camera.hpp"
#include "../common/CommonAPI.h"

#include "nbl/asset/utils/IMeshManipulator.h"
#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/utilities/IGPUObjectFromAssetConverter.h"

#include "nbl/ext/DebugDraw/CDraw3DLine.h"
#include "nbl/ext/ScreenShot/ScreenShot.h"

using namespace nbl;
using namespace core;
using namespace ui;
using namespace asset;
using namespace video;

class OrientedBoundingBox : public ApplicationBase
{
  static constexpr uint32_t WIN_W = 1280;
  static constexpr uint32_t WIN_H = 720;
  static constexpr uint32_t SC_IMG_COUNT = 3u;
  static constexpr uint32_t FRAMES_IN_FLIGHT = 5u;
  static constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;
  static constexpr size_t NBL_FRAMES_TO_AVERAGE = 100ull;
  static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

  public:
    void setWindow(core::smart_refctd_ptr<nbl::ui::IWindow>&& wnd) override
    {
      m_window = std::move(wnd);
    }
    nbl::ui::IWindow* getWindow() override
    {
      return m_window.get();
    }
    void setSystem(core::smart_refctd_ptr<nbl::system::ISystem>&& s) override
    {
      m_system = std::move(s);
    }
    video::IAPIConnection* getAPIConnection() override
    {
      return m_apiConnection.get();
    }
    video::ILogicalDevice* getLogicalDevice()  override
    {
      return m_logicalDevice.get();
    }
    video::IGPURenderpass* getRenderpass() override
    {
      return m_renderpass.get();
    }
    void setSurface(core::smart_refctd_ptr<video::ISurface>&& s) override
    {
      m_surface = std::move(s);
    }
    void setFBOs(std::vector<core::smart_refctd_ptr<video::IGPUFramebuffer>>& f) override
    {
      for (int i = 0; i < f.size(); i++)
      {
        m_fbos[i] = core::smart_refctd_ptr(f[i]);
      }
    }
    void setSwapchain(core::smart_refctd_ptr<video::ISwapchain>&& s) override
    {
      m_swapchain = std::move(s);
    }
    uint32_t getSwapchainImageCount() override
    {
      return SC_IMG_COUNT;
    }
    virtual nbl::asset::E_FORMAT getDepthFormat() override
    {
      return nbl::asset::EF_D32_SFLOAT;
    }

    APP_CONSTRUCTOR(OrientedBoundingBox)

    bool keepRunning() override { return m_windowCallback->isWindowOpen(); }

    void onAppInitialized_impl() override
    {
      CommonAPI::InitOutput initOutput;
      initOutput.window = core::smart_refctd_ptr(m_window);
      initOutput.system = core::smart_refctd_ptr(m_system);

      const auto swapchainImageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(
        asset::IImage::EUF_COLOR_ATTACHMENT_BIT// | asset::IImage::EUF_STORAGE_BIT
      );
      const video::ISurface::SFormat surfaceFormat(
        asset::EF_R8G8B8A8_SRGB, asset::ECP_COUNT, asset::EOTF_UNKNOWN
      );

      CommonAPI::InitWithDefaultExt(
        initOutput,
        video::EAT_OPENGL,
        "61. Oriented Bounding Box",
        WIN_W, WIN_H, SC_IMG_COUNT,
        swapchainImageUsage,
        surfaceFormat,
        nbl::asset::EF_D32_SFLOAT
      );

      m_window = std::move(initOutput.window);
      m_apiConnection = std::move(initOutput.apiConnection);
      m_surface = std::move(initOutput.surface);
      m_logicalDevice = std::move(initOutput.logicalDevice);
      m_queues = initOutput.queues;
      m_swapchain = std::move(initOutput.swapchain);
      m_renderpass = std::move(initOutput.renderpass);
      m_fbos = std::move(initOutput.fbo);
      m_commandPools = std::move(initOutput.commandPools);
      m_assetManager = std::move(initOutput.assetManager);
      m_cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
      m_logger = std::move(initOutput.logger);
      m_inputSystem = std::move(initOutput.inputSystem);
      m_system = std::move(initOutput.system);
      m_windowCallback = std::move(initOutput.windowCb);
      m_utilities = std::move(initOutput.utilities);

      IAssetLoader::SAssetLoadParams loadParams;
      loadParams.workingDirectory = sharedInputCWD;
      loadParams.logger = m_logger.get();
      auto meshes_bundle = m_assetManager->getAsset((sharedInputCWD / "cow.obj").string(), loadParams);
      assert(!meshes_bundle.getContents().empty());

      auto metaOBJ = meshes_bundle.getMetadata()->selfCast<const asset::COBJMetadata>();
      auto mesh = meshes_bundle.getContents().begin()[0];
      auto mesh_raw = dynamic_cast<asset::ICPUMesh*>(mesh.get());
      const asset::ICPUMeshBuffer* cpuMB = mesh_raw->getMeshBuffers()[0];
      const auto cpuMBPipeline = cpuMB->getPipeline();

      auto vertexShaderBundle = m_assetManager->getAsset("../obb.vert", loadParams);
      auto fragShaderBundle = m_assetManager->getAsset("../obb.frag", loadParams);
      auto cpuVertexShader = IAsset::castDown<ICPUSpecializedShader>(vertexShaderBundle.getContents().begin()->get());
      auto cpuFragmentShader = IAsset::castDown<ICPUSpecializedShader>(fragShaderBundle.getContents().begin()->get());

      m_cpu2gpuParams.beginCommandBuffers();
      auto gpuVertexShader = m_cpu2gpu.getGPUObjectsFromAssets(&cpuVertexShader, &cpuVertexShader + 1, m_cpu2gpuParams)->front();
      auto gpuFragmentShader = m_cpu2gpu.getGPUObjectsFromAssets(&cpuFragmentShader, &cpuFragmentShader + 1, m_cpu2gpuParams)->front();
      m_cpu2gpuParams.waitForCreationToComplete();
      std::array<IGPUSpecializedShader*, 2> gpuShaders = { gpuVertexShader.get(), gpuFragmentShader.get() };

      asset::SPushConstantRange pcRange = { asset::ICPUShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD) };
      auto gpuPipelineLayout = m_logicalDevice->createGPUPipelineLayout(&pcRange, &pcRange + 1);

      const auto vtxCount = IMeshManipulator::calcVertexSize(cpuMB);
      const auto vtxInputParams = cpuMBPipeline->getVertexInputParams();

      {
        uint32_t posAttrIdx = cpuMB->getPositionAttributeIx();
        auto posPtr = const_cast<uint8_t *>(cpuMB->getAttribPointer(posAttrIdx));
        const uint32_t stride = vtxInputParams.bindings[0].stride;
        core::matrix3x4SIMD rs;
        const auto quarterPI = core::QUARTER_PI<float>();
        rs.setRotation(core::quaternion(0.0f, -quarterPI, -quarterPI));
        core::matrix3x4SIMD s;
        //s.setScale(core::vectorSIMDf(10.0f, 1.0f, 1.0f));
        rs = core::matrix3x4SIMD::concatenateBFollowedByA(rs, s);

        for (size_t i = 0ull; i < vtxCount; i++)
        {
          auto vtxPos = cpuMB->getPosition(i);
          rs.pseudoMulWith4x1(vtxPos);
          memcpy(posPtr, vtxPos.pointer, sizeof(float) * 3);
          posPtr += stride;
        }
      }

      m_draw3DLine = ext::DebugDraw::CDraw3DLine::create(m_logicalDevice);
      core::OBB obb = IMeshManipulator::calculateOrientedBBox(cpuMB, vtxCount);
      m_draw3DLine->addBox(core::aabbox3df(), 0, 1, 0, 1, obb.asMat3x4);
      core::smart_refctd_ptr<video::IGPUFence> fence;
      m_draw3DLine->updateVertexBuffer(m_utilities.get(), m_queues[CommonAPI::InitOutput::EQT_GRAPHICS], &fence);
      m_logicalDevice->waitForFences(1, const_cast<video::IGPUFence**>(&fence.get()), false, MAX_TIMEOUT);

      asset::SBlendParams blendParams;
      asset::SPrimitiveAssemblyParams primAsmParams;
      asset::SRasterizationParams rasterParams;
      rasterParams.faceCullingMode = asset::EFCM_NONE;

      m_gpuPipeline = m_logicalDevice->createGPURenderpassIndependentPipeline(
        nullptr,
        std::move(gpuPipelineLayout),
        gpuShaders.data(),
        gpuShaders.data() + gpuShaders.size(),
        vtxInputParams,
        blendParams,
        primAsmParams,
        rasterParams
      );

      nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams;

      graphicsPipelineParams.renderpassIndependent = core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline>(m_gpuPipeline.get());
      graphicsPipelineParams.renderpass = core::smart_refctd_ptr(m_renderpass);

      m_gpuGraphicsPipeline = m_logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(graphicsPipelineParams));

      m_camera = Camera(
        core::vectorSIMDf(0, -2, 1),
        core::vectorSIMDf(0, 0, 0),
        matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60.0f), float(WIN_W) / WIN_H, 0.1, 1000),
        10.f, 1.f
      );

      m_logicalDevice->createCommandBuffers(
        m_commandPools[CommonAPI::InitOutput::EQT_GRAPHICS].get(),
        video::IGPUCommandBuffer::EL_PRIMARY,
        FRAMES_IN_FLIGHT, m_commandBuffers
      );

      m_cpu2gpuParams.beginCommandBuffers();
      m_gpuMeshBuffer = m_cpu2gpu.getGPUObjectsFromAssets(&cpuMB, &cpuMB + 1, m_cpu2gpuParams)->front();
      m_cpu2gpuParams.waitForCreationToComplete();

      for(uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
      {
        m_imageAcquire[i] = m_logicalDevice->createSemaphore();
        m_renderFinished[i] = m_logicalDevice->createSemaphore();
      }

      m_oracle.reportBeginFrameRecord();
    }

    void workLoopBody() override
    {
      ++m_resourceIx;
      if (m_resourceIx >= FRAMES_IN_FLIGHT)
        m_resourceIx = 0;

      auto& commandBuffer = m_commandBuffers[m_resourceIx];
      auto& fence = m_frameComplete[m_resourceIx];

      if(fence)
      {
        while(m_logicalDevice->waitForFences(1u, &fence.get(), false, MAX_TIMEOUT) == video::IGPUFence::ES_TIMEOUT) {}
        m_logicalDevice->resetFences(1u, &fence.get());
      }
      else
      {
        fence = m_logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));
      }

      const auto nextPresentationTimestamp = m_oracle.acquireNextImage(m_swapchain.get(), m_imageAcquire[m_resourceIx].get(), nullptr, &m_acquiredNextFBO);

      m_inputSystem->getDefaultMouse(&m_mouse);
      m_inputSystem->getDefaultKeyboard(&m_keyboard);

      m_camera.beginInputProcessing(nextPresentationTimestamp);
      m_mouse.consumeEvents([=](const IMouseEventChannel::range_t& events) { m_camera.mouseProcess(events); }, m_logger.get());
      m_keyboard.consumeEvents([=](const IKeyboardEventChannel::range_t& events) { m_camera.keyboardProcess(events); }, m_logger.get());
      m_camera.endInputProcessing(nextPresentationTimestamp);

      commandBuffer->reset(nbl::video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
      commandBuffer->begin(0);

      {
        asset::SViewport viewport = {};
        viewport.minDepth = 1.f;
        viewport.maxDepth = 0.f;
        viewport.x = 0u;
        viewport.y = 0u;
        viewport.width = WIN_W;
        viewport.height = WIN_H;
        commandBuffer->setViewport(0u, 1u, &viewport);
        VkRect2D scissor;
        scissor.offset = {0u,0u};
        scissor.extent = {WIN_W,WIN_H};
        commandBuffer->setScissor(0u,1u,&scissor);

        m_swapchain->acquireNextImage(MAX_TIMEOUT, m_imageAcquire[m_resourceIx].get(), nullptr, &m_acquiredNextFBO);

        const auto& viewProjection = m_camera.getConcatenatedMatrix();

        m_draw3DLine->setViewProjMatrix(viewProjection);

        nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
        {
          VkRect2D area;
          area.offset = { 0,0 };
          area.extent = { WIN_W, WIN_H };
          asset::SClearValue clear[2] = {};
          clear[0].color.float32[0] = 0.f;
          clear[0].color.float32[1] = 0.f;
          clear[0].color.float32[2] = 0.f;
          clear[0].color.float32[3] = 1.f;
          clear[1].depthStencil.depth = 0.f;

          beginInfo.clearValueCount = 2u;
          beginInfo.framebuffer = m_fbos[m_acquiredNextFBO];
          beginInfo.renderpass = m_renderpass;
          beginInfo.renderArea = area;
          beginInfo.clearValues = clear;
        }

        commandBuffer->beginRenderPass(&beginInfo, nbl::asset::ESC_INLINE);

        m_draw3DLine->recordToCommandBuffer(commandBuffer.get(), m_gpuGraphicsPipeline.get());
        commandBuffer->drawMeshBuffer(m_gpuMeshBuffer.get());

        commandBuffer->endRenderPass();
      }

      commandBuffer->end();

      CommonAPI::Submit(
        m_logicalDevice.get(), m_swapchain.get(),commandBuffer.get(),
        m_queues[CommonAPI::InitOutput::EQT_GRAPHICS],
        m_imageAcquire[m_resourceIx].get(),m_renderFinished[m_resourceIx].get(),
        fence.get()
      );
      CommonAPI::Present(
        m_logicalDevice.get(), m_swapchain.get(),
        m_queues[CommonAPI::InitOutput::EQT_GRAPHICS],
        m_renderFinished[m_resourceIx].get(), m_acquiredNextFBO
      );
    }

    void onAppTerminated_impl() override
    {
      const auto& fboCreationParams = m_fbos[m_acquiredNextFBO]->getCreationParameters();
      auto gpuSourceImageView = fboCreationParams.attachments[0];

      bool status = ext::ScreenShot::createScreenShot(
        m_logicalDevice.get(),
        m_queues[CommonAPI::InitOutput::EQT_TRANSFER_DOWN],
        m_renderFinished[m_resourceIx].get(),
        gpuSourceImageView.get(),
        m_assetManager.get(),
        "ScreenShot.png",
        asset::EIL_PRESENT_SRC,
        static_cast<asset::E_ACCESS_FLAGS>(0u));

      assert(status);
    }

  private:
    nbl::core::smart_refctd_ptr<nbl::ui::IWindow> m_window;
    nbl::core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> m_windowCallback;
    nbl::core::smart_refctd_ptr<nbl::video::IAPIConnection> m_apiConnection;
    nbl::core::smart_refctd_ptr<nbl::video::ISurface> m_surface;
    nbl::core::smart_refctd_ptr<nbl::video::IUtilities> m_utilities;
    nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> m_logicalDevice;
    std::array<nbl::video::IGPUQueue*, CommonAPI::InitOutput::MaxQueuesCount> m_queues = { nullptr, nullptr, nullptr, nullptr };
    nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> m_swapchain;
    nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> m_renderpass;
    std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, CommonAPI::InitOutput::MaxSwapChainImageCount> m_fbos;
    std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, CommonAPI::InitOutput::MaxQueuesCount> m_commandPools;
    nbl::core::smart_refctd_ptr<nbl::system::ISystem> m_system;
    nbl::core::smart_refctd_ptr<nbl::asset::IAssetManager> m_assetManager;
    nbl::video::IGPUObjectFromAssetConverter::SParams m_cpu2gpuParams;
    nbl::core::smart_refctd_ptr<nbl::system::ILogger> m_logger;
    nbl::core::smart_refctd_ptr<CommonAPI::InputSystem> m_inputSystem;

    nbl::video::IGPUObjectFromAssetConverter m_cpu2gpu;
    core::smart_refctd_ptr<video::IGPUMeshBuffer> m_gpuMeshBuffer;

    core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> m_gpuPipeline;
    core::smart_refctd_ptr<IGPUGraphicsPipeline> m_gpuGraphicsPipeline;
    core::smart_refctd_ptr<ext::DebugDraw::CDraw3DLine> m_draw3DLine;

    video::CDumbPresentationOracle m_oracle;

    uint32_t m_acquiredNextFBO = {};
    int m_resourceIx = -1;

    core::smart_refctd_ptr<video::IGPUCommandBuffer> m_commandBuffers[FRAMES_IN_FLIGHT];

    core::smart_refctd_ptr<video::IGPUFence> m_frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
    core::smart_refctd_ptr<video::IGPUSemaphore> m_imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
    core::smart_refctd_ptr<video::IGPUSemaphore> m_renderFinished[FRAMES_IN_FLIGHT] = { nullptr };

    CommonAPI::InputSystem::ChannelReader<ui::IMouseEventChannel> m_mouse;
    CommonAPI::InputSystem::ChannelReader<ui::IKeyboardEventChannel> m_keyboard;

    Camera m_camera = Camera(core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 0, 0), core::matrix4SIMD());
};

NBL_COMMON_API_MAIN(OrientedBoundingBox)