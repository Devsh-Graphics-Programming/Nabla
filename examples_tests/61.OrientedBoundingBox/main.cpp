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
using namespace asset;
using namespace video;

class OrientedBoundingBox : public ApplicationBase
{
  template<typename T>
  using shared_ptr = core::smart_refctd_ptr<T>;

  template<typename T>
  using make_shared_ptr = core::smart_refctd_ptr<T>;

  using IWindow                   = ui::IWindow;
  using IMouseEventChannel        = ui::IMouseEventChannel;
  using IKeyboardEventChannel     = ui::IKeyboardEventChannel;

  static constexpr uint32_t WIN_W = 1280;
  static constexpr uint32_t WIN_H = 720;
  static constexpr uint32_t SC_IMG_COUNT = 3u;
  static constexpr uint32_t FRAMES_IN_FLIGHT = 5u;
  static constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;
  static constexpr size_t NBL_FRAMES_TO_AVERAGE = 100ull;
  static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

  public:
    void            setSystem   (shared_ptr<system::ISystem>&& s)             override {}
    void            setWindow   (shared_ptr<IWindow>&& wnd)                   override {}
    void            setSurface  (shared_ptr<ISurface>&& s)                    override {}
    void            setSwapchain(shared_ptr<ISwapchain>&& s)                  override {}
    void            setFBOs     (std::vector<shared_ptr<IGPUFramebuffer>>& f) override {}
    IWindow*        getWindow()                                               override { return m_window.get(); }
    IAPIConnection* getAPIConnection()                                        override { return m_apiConnection.get(); }
    ILogicalDevice* getLogicalDevice()                                        override { return m_logicalDevice.get(); }
    IGPURenderpass* getRenderpass()                                           override { return m_renderpass.get(); }
    uint32_t        getSwapchainImageCount()                                  override { return SC_IMG_COUNT; }
    asset::E_FORMAT getDepthFormat()                                          override { return asset::EF_D32_SFLOAT; }

    APP_CONSTRUCTOR(OrientedBoundingBox)

    bool keepRunning() override { return m_windowCallback->isWindowOpen(); }

    void onAppInitialized_impl() override
    {
      CommonAPI::InitOutput initOutput;
      DEBUG_DATA debugData(DEBUG_DATA::MESH_NAME::FLOWER);

      CommonAPI::InitWithDefaultExt(
        initOutput,
        video::EAT_OPENGL,
        "61. Oriented Bounding Box",
        WIN_W, WIN_H, SC_IMG_COUNT,
        static_cast<IImage::E_USAGE_FLAGS>(IImage::EUF_COLOR_ATTACHMENT_BIT),
        ISurface::SFormat(asset::EF_R8G8B8A8_SRGB, asset::ECP_SRGB, asset::EOTF_sRGB),
        getDepthFormat()
      );

      m_queues          = initOutput.queues;
      m_swapchain       = std::move(initOutput.swapchain);
      m_logicalDevice   = std::move(initOutput.logicalDevice);
      m_renderpass      = std::move(initOutput.renderpass);
      m_fbos            = std::move(initOutput.fbo);
      m_commandPools    = std::move(initOutput.commandPools);
      m_assetManager    = std::move(initOutput.assetManager);
      m_cpu2gpuParams   = std::move(initOutput.cpu2gpuParams);
      m_logger          = std::move(initOutput.logger);
      m_inputSystem     = std::move(initOutput.inputSystem);
      m_windowCallback  = std::move(initOutput.windowCb);
      m_utilities       = std::move(initOutput.utilities);

      m_camera = Camera(
        debugData.CAM_POS(),
        core::vectorSIMDf(0, 0, 0),
        core::matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(debugData.CAM_FOV()), float(WIN_W) / WIN_H, 0.1, 1000),
        10.f, 1.f
      );

      IAssetLoader::SAssetLoadParams loadParams;
      loadParams.workingDirectory = sharedInputCWD;
      loadParams.logger = m_logger.get();
      loadParams.isOBBDisabled = false;

      auto meshes_bundle = m_assetManager->getAsset((sharedInputCWD / debugData.MESH_PATH()).string(), loadParams);
      assert(!meshes_bundle.getContents().empty());

      const auto mesh = meshes_bundle.getContents().begin()[0];
      const auto mesh_raw = dynamic_cast<ICPUMesh*>(mesh.get());
      const auto* cpuMB = mesh_raw->getMeshBuffers()[0];
      const auto cpuMBPipeline = cpuMB->getPipeline();
      const auto vtxInputParams = cpuMBPipeline->getVertexInputParams();

      // mesh draw pipeline
      {
        core::matrix3x4SIMD modelMatrix;
        modelMatrix.setRotation(core::quaternion(0, 1, 0));
//        modelMatrix.setTranslation(core::vectorSIMDf(0, 0, 0, 0));

        auto vertexShaderBundle = m_assetManager->getAsset("../example_mesh.vert", loadParams);
        auto fragShaderBundle = m_assetManager->getAsset("../example_mesh.frag", loadParams);
        auto cpuVertexShader = IAsset::castDown<ICPUSpecializedShader>(vertexShaderBundle.getContents().begin()->get());
        auto cpuFragmentShader = IAsset::castDown<ICPUSpecializedShader>(fragShaderBundle.getContents().begin()->get());

        m_cpu2gpuParams.beginCommandBuffers();
        auto gpuVertexShader = m_cpu2gpu.getGPUObjectsFromAssets(&cpuVertexShader, &cpuVertexShader + 1, m_cpu2gpuParams)->front();
        auto gpuFragmentShader = m_cpu2gpu.getGPUObjectsFromAssets(&cpuFragmentShader, &cpuFragmentShader + 1, m_cpu2gpuParams)->front();
        m_cpu2gpuParams.waitForCreationToComplete();

        std::array<IGPUSpecializedShader*, 2> gpuShaders = { gpuVertexShader.get(), gpuFragmentShader.get() };

        SPushConstantRange pcRange = { ICPUShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD) };
        auto gpuPipelineLayout = m_logicalDevice->createGPUPipelineLayout(&pcRange, &pcRange + 1);

        SBlendParams blendParams;
        SPrimitiveAssemblyParams primAsmParams;
        SRasterizationParams rasterParams;
        rasterParams.faceCullingMode = asset::EFCM_NONE;

        auto meshPipeline = m_logicalDevice->createGPURenderpassIndependentPipeline(
          nullptr,
          std::move(gpuPipelineLayout),
          gpuShaders.data(),
          gpuShaders.data() + gpuShaders.size(),
          vtxInputParams,
          blendParams,
          primAsmParams,
          rasterParams
        );
        IGPUGraphicsPipeline::SCreationParams meshPipelineParams;
        meshPipelineParams.renderpassIndependent = make_shared_ptr<IGPURenderpassIndependentPipeline>(meshPipeline.get());
        meshPipelineParams.renderpass = make_shared_ptr<IGPURenderpass>(m_renderpass);
        m_gpuMeshPipeline = m_logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(meshPipelineParams));
      }

      // obb debug draw pipeline
      {
        m_draw3DLine = ext::DebugDraw::CDraw3DLine::create(m_logicalDevice, true);
        m_draw3DLine->setViewProj(m_camera.getConcatenatedMatrix());

        if(loadParams.isOBBDisabled) m_draw3DLine->addBox(cpuMB->getBoundingBox());
        else
        {
          const auto& bbox = cpuMB->getOrientedBoundingBox();
          m_draw3DLine->addBox(bbox, bbox.orientation);
        }

        shared_ptr<IGPUFence> fence;
        m_draw3DLine->updateVertexBuffer(m_utilities.get(), m_queues[CommonAPI::InitOutput::EQT_GRAPHICS], &fence);
        m_logicalDevice->waitForFences(1, const_cast<IGPUFence**>(&fence.get()), false, MAX_TIMEOUT);

        auto obbPipeline = m_draw3DLine->getRenderpassIndependentPipeline();
        IGPUGraphicsPipeline::SCreationParams obbPipelineParams;
        obbPipelineParams.renderpassIndependent = make_shared_ptr<IGPURenderpassIndependentPipeline>(obbPipeline);
        obbPipelineParams.renderpass = make_shared_ptr<IGPURenderpass>(m_renderpass);
        m_gpuOBBPipeline = m_logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(obbPipelineParams));
      }

      m_logicalDevice->createCommandBuffers(
        m_commandPools[CommonAPI::InitOutput::EQT_GRAPHICS].get(),
        IGPUCommandBuffer::EL_PRIMARY,
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
      if(m_resourceIx >= FRAMES_IN_FLIGHT) m_resourceIx = 0;

      const auto& gpuQueue = m_queues[CommonAPI::InitOutput::EQT_GRAPHICS];
      const auto& logicalDevicePtr = getLogicalDevice();
      const auto& swapchainPtr  = m_swapchain.get();
      const auto& renderSemaphorePtr = m_renderFinished[m_resourceIx].get();
      const auto& imageSemaphorePtr = m_imageAcquire[m_resourceIx].get();
      auto& commandBuffer = m_commandBuffers[m_resourceIx];
      auto& fence = m_frameComplete[m_resourceIx];

      if(fence)
      {
        while(m_logicalDevice->waitForFences(1u, &fence.get(), false, MAX_TIMEOUT) == IGPUFence::ES_TIMEOUT) {}
        m_logicalDevice->resetFences(1u, &fence.get());
      }
      else
      { fence = m_logicalDevice->createFence(static_cast<IGPUFence::E_CREATE_FLAGS>(0)); }

      const auto nextPresentationTimestamp = m_oracle.acquireNextImage(swapchainPtr, imageSemaphorePtr, nullptr, &m_acquiredNextFBO);

      m_inputSystem->getDefaultMouse(&m_mouse);
      m_inputSystem->getDefaultKeyboard(&m_keyboard);

      m_camera.beginInputProcessing(nextPresentationTimestamp);
      m_mouse.consumeEvents([=](const IMouseEventChannel::range_t& events) { m_camera.mouseProcess(events); }, m_logger.get());
      m_keyboard.consumeEvents([=](const IKeyboardEventChannel::range_t& events) { m_camera.keyboardProcess(events); }, m_logger.get());
      m_camera.endInputProcessing(nextPresentationTimestamp);

      commandBuffer->reset(IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
      commandBuffer->begin(0);

      {
        SViewport viewport = {};
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

        m_swapchain->acquireNextImage(MAX_TIMEOUT, imageSemaphorePtr, nullptr, &m_acquiredNextFBO);

        IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
        {
          VkRect2D area;
          area.offset = { 0,0 };
          area.extent = { WIN_W, WIN_H };
          SClearValue clear[2] = {};
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

        commandBuffer->beginRenderPass(&beginInfo, asset::ESC_INLINE);

        const auto& viewProj = m_camera.getConcatenatedMatrix();

        // draw mesh
        {
          const auto meshIndependentPipeline = m_gpuMeshBuffer->getPipeline();
          const auto layout = const_cast<IGPUPipelineLayout*>(meshIndependentPipeline->getLayout());

          commandBuffer->pushConstants(layout, IShader::ESS_VERTEX, 0, sizeof(viewProj), &viewProj);
          commandBuffer->bindGraphicsPipeline(m_gpuMeshPipeline.get());
          commandBuffer->drawMeshBuffer(m_gpuMeshBuffer.get());
        }

        // debug draw obb
        m_draw3DLine->setViewProj(viewProj);
        m_draw3DLine->recordToCommandBuffer(commandBuffer.get(), m_gpuOBBPipeline.get());

        commandBuffer->endRenderPass();
      }

      commandBuffer->end();

      CommonAPI::Submit(
        logicalDevicePtr, swapchainPtr, commandBuffer.get(),
        gpuQueue,
        imageSemaphorePtr, renderSemaphorePtr,
        fence.get()
      );
      CommonAPI::Present(
        logicalDevicePtr, swapchainPtr,
        gpuQueue,
        renderSemaphorePtr, m_acquiredNextFBO
      );
    }

    void onAppTerminated_impl() override
    {
      const auto& fboCreationParams = m_fbos[m_acquiredNextFBO]->getCreationParameters();
      auto gpuSourceImageView = fboCreationParams.attachments[0];

      // TODO: is upvector upside down at this point?
      bool status = ext::ScreenShot::createScreenShot(
        getLogicalDevice(),
        m_queues[CommonAPI::InitOutput::EQT_TRANSFER_DOWN],
        m_renderFinished[m_resourceIx].get(),
        gpuSourceImageView.get(),
        m_assetManager.get(),
        "ScreenShot.png",
        asset::EIL_PRESENT_SRC,
        static_cast<asset::E_ACCESS_FLAGS>(0u)
      );

      assert(status);
      m_logicalDevice->waitIdle();
    }

  private:
    shared_ptr<IWindow> m_window;
    shared_ptr<CommonAPI::CommonAPIEventCallback> m_windowCallback;
    shared_ptr<IAPIConnection> m_apiConnection;
    shared_ptr<IUtilities> m_utilities;
    shared_ptr<ILogicalDevice> m_logicalDevice;
    shared_ptr<ISwapchain> m_swapchain;
    shared_ptr<IGPURenderpass> m_renderpass;
    shared_ptr<system::ISystem> m_system;
    shared_ptr<IAssetManager> m_assetManager;
    shared_ptr<system::ILogger> m_logger;
    shared_ptr<CommonAPI::InputSystem> m_inputSystem;

    shared_ptr<IGPUMeshBuffer> m_gpuMeshBuffer;

    shared_ptr<IGPUGraphicsPipeline> m_gpuMeshPipeline;
    shared_ptr<IGPUGraphicsPipeline> m_gpuOBBPipeline;
    shared_ptr<ext::DebugDraw::CDraw3DLine> m_draw3DLine;

    shared_ptr<IGPUCommandBuffer> m_commandBuffers[FRAMES_IN_FLIGHT];
    shared_ptr<IGPUFence> m_frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
    shared_ptr<IGPUSemaphore> m_imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
    shared_ptr<IGPUSemaphore> m_renderFinished[FRAMES_IN_FLIGHT] = { nullptr };

    std::array<IGPUQueue*, CommonAPI::InitOutput::MaxQueuesCount> m_queues = { nullptr, nullptr, nullptr, nullptr };
    std::array<shared_ptr<IGPUFramebuffer>, CommonAPI::InitOutput::MaxSwapChainImageCount> m_fbos;
    std::array<shared_ptr<IGPUCommandPool>, CommonAPI::InitOutput::MaxQueuesCount> m_commandPools;

    IGPUObjectFromAssetConverter::SParams m_cpu2gpuParams;
    IGPUObjectFromAssetConverter m_cpu2gpu;

    CDumbPresentationOracle m_oracle;

    uint32_t m_acquiredNextFBO = {};
    int m_resourceIx = -1;

    CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> m_mouse;
    CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> m_keyboard;

    Camera m_camera;

  private:
    struct DEBUG_DATA
    {
      static constexpr const auto COW     = "cow.obj";
      static constexpr const auto FLOWER  = "yellowflower.obj";
      static constexpr const auto SPANNER = "ply/Spanner-ply.ply";
      static constexpr const auto GUN     = "Cerberus_by_Andrew_Maximov/Cerberus_LP.obj";
      enum class MESH_NAME { COW, FLOWER, SPANNER, GUN }; // COW, FLOWER, SPANNER, GUN
      MESH_NAME meshName;
      explicit DEBUG_DATA(const MESH_NAME& _meshName = MESH_NAME::COW) : meshName(_meshName) {}
      inline float CAM_FOV() const noexcept
      {
        switch(meshName)
        {
          case MESH_NAME::COW:      return 20.f;
          case MESH_NAME::FLOWER:   return 35.f;
          case MESH_NAME::SPANNER:  return 95.f;
          case MESH_NAME::GUN:      return 80.f;
        } return 0.f;
      }
      inline const char* const MESH_PATH() const noexcept
      {
        switch(meshName)
        {
          case MESH_NAME::COW:      return COW;
          case MESH_NAME::FLOWER:   return FLOWER;
          case MESH_NAME::SPANNER:  return SPANNER;
          case MESH_NAME::GUN:      return GUN;
        } return "N/A";
      }
      inline core::vectorSIMDf CAM_POS() const noexcept
      {
        switch(meshName)
        {
          case MESH_NAME::COW:
          case MESH_NAME::FLOWER:   return core::vectorSIMDf(-4, 5, 5);
          case MESH_NAME::SPANNER:  return core::vectorSIMDf(-4, 5, -70);
          case MESH_NAME::GUN:      return core::vectorSIMDf(-100, 30, -140);
        } return {};
      }
    };
};

NBL_COMMON_API_MAIN(OrientedBoundingBox)