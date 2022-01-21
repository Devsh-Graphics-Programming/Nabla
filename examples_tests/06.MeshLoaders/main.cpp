// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nabla.h>

#include "../common/Camera.hpp"
#include "../common/CommonAPI.h"
#include "nbl/ext/ScreenShot/ScreenShot.h"

using namespace nbl;
using namespace core;
using namespace ui;
/*
    Uncomment for more detailed logging
*/

// #define NBL_MORE_LOGS

class MeshLoadersApp : public ApplicationBase
{
    constexpr static uint32_t WIN_W = 1280;
    constexpr static uint32_t WIN_H = 720;
    constexpr static uint32_t SC_IMG_COUNT = 3u;
    constexpr static uint32_t FRAMES_IN_FLIGHT = 5u;
    constexpr static uint64_t MAX_TIMEOUT = 99999999999999ull;
    constexpr static size_t NBL_FRAMES_TO_AVERAGE = 100ull;

    static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);
public:
    nbl::core::smart_refctd_ptr<nbl::ui::IWindowManager> windowManager;
    nbl::core::smart_refctd_ptr<nbl::ui::IWindow> window;
    nbl::core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCb;
    nbl::core::smart_refctd_ptr<nbl::video::IAPIConnection> apiConnection;
    nbl::core::smart_refctd_ptr<nbl::video::ISurface> surface;
    nbl::core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
    nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
    nbl::video::IPhysicalDevice* physicalDevice;
    std::array<video::IGPUQueue*, CommonAPI::InitOutput::MaxQueuesCount> queues;
    nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
    nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
    std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, CommonAPI::InitOutput::MaxSwapChainImageCount> fbo;
    std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, CommonAPI::InitOutput::MaxQueuesCount> commandPools;
    nbl::core::smart_refctd_ptr<nbl::system::ISystem> system;
    nbl::core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
    nbl::video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
    nbl::core::smart_refctd_ptr<nbl::system::ILogger> logger;
    nbl::core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;

    nbl::video::IGPUObjectFromAssetConverter cpu2gpu;
    
    core::smart_refctd_ptr<video::IDescriptorPool> descriptorPool;
    video::IDriverMemoryBacked::SDriverMemoryRequirements ubomemreq;
    core::smart_refctd_ptr<video::IGPUBuffer> gpuubo;
    core::smart_refctd_ptr<video::IGPUDescriptorSet> gpuds1;

    asset::ICPUMesh* meshRaw = nullptr;
    const asset::COBJMetadata* metaOBJ = nullptr;

    core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
    core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
    core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
    core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[FRAMES_IN_FLIGHT];

    CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> mouse;
    CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;
    Camera camera = Camera(vectorSIMDf(0, 0, 0), vectorSIMDf(0, 0, 0), matrix4SIMD());

    using RENDERPASS_INDEPENDENT_PIPELINE_ADRESS = size_t;
    std::map<RENDERPASS_INDEPENDENT_PIPELINE_ADRESS, core::smart_refctd_ptr<video::IGPUGraphicsPipeline>> gpuPipelines;
    core::smart_refctd_ptr<video::IGPUMesh> gpumesh;
    const asset::ICPUMeshBuffer* firstMeshBuffer;
    const nbl::asset::COBJMetadata::CRenderpassIndependentPipeline* pipelineMetadata;

    uint32_t ds1UboBinding = 0;
    int resourceIx;
    uint32_t acquiredNextFBO = {};
    std::chrono::steady_clock::time_point lastTime;
    bool frameDataFilled = false;
    size_t frame_count = 0ull;
    double time_sum = 0;
    double dtList[NBL_FRAMES_TO_AVERAGE] = {};

    video::CDumbPresentationOracle oracle;

    auto createDescriptorPool(const uint32_t textureCount)
    {
        constexpr uint32_t maxItemCount = 256u;
        {
            nbl::video::IDescriptorPool::SDescriptorPoolSize poolSize;
            poolSize.count = textureCount;
            poolSize.type = nbl::asset::EDT_COMBINED_IMAGE_SAMPLER;
            return logicalDevice->createDescriptorPool(static_cast<nbl::video::IDescriptorPool::E_CREATE_FLAGS>(0), maxItemCount, 1u, &poolSize);
        }
    }

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
        return apiConnection.get();
    }
    video::ILogicalDevice* getLogicalDevice()  override
    {
        return logicalDevice.get();
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
        swapchain = std::move(s);
    }
    uint32_t getSwapchainImageCount() override
    {
        return SC_IMG_COUNT;
    }
    virtual nbl::asset::E_FORMAT getDepthFormat() override
    {
        return nbl::asset::EF_D32_SFLOAT;
    }

    APP_CONSTRUCTOR(MeshLoadersApp)
    void onAppInitialized_impl() override
    {
        CommonAPI::InitOutput initOutput;
        initOutput.window = core::smart_refctd_ptr(window);
        initOutput.system = core::smart_refctd_ptr(system);

        const auto swapchainImageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_COLOR_ATTACHMENT_BIT);
        const video::ISurface::SFormat surfaceFormat(asset::EF_R8G8B8A8_SRGB, asset::ECP_SRGB, asset::EOTF_sRGB);

        CommonAPI::InitWithDefaultExt(initOutput, video::EAT_OPENGL_ES, "MeshLoaders", WIN_W, WIN_H, SC_IMG_COUNT, swapchainImageUsage, surfaceFormat, nbl::asset::EF_D32_SFLOAT);
        window = std::move(initOutput.window);
        windowCb = std::move(initOutput.windowCb);
        apiConnection = std::move(initOutput.apiConnection);
        surface = std::move(initOutput.surface);
        utilities = std::move(initOutput.utilities);
        logicalDevice = std::move(initOutput.logicalDevice);
        physicalDevice = initOutput.physicalDevice;
        queues = std::move(initOutput.queues);
        swapchain = std::move(initOutput.swapchain);
        renderpass = std::move(initOutput.renderpass);
        fbo = std::move(initOutput.fbo);
        commandPools = std::move(initOutput.commandPools);
        system = std::move(initOutput.system);
        assetManager = std::move(initOutput.assetManager);
        cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
        logger = std::move(initOutput.logger);
        inputSystem = std::move(initOutput.inputSystem);

        nbl::video::IGPUObjectFromAssetConverter cpu2gpu;
        {
            auto* quantNormalCache = assetManager->getMeshManipulator()->getQuantNormalCache();
            quantNormalCache->loadCacheFromFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), sharedOutputCWD / "normalCache101010.sse");

            system::path archPath = sharedInputCWD / "sponza.zip";
            auto arch = system->openFileArchive(archPath);
            // test no alias loading (TODO: fix loading from absolute paths)
            system->mount(std::move(arch));
            asset::IAssetLoader::SAssetLoadParams loadParams;
            loadParams.workingDirectory = sharedInputCWD;
            loadParams.logger = logger.get();
            auto meshes_bundle = assetManager->getAsset((sharedInputCWD / "sponza.zip/sponza.obj").string(), loadParams);
            assert(!meshes_bundle.getContents().empty());

            metaOBJ = meshes_bundle.getMetadata()->selfCast<const asset::COBJMetadata>();

            auto cpuMesh = meshes_bundle.getContents().begin()[0];
            meshRaw = static_cast<asset::ICPUMesh*>(cpuMesh.get());

            quantNormalCache->saveCacheToFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), sharedOutputCWD / "normalCache101010.sse");
        }
        // we can safely assume that all meshbuffers within mesh loaded from OBJ has same DS1 layout (used for camera-specific data)
        firstMeshBuffer = *meshRaw->getMeshBuffers().begin();
        pipelineMetadata = metaOBJ->getAssetSpecificMetadata(firstMeshBuffer->getPipeline());

        // so we can create just one DS
        const asset::ICPUDescriptorSetLayout* ds1layout = firstMeshBuffer->getPipeline()->getLayout()->getDescriptorSetLayout(1u);
        ds1UboBinding = 0u;
        for (const auto& bnd : ds1layout->getBindings())
            if (bnd.type == asset::EDT_UNIFORM_BUFFER)
            {
                ds1UboBinding = bnd.binding;
                break;
            }

        size_t neededDS1UBOsz = 0ull;
        {
            for (const auto& shdrIn : pipelineMetadata->m_inputSemantics)
                if (shdrIn.descriptorSection.type == asset::IRenderpassIndependentPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set == 1u && shdrIn.descriptorSection.uniformBufferObject.binding == ds1UboBinding)
                    neededDS1UBOsz = std::max<size_t>(neededDS1UBOsz, shdrIn.descriptorSection.uniformBufferObject.relByteoffset + shdrIn.descriptorSection.uniformBufferObject.bytesize);
        }

        core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> gpuds1layout;
        {
            auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&ds1layout, &ds1layout + 1, cpu2gpuParams);
            if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
                assert(false);

            gpuds1layout = (*gpu_array)[0];
        }

        descriptorPool = createDescriptorPool(1u);

        ubomemreq = logicalDevice->getDeviceLocalGPUMemoryReqs();
        ubomemreq.vulkanReqs.size = neededDS1UBOsz;
        video::IGPUBuffer::SCreationParams gpuuboCreationParams;
        gpuuboCreationParams.canUpdateSubRange = true;
        gpuuboCreationParams.usage = core::bitflag<asset::IBuffer::E_USAGE_FLAGS>(asset::IBuffer::EUF_UNIFORM_BUFFER_BIT) | asset::IBuffer::EUF_TRANSFER_DST_BIT;
        gpuuboCreationParams.sharingMode = asset::E_SHARING_MODE::ESM_EXCLUSIVE;
        gpuuboCreationParams.queueFamilyIndexCount = 0u;
        gpuuboCreationParams.queueFamilyIndices = nullptr;

        gpuubo = logicalDevice->createGPUBufferOnDedMem(gpuuboCreationParams,ubomemreq);
        gpuds1 = logicalDevice->createGPUDescriptorSet(descriptorPool.get(), std::move(gpuds1layout));

        {
            video::IGPUDescriptorSet::SWriteDescriptorSet write;
            write.dstSet = gpuds1.get();
            write.binding = ds1UboBinding;
            write.count = 1u;
            write.arrayElement = 0u;
            write.descriptorType = asset::EDT_UNIFORM_BUFFER;
            video::IGPUDescriptorSet::SDescriptorInfo info;
            {
                info.desc = gpuubo;
                info.buffer.offset = 0ull;
                info.buffer.size = neededDS1UBOsz;
            }
            write.info = &info;
            logicalDevice->updateDescriptorSets(1u, &write, 0u, nullptr);
        }
        {
            cpu2gpuParams.beginCommandBuffers();

            auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&meshRaw, &meshRaw + 1, cpu2gpuParams);
            if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
                assert(false);
            
            cpu2gpuParams.waitForCreationToComplete(false);

            gpumesh = (*gpu_array)[0];
        }
       
        {
            for (size_t i = 0; i < gpumesh->getMeshBuffers().size(); ++i)
            {
                auto gpuIndependentPipeline = gpumesh->getMeshBuffers().begin()[i]->getPipeline();

                nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams;
                graphicsPipelineParams.renderpassIndependent = core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline>(const_cast<video::IGPURenderpassIndependentPipeline*>(gpuIndependentPipeline));
                graphicsPipelineParams.renderpass = core::smart_refctd_ptr(renderpass);

                const RENDERPASS_INDEPENDENT_PIPELINE_ADRESS adress = reinterpret_cast<RENDERPASS_INDEPENDENT_PIPELINE_ADRESS>(graphicsPipelineParams.renderpassIndependent.get());
                gpuPipelines[adress] = logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(graphicsPipelineParams));
            }
        }

        core::vectorSIMDf cameraPosition(0, 5, -10);
        matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60.0f), float(WIN_W) / WIN_H, 0.1, 1000);
        camera = Camera(cameraPosition, core::vectorSIMDf(0, 0, 0), projectionMatrix, 10.f, 1.f);
        lastTime = std::chrono::steady_clock::now();

        for (size_t i = 0ull; i < NBL_FRAMES_TO_AVERAGE; ++i)
            dtList[i] = 0.0;

        oracle.reportBeginFrameRecord();
        logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_GRAPHICS].get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, commandBuffers);

        for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
        {
            imageAcquire[i] = logicalDevice->createSemaphore();
            renderFinished[i] = logicalDevice->createSemaphore();
        }

        constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;
        uint32_t acquiredNextFBO = {};
        resourceIx = -1;
    }
    void onAppTerminated_impl() override
    {
    }
    void workLoopBody() override
    {
        ++resourceIx;
        if (resourceIx >= FRAMES_IN_FLIGHT)
            resourceIx = 0;

        auto& commandBuffer = commandBuffers[resourceIx];
        auto& fence = frameComplete[resourceIx];
        if (fence)
            logicalDevice->blockForFences(1u, &fence.get());
        else
            fence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

        commandBuffer->reset(nbl::video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
        commandBuffer->begin(0);

        const auto nextPresentationTimestamp = oracle.acquireNextImage(swapchain.get(), imageAcquire[resourceIx].get(), nullptr, &acquiredNextFBO);
        {
            inputSystem->getDefaultMouse(&mouse);
            inputSystem->getDefaultKeyboard(&keyboard);

            camera.beginInputProcessing(nextPresentationTimestamp);
            mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); }, logger.get());
            keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { camera.keyboardProcess(events); }, logger.get());
            camera.endInputProcessing(nextPresentationTimestamp);
        }

        const auto& viewMatrix = camera.getViewMatrix();
        const auto& viewProjectionMatrix = camera.getConcatenatedMatrix();

        asset::SViewport viewport;
        viewport.minDepth = 1.f;
        viewport.maxDepth = 0.f;
        viewport.x = 0u;
        viewport.y = 0u;
        viewport.width = WIN_W;
        viewport.height = WIN_H;
        commandBuffer->setViewport(0u, 1u, &viewport);
        
        VkRect2D scissor = {};
        scissor.offset = { 0, 0 };
        scissor.extent = { WIN_W, WIN_H };
        commandBuffer->setScissor(0u, 1u, &scissor);

        core::matrix3x4SIMD modelMatrix;
        modelMatrix.setTranslation(nbl::core::vectorSIMDf(0, 0, 0, 0));
        core::matrix4SIMD mvp = core::concatenateBFollowedByA(viewProjectionMatrix, modelMatrix);

        const size_t uboSize = gpuubo->getCachedCreationParams().declaredSize;
        core::vector<uint8_t> uboData(uboSize);
        for (const auto& shdrIn : pipelineMetadata->m_inputSemantics)
        {
            if (shdrIn.descriptorSection.type == asset::IRenderpassIndependentPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set == 1u && shdrIn.descriptorSection.uniformBufferObject.binding == ds1UboBinding)
            {
                switch (shdrIn.type)
                {
                case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW_PROJ:
                {
                    memcpy(uboData.data() + shdrIn.descriptorSection.uniformBufferObject.relByteoffset, mvp.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
                } break;

                case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW:
                {
                    memcpy(uboData.data() + shdrIn.descriptorSection.uniformBufferObject.relByteoffset, viewMatrix.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
                } break;

                case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW_INVERSE_TRANSPOSE:
                {
                    memcpy(uboData.data() + shdrIn.descriptorSection.uniformBufferObject.relByteoffset, viewMatrix.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
                } break;
                }
            }
        }
        commandBuffer->updateBuffer(gpuubo.get(), 0ull, uboSize, uboData.data());
        
        nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
        {
            VkRect2D area;
            area.offset = { 0,0 };
            area.extent = { WIN_W, WIN_H };
            asset::SClearValue clear[2] = {};
            clear[0].color.float32[0] = 1.f;
            clear[0].color.float32[1] = 1.f;
            clear[0].color.float32[2] = 1.f;
            clear[0].color.float32[3] = 1.f;
            clear[1].depthStencil.depth = 0.f;

            beginInfo.clearValueCount = 2u;
            beginInfo.framebuffer = fbo[acquiredNextFBO];
            beginInfo.renderpass = renderpass;
            beginInfo.renderArea = area;
            beginInfo.clearValues = clear;
        }

        commandBuffer->beginRenderPass(&beginInfo, nbl::asset::ESC_INLINE);

        for (size_t i = 0; i < gpumesh->getMeshBuffers().size(); ++i)
        {
            auto gpuMeshBuffer = gpumesh->getMeshBuffers().begin()[i];
            auto gpuGraphicsPipeline = gpuPipelines[reinterpret_cast<RENDERPASS_INDEPENDENT_PIPELINE_ADRESS>(gpuMeshBuffer->getPipeline())];

            const video::IGPURenderpassIndependentPipeline* gpuRenderpassIndependentPipeline = gpuMeshBuffer->getPipeline();
            const video::IGPUDescriptorSet* ds3 = gpuMeshBuffer->getAttachedDescriptorSet();

            commandBuffer->bindGraphicsPipeline(gpuGraphicsPipeline.get());

            const video::IGPUDescriptorSet* gpuds1_ptr = gpuds1.get();
            commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuRenderpassIndependentPipeline->getLayout(), 1u, 1u, &gpuds1_ptr);
            const video::IGPUDescriptorSet* gpuds3_ptr = gpuMeshBuffer->getAttachedDescriptorSet();
            if (gpuds3_ptr)
                commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuRenderpassIndependentPipeline->getLayout(), 3u, 1u, &gpuds3_ptr);
            commandBuffer->pushConstants(gpuRenderpassIndependentPipeline->getLayout(), asset::IShader::ESS_FRAGMENT, 0u, gpuMeshBuffer->MAX_PUSH_CONSTANT_BYTESIZE, gpuMeshBuffer->getPushConstantsDataPtr());

            commandBuffer->drawMeshBuffer(gpuMeshBuffer);
        }

        commandBuffer->endRenderPass();
        commandBuffer->end();
        
        logicalDevice->resetFences(1, &fence.get());
        CommonAPI::Submit(logicalDevice.get(),
            swapchain.get(), 
            commandBuffer.get(),
            queues[CommonAPI::InitOutput::EQT_GRAPHICS],
            imageAcquire[resourceIx].get(),
            renderFinished[resourceIx].get(),
            fence.get());
        CommonAPI::Present(logicalDevice.get(), 
            swapchain.get(),
            queues[CommonAPI::InitOutput::EQT_GRAPHICS], renderFinished[resourceIx].get(), acquiredNextFBO);
    }
    bool keepRunning() override
    {
        return true;// windowCb->isWindowOpen();
    }
};

NBL_COMMON_API_MAIN(MeshLoadersApp)
