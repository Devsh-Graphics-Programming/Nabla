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
    core::smart_refctd_ptr<nbl::ui::IWindowManager> windowManager;
    core::smart_refctd_ptr<nbl::ui::IWindow> window;
    core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCb;
    core::smart_refctd_ptr<nbl::video::IAPIConnection> apiConnection;
    core::smart_refctd_ptr<nbl::video::ISurface> surface;
    core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
    core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
    video::IPhysicalDevice* physicalDevice;
    std::array<video::IGPUQueue*, CommonAPI::InitOutput::MaxQueuesCount> queues;
    core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
    core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
    std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, CommonAPI::InitOutput::MaxSwapChainImageCount> fbo;
    std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, CommonAPI::InitOutput::MaxQueuesCount> commandPools; // TODO: Multibuffer and reset the commandpools
    core::smart_refctd_ptr<nbl::system::ISystem> system;
    core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
    video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
    core::smart_refctd_ptr<nbl::system::ILogger> logger;
    core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;
    video::IGPUObjectFromAssetConverter cpu2gpu;
    
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
    std::chrono::system_clock::time_point lastTime;
    bool frameDataFilled = false;
    size_t frame_count = 0ull;
    double time_sum = 0;
    double dtList[NBL_FRAMES_TO_AVERAGE] = {};

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

    APP_CONSTRUCTOR(MeshLoadersApp)
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

        const auto swapchainImageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_COLOR_ATTACHMENT_BIT);
        const video::ISurface::SFormat surfaceFormat(asset::EF_B8G8R8A8_SRGB, asset::ECP_COUNT, asset::EOTF_UNKNOWN);

        CommonAPI::InitOutput initOutput;
        initOutput.window = core::smart_refctd_ptr(window);
        CommonAPI::Init(
            initOutput,
            video::EAT_VULKAN,
            "MeshLoaders",
            requiredInstanceFeatures,
            optionalInstanceFeatures,
            requiredDeviceFeatures,
            optionalDeviceFeatures,
            WIN_W, WIN_H, SC_IMG_COUNT,
            swapchainImageUsage,
            surfaceFormat,
            nbl::asset::EF_D32_SFLOAT);

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
            quantNormalCache->loadCacheFromFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), "../../tmp/normalCache101010.sse");

            system::path archPath = CWDOnStartup / "../../media/sponza.zip";
            auto arch = system->openFileArchive(archPath);
            // test no alias loading (TODO: fix loading from absolute paths)
            system->mount(std::move(arch));
            asset::IAssetLoader::SAssetLoadParams loadParams;
            loadParams.workingDirectory = CWDOnStartup;
            loadParams.logger = logger.get();
            auto meshes_bundle = assetManager->getAsset((CWDOnStartup / "../../media/sponza.zip/sponza.obj").string(), loadParams);
            assert(!meshes_bundle.getContents().empty());

            metaOBJ = meshes_bundle.getMetadata()->selfCast<const asset::COBJMetadata>();

            auto cpuMesh = meshes_bundle.getContents().begin()[0];
            meshRaw = static_cast<asset::ICPUMesh*>(cpuMesh.get());

            quantNormalCache->saveCacheToFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), "../../tmp/normalCache101010.sse");
        }
        // we can safely assume that all meshbuffers within mesh loaded from OBJ has same DS1 layout (used for camera-specific data)
        firstMeshBuffer = *meshRaw->getMeshBuffers().begin();
        pipelineMetadata = metaOBJ->getAssetSpecificMetadata(firstMeshBuffer->getPipeline());

        // so we can create just one DS
        const asset::ICPUDescriptorSetLayout* ds1layout = firstMeshBuffer->getPipeline()->getLayout()->getDescriptorSetLayout(1u);
        ds1UboBinding = 0u;
        for (const auto& bnd : ds1layout->getBindings())
        {
            if (bnd.type == asset::EDT_UNIFORM_BUFFER)
            {
                ds1UboBinding = bnd.binding;
                break;
            }
        }

        size_t neededDS1UBOsz = 0ull;
        {
            for (const auto& shdrIn : pipelineMetadata->m_inputSemantics)
                if (shdrIn.descriptorSection.type == asset::IRenderpassIndependentPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set == 1u && shdrIn.descriptorSection.uniformBufferObject.binding == ds1UboBinding)
                    neededDS1UBOsz = std::max<size_t>(neededDS1UBOsz, shdrIn.descriptorSection.uniformBufferObject.relByteoffset + shdrIn.descriptorSection.uniformBufferObject.bytesize);
        }

        core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> gpuds1layout;
        {
            cpu2gpuParams.beginCommandBuffers();
            auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&ds1layout, &ds1layout + 1, cpu2gpuParams);
            if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
                assert(false);
            cpu2gpuParams.waitForCreationToComplete();
            gpuds1layout = (*gpu_array)[0];
        }

        descriptorPool = createDescriptorPool(1u);

        ubomemreq = logicalDevice->getDeviceLocalGPUMemoryReqs();
        ubomemreq.vulkanReqs.size = neededDS1UBOsz;
        video::IGPUBuffer::SCreationParams gpuuboCreationParams;
        gpuuboCreationParams.usage = static_cast<asset::IBuffer::E_USAGE_FLAGS>(asset::IBuffer::EUF_UNIFORM_BUFFER_BIT | asset::IBuffer::EUF_TRANSFER_DST_BIT);
        gpuuboCreationParams.sharingMode = asset::E_SHARING_MODE::ESM_EXCLUSIVE;
        gpuuboCreationParams.queueFamilyIndexCount = 0u;
        gpuuboCreationParams.queueFamilyIndices = nullptr;

        gpuubo = logicalDevice->createGPUBufferOnDedMem(gpuuboCreationParams,ubomemreq,true);
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
            for (size_t i = 0ull; i < meshRaw->getMeshBuffers().size(); ++i)
            {
                auto& meshBuffer = meshRaw->getMeshBuffers().begin()[i];

                for (size_t i = 0ull; i < nbl::asset::SBlendParams::MAX_COLOR_ATTACHMENT_COUNT; i++)
                    meshBuffer->getPipeline()->getBlendParams().blendParams[i].attachmentEnabled = (i == 0ull);

                meshBuffer->getPipeline()->getRasterizationParams().frontFaceIsCCW = false;

                // TODO: This should come in set by the loader
                asset::ICPUDescriptorSet* ds = meshBuffer->getAttachedDescriptorSet();
                for (uint32_t j = 0u; j < ds->getMaxDescriptorBindingIndex(); ++j)
                {
                    const size_t arrayElementCount = ds->getDescriptors(j).size();
                    for (size_t k = 0ull; k < arrayElementCount; ++k)
                    {
                        auto& currentDescriptor = ds->getDescriptors(j).begin()[k].desc;
                        if (currentDescriptor->getTypeCategory() == asset::IDescriptor::EC_IMAGE)
                        {
                            asset::ICPUImageView* cpuImageView = static_cast<asset::ICPUImageView*>(currentDescriptor.get());
                            const asset::E_FORMAT cpuImageViewFormat = cpuImageView->getCreationParameters().format;

                            asset::IImage::E_ASPECT_FLAGS aspectFlags = asset::IImage::EAF_COLOR_BIT;
                            if (isDepthOrStencilFormat(cpuImageViewFormat) && !isDepthOnlyFormat(cpuImageViewFormat))
                            {
                                if (isStencilOnlyFormat(cpuImageViewFormat))
                                    aspectFlags = asset::IImage::EAF_STENCIL_BIT;
                                else
                                    aspectFlags = asset::IImage::EAF_DEPTH_BIT;
                            }

                            cpuImageView->setAspectFlags(aspectFlags);
                        }
                    }
                }
            }

            cpu2gpuParams.beginCommandBuffers();
            auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&meshRaw, &meshRaw + 1, cpu2gpuParams);
            if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
                assert(false);
            cpu2gpuParams.waitForCreationToComplete();

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
        matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60), float(WIN_W) / WIN_H, 0.1, 1000);
        camera = Camera(cameraPosition, core::vectorSIMDf(0, 0, 0), projectionMatrix, 10.f, 1.f);
        lastTime = std::chrono::system_clock::now();

        for (size_t i = 0ull; i < NBL_FRAMES_TO_AVERAGE; ++i)
            dtList[i] = 0.0;

        logicalDevice->createCommandBuffers(
            commandPools[CommonAPI::InitOutput::EQT_GRAPHICS].get(),
            video::IGPUCommandBuffer::EL_PRIMARY,
            FRAMES_IN_FLIGHT,
            commandBuffers);

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
        logicalDevice->waitIdle();
    }
    void workLoopBody() override
    {
        ++resourceIx;
        if (resourceIx >= FRAMES_IN_FLIGHT)
            resourceIx = 0;

        auto& commandBuffer = commandBuffers[resourceIx];
        auto& fence = frameComplete[resourceIx];

        if (fence)
        {
            while (logicalDevice->waitForFences(1u, &fence.get(), false, MAX_TIMEOUT) == video::IGPUFence::ES_TIMEOUT) {}
            logicalDevice->resetFences(1u, &fence.get());
        }
        else
            fence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

        auto renderStart = std::chrono::system_clock::now();
        const auto renderDt = std::chrono::duration_cast<std::chrono::milliseconds>(renderStart - lastTime).count();
        lastTime = renderStart;
        { // Calculate Simple Moving Average for FrameTime
            time_sum -= dtList[frame_count];
            time_sum += renderDt;
            dtList[frame_count] = renderDt;
            frame_count++;
            if (frame_count >= NBL_FRAMES_TO_AVERAGE)
            {
                frameDataFilled = true;
                frame_count = 0;
            }

        }
        const double averageFrameTime = frameDataFilled ? (time_sum / (double)NBL_FRAMES_TO_AVERAGE) : (time_sum / frame_count);

#ifdef NBL_MORE_LOGS
        logger->log("renderDt = %f ------ averageFrameTime = %f", system::ILogger::ELL_INFO, renderDt, averageFrameTime);
#endif // NBL_MORE_LOGS

        auto averageFrameTimeDuration = std::chrono::duration<double, std::milli>(averageFrameTime);
        auto nextPresentationTime = renderStart + averageFrameTimeDuration;
        auto nextPresentationTimeStamp = std::chrono::duration_cast<std::chrono::microseconds>(nextPresentationTime.time_since_epoch());

        inputSystem->getDefaultMouse(&mouse);
        inputSystem->getDefaultKeyboard(&keyboard);

        camera.beginInputProcessing(nextPresentationTimeStamp);
        mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); }, logger.get());
        keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { camera.keyboardProcess(events); }, logger.get());
        camera.endInputProcessing(nextPresentationTimeStamp);

        const auto& viewMatrix = camera.getViewMatrix();
        const auto& viewProjectionMatrix = camera.getConcatenatedMatrix();

        commandBuffer->reset(nbl::video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
        commandBuffer->begin(0);

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

        swapchain->acquireNextImage(MAX_TIMEOUT, imageAcquire[resourceIx].get(), nullptr, &acquiredNextFBO);


        core::matrix3x4SIMD modelMatrix;
        modelMatrix.setTranslation(nbl::core::vectorSIMDf(0, 0, 0, 0));

        core::matrix4SIMD mvp = core::concatenateBFollowedByA(viewProjectionMatrix, modelMatrix);

        core::vector<uint8_t> uboData(gpuubo->getSize());
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

        commandBuffer->updateBuffer(gpuubo.get(), 0ull, gpuubo->getSize(), uboData.data());

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
            commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuRenderpassIndependentPipeline->getLayout(), 1u, 1u, &gpuds1_ptr, nullptr);
            const video::IGPUDescriptorSet* gpuds3_ptr = gpuMeshBuffer->getAttachedDescriptorSet();
            if (gpuds3_ptr)
                commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuRenderpassIndependentPipeline->getLayout(), 3u, 1u, &gpuds3_ptr, nullptr);
            commandBuffer->pushConstants(gpuRenderpassIndependentPipeline->getLayout(), video::IGPUShader::ESS_FRAGMENT, 0u, gpuMeshBuffer->MAX_PUSH_CONSTANT_BYTESIZE, gpuMeshBuffer->getPushConstantsDataPtr());

            commandBuffer->drawMeshBuffer(gpuMeshBuffer);
        }

        commandBuffer->endRenderPass();
        commandBuffer->end();

        CommonAPI::Submit(logicalDevice.get(),
            swapchain.get(), 
            commandBuffer.get(),
            queues[CommonAPI::InitOutput::EQT_GRAPHICS],
            imageAcquire[resourceIx].get(),
            renderFinished[resourceIx].get(),
            fence.get());
        CommonAPI::Present(logicalDevice.get(), 
            swapchain.get(),
            queues[CommonAPI::InitOutput::EQT_GRAPHICS],
            renderFinished[resourceIx].get(),
            acquiredNextFBO);
    }
    bool keepRunning() override
    {
        return windowCb->isWindowOpen();
    }
};

NBL_COMMON_API_MAIN(MeshLoadersApp)

extern "C" {  _declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001; }

