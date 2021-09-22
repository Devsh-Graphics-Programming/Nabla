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

class MeshLoaderApp : public ApplicationBase
{
    constexpr static uint32_t WIN_W = 1280;
    constexpr static uint32_t WIN_H = 720;
    constexpr static uint32_t SC_IMG_COUNT = 3u;
    constexpr static uint32_t FRAMES_IN_FLIGHT = 5u;
    constexpr static uint64_t MAX_TIMEOUT = 99999999999999ull;
    constexpr static size_t NBL_FRAMES_TO_AVERAGE = 100ull;

    static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);
public:
    struct nabla
    {
        nbl::core::smart_refctd_ptr<nbl::ui::IWindow> window;
        nbl::core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCb;
        nbl::core::smart_refctd_ptr<nbl::video::IAPIConnection> apiConnection;
        nbl::core::smart_refctd_ptr<nbl::video::ISurface> surface;
        nbl::core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
        nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
        nbl::video::IPhysicalDevice* physicalDevice;
        std::array<nbl::video::IGPUQueue*, CommonAPI::InitOutput<SC_IMG_COUNT>::EQT_COUNT> queues = { nullptr, nullptr, nullptr, nullptr };
        nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
        nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
        std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, SC_IMG_COUNT> fbo;
        nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool> commandPool; // TODO: Multibuffer and reset the commandpools
        nbl::core::smart_refctd_ptr<nbl::system::ISystem> system;
        nbl::core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
        nbl::video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
        nbl::core::smart_refctd_ptr<nbl::system::ILogger> logger;
        nbl::core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;

        nbl::core::smart_refctd_ptr<video::IGPUFence> gpuTransferFence;
        nbl::core::smart_refctd_ptr<video::IGPUFence> gpuComputeFence;
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
        std::chrono::system_clock::time_point lastTime;
        bool frameDataFilled = false;
        size_t frame_count = 0ull;
        double time_sum = 0;
        double dtList[NBL_FRAMES_TO_AVERAGE] = {};

        void cpu2gpuWaitForFences()
        {
            video::IGPUFence::E_STATUS waitStatus = video::IGPUFence::ES_NOT_READY;
            while (waitStatus != video::IGPUFence::ES_SUCCESS)
            {
                waitStatus = logicalDevice->waitForFences(1u, &gpuTransferFence.get(), false, 999999999ull);
                if (waitStatus == video::IGPUFence::ES_ERROR)
                    assert(false);
                else if (waitStatus == video::IGPUFence::ES_TIMEOUT)
                    break;
            }

            waitStatus = video::IGPUFence::ES_NOT_READY;
            while (waitStatus != video::IGPUFence::ES_SUCCESS)
            {
                waitStatus = logicalDevice->waitForFences(1u, &gpuComputeFence.get(), false, 999999999ull);
                if (waitStatus == video::IGPUFence::ES_ERROR)
                    assert(false);
                else if (waitStatus == video::IGPUFence::ES_TIMEOUT)
                    break;
            }
        }
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
    };
    APP_CONSTRUCTOR(MeshLoaderApp)
    void onAppInitialized_impl(void* data) override
    {
        nabla* engine = (nabla*)data;
        auto initOutput = CommonAPI::Init<WIN_W, WIN_H, SC_IMG_COUNT>(video::EAT_OPENGL, "MeshLoaders", nbl::asset::EF_D32_SFLOAT);
        engine->window = std::move(initOutput.window);
        engine->windowCb = std::move(initOutput.windowCb);
        engine->apiConnection = std::move(initOutput.apiConnection);
        engine->surface = std::move(initOutput.surface);
        engine->utilities = std::move(initOutput.utilities);
        engine->logicalDevice = std::move(initOutput.logicalDevice);
        engine->physicalDevice = initOutput.physicalDevice;
        engine->queues = std::move(initOutput.queues);
        engine->swapchain = std::move(initOutput.swapchain);
        engine->renderpass = std::move(initOutput.renderpass);
        engine->fbo = std::move(initOutput.fbo);
        engine->commandPool = std::move(initOutput.commandPool);
        engine->system = std::move(initOutput.system);
        engine->assetManager = std::move(initOutput.assetManager);
        engine->cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
        engine->logger = std::move(initOutput.logger);
        engine->inputSystem = std::move(initOutput.inputSystem);

        engine->gpuTransferFence = engine->logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));
        engine->gpuComputeFence = engine->logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

        nbl::video::IGPUObjectFromAssetConverter cpu2gpu;
        {
            engine->cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].fence = &engine->gpuTransferFence;
            engine->cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].fence = &engine->gpuComputeFence;
        }

        {
            auto* quantNormalCache = engine->assetManager->getMeshManipulator()->getQuantNormalCache();
            quantNormalCache->loadCacheFromFile<asset::EF_A2B10G10R10_SNORM_PACK32>(engine->system.get(), "../../tmp/normalCache101010.sse");

            system::path archPath = CWD / "../../media/sponza.zip";
            auto arch = engine->system->openFileArchive(archPath);
            // test no alias loading (TODO: fix loading from absolute paths)
            engine->system->mount(std::move(arch));
            asset::IAssetLoader::SAssetLoadParams loadParams;
            loadParams.workingDirectory = CWD;
            loadParams.logger = engine->logger.get();
            auto meshes_bundle = engine->assetManager->getAsset((CWD / "../../media/sponza.zip/sponza.obj").string(), loadParams);
            assert(!meshes_bundle.getContents().empty());

            engine->metaOBJ = meshes_bundle.getMetadata()->selfCast<const asset::COBJMetadata>();

            auto cpuMesh = meshes_bundle.getContents().begin()[0];
            engine->meshRaw = static_cast<asset::ICPUMesh*>(cpuMesh.get());

            quantNormalCache->saveCacheToFile<asset::EF_A2B10G10R10_SNORM_PACK32>(engine->system.get(), "../../tmp/normalCache101010.sse");
        }
        // we can safely assume that all meshbuffers within mesh loaded from OBJ has same DS1 layout (used for camera-specific data)
        engine->firstMeshBuffer = *engine->meshRaw->getMeshBuffers().begin();
        engine->pipelineMetadata = engine->metaOBJ->getAssetSpecificMetadata(engine->firstMeshBuffer->getPipeline());

        // so we can create just one DS
        const asset::ICPUDescriptorSetLayout* ds1layout = engine->firstMeshBuffer->getPipeline()->getLayout()->getDescriptorSetLayout(1u);
        engine->ds1UboBinding = 0u;
        for (const auto& bnd : ds1layout->getBindings())
            if (bnd.type == asset::EDT_UNIFORM_BUFFER)
            {
                engine->ds1UboBinding = bnd.binding;
                break;
            }

        size_t neededDS1UBOsz = 0ull;
        {
            for (const auto& shdrIn : engine->pipelineMetadata->m_inputSemantics)
                if (shdrIn.descriptorSection.type == asset::IRenderpassIndependentPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set == 1u && shdrIn.descriptorSection.uniformBufferObject.binding == engine->ds1UboBinding)
                    neededDS1UBOsz = std::max<size_t>(neededDS1UBOsz, shdrIn.descriptorSection.uniformBufferObject.relByteoffset + shdrIn.descriptorSection.uniformBufferObject.bytesize);
        }

        core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> gpuds1layout;
        {
            auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&ds1layout, &ds1layout + 1, engine->cpu2gpuParams);
            if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
                assert(false);

            //cpu2gpuWaitForFences();
            gpuds1layout = (*gpu_array)[0];
        }

        engine->descriptorPool = engine->createDescriptorPool(1u);

        engine->ubomemreq = engine->logicalDevice->getDeviceLocalGPUMemoryReqs();
        engine->ubomemreq.vulkanReqs.size = neededDS1UBOsz;
        engine->gpuubo = engine->logicalDevice->createGPUBufferOnDedMem(engine->ubomemreq, true);
        engine->gpuds1 = engine->logicalDevice->createGPUDescriptorSet(engine->descriptorPool.get(), std::move(gpuds1layout));
        {
            video::IGPUDescriptorSet::SWriteDescriptorSet write;
            write.dstSet = engine->gpuds1.get();
            write.binding = engine->ds1UboBinding;
            write.count = 1u;
            write.arrayElement = 0u;
            write.descriptorType = asset::EDT_UNIFORM_BUFFER;
            video::IGPUDescriptorSet::SDescriptorInfo info;
            {
                info.desc = engine->gpuubo;
                info.buffer.offset = 0ull;
                info.buffer.size = neededDS1UBOsz;
            }
            write.info = &info;
            engine->logicalDevice->updateDescriptorSets(1u, &write, 0u, nullptr);
        }
        {
            auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&engine->meshRaw, &engine->meshRaw + 1, engine->cpu2gpuParams);
            if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
                assert(false);

            engine->cpu2gpuWaitForFences();
            engine->gpumesh = (*gpu_array)[0];
        }

       
        {
            for (size_t i = 0; i < engine->gpumesh->getMeshBuffers().size(); ++i)
            {
                auto gpuIndependentPipeline = engine->gpumesh->getMeshBuffers().begin()[i]->getPipeline();

                nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams;
                graphicsPipelineParams.renderpassIndependent = core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline>(const_cast<video::IGPURenderpassIndependentPipeline*>(gpuIndependentPipeline));
                graphicsPipelineParams.renderpass = core::smart_refctd_ptr(engine->renderpass);

                const nabla::RENDERPASS_INDEPENDENT_PIPELINE_ADRESS adress = reinterpret_cast<nabla::RENDERPASS_INDEPENDENT_PIPELINE_ADRESS>(graphicsPipelineParams.renderpassIndependent.get());
                engine->gpuPipelines[adress] = engine->logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(graphicsPipelineParams));
            }
        }

        

        core::vectorSIMDf cameraPosition(0, 5, -10);
        matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60), float(WIN_W) / WIN_H, 0.1, 1000);
        engine->camera = Camera(cameraPosition, core::vectorSIMDf(0, 0, 0), projectionMatrix, 10.f, 1.f);
        engine->lastTime = std::chrono::system_clock::now();

        for (size_t i = 0ull; i < NBL_FRAMES_TO_AVERAGE; ++i)
            engine->dtList[i] = 0.0;

        engine->logicalDevice->createCommandBuffers(engine->commandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, engine->commandBuffers);

        for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
        {
            engine->imageAcquire[i] = engine->logicalDevice->createSemaphore();
            engine->renderFinished[i] = engine->logicalDevice->createSemaphore();
        }

        constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;
        uint32_t acquiredNextFBO = {};
        engine->resourceIx = -1;
    }
    void onAppTerminated_impl(void* data) override
    {
        nabla* engine = (nabla*)data;

    }
    void workLoopBody(void* data) override
    {
        nabla* engine = (nabla*)data;
        ++engine->resourceIx;
        if (engine->resourceIx >= FRAMES_IN_FLIGHT)
            engine->resourceIx = 0;

        auto& commandBuffer = engine->commandBuffers[engine->resourceIx];
        auto& fence = engine->frameComplete[engine->resourceIx];

        if (fence)
            while (engine->logicalDevice->waitForFences(1u, &fence.get(), false, MAX_TIMEOUT) == video::IGPUFence::ES_TIMEOUT) {}
        else
            fence = engine->logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

        auto renderStart = std::chrono::system_clock::now();
        const auto renderDt = std::chrono::duration_cast<std::chrono::milliseconds>(renderStart - engine->lastTime).count();
        engine->lastTime = renderStart;
        { // Calculate Simple Moving Average for FrameTime
            engine->time_sum -= engine->dtList[engine->frame_count];
            engine->time_sum += renderDt;
            engine->dtList[engine->frame_count] = renderDt;
            engine->frame_count++;
            if (engine->frame_count >= NBL_FRAMES_TO_AVERAGE)
            {
                engine->frameDataFilled = true;
                engine->frame_count = 0;
            }

        }
        const double averageFrameTime = engine->frameDataFilled ? (engine->time_sum / (double)NBL_FRAMES_TO_AVERAGE) : (engine->time_sum / engine->frame_count);

#ifdef NBL_MORE_LOGS
        logger->log("renderDt = %f ------ averageFrameTime = %f", system::ILogger::ELL_INFO, renderDt, averageFrameTime);
#endif // NBL_MORE_LOGS

        auto averageFrameTimeDuration = std::chrono::duration<double, std::milli>(averageFrameTime);
        auto nextPresentationTime = renderStart + averageFrameTimeDuration;
        auto nextPresentationTimeStamp = std::chrono::duration_cast<std::chrono::microseconds>(nextPresentationTime.time_since_epoch());

        engine->inputSystem->getDefaultMouse(&engine->mouse);
        engine->inputSystem->getDefaultKeyboard(&engine->keyboard);

        engine->camera.beginInputProcessing(nextPresentationTimeStamp);
        engine->mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { engine->camera.mouseProcess(events); }, engine->logger.get());
        engine->keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { engine->camera.keyboardProcess(events); }, engine->logger.get());
        engine->camera.endInputProcessing(nextPresentationTimeStamp);

        const auto& viewMatrix = engine->camera.getViewMatrix();
        const auto& viewProjectionMatrix = engine->camera.getConcatenatedMatrix();

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

        engine->swapchain->acquireNextImage(MAX_TIMEOUT, engine->imageAcquire[engine->resourceIx].get(), nullptr, &engine->acquiredNextFBO);

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
            beginInfo.framebuffer = engine->fbo[engine->acquiredNextFBO];
            beginInfo.renderpass = engine->renderpass;
            beginInfo.renderArea = area;
            beginInfo.clearValues = clear;
        }

        commandBuffer->beginRenderPass(&beginInfo, nbl::asset::ESC_INLINE);

        core::matrix3x4SIMD modelMatrix;
        modelMatrix.setTranslation(nbl::core::vectorSIMDf(0, 0, 0, 0));

        core::matrix4SIMD mvp = core::concatenateBFollowedByA(viewProjectionMatrix, modelMatrix);

        core::vector<uint8_t> uboData(engine->gpuubo->getSize());
        for (const auto& shdrIn : engine->pipelineMetadata->m_inputSemantics)
        {
            if (shdrIn.descriptorSection.type == asset::IRenderpassIndependentPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set == 1u && shdrIn.descriptorSection.uniformBufferObject.binding == engine->ds1UboBinding)
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

        commandBuffer->updateBuffer(engine->gpuubo.get(), 0ull, engine->gpuubo->getSize(), uboData.data());

        for (size_t i = 0; i < engine->gpumesh->getMeshBuffers().size(); ++i)
        {
            auto gpuMeshBuffer = engine->gpumesh->getMeshBuffers().begin()[i];
            auto gpuGraphicsPipeline = engine->gpuPipelines[reinterpret_cast<nabla::RENDERPASS_INDEPENDENT_PIPELINE_ADRESS>(gpuMeshBuffer->getPipeline())];

            const video::IGPURenderpassIndependentPipeline* gpuRenderpassIndependentPipeline = gpuMeshBuffer->getPipeline();
            const video::IGPUDescriptorSet* ds3 = gpuMeshBuffer->getAttachedDescriptorSet();

            commandBuffer->bindGraphicsPipeline(gpuGraphicsPipeline.get());

            const video::IGPUDescriptorSet* gpuds1_ptr = engine->gpuds1.get();
            commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuRenderpassIndependentPipeline->getLayout(), 1u, 1u, &gpuds1_ptr, nullptr);
            const video::IGPUDescriptorSet* gpuds3_ptr = gpuMeshBuffer->getAttachedDescriptorSet();
            if (gpuds3_ptr)
                commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuRenderpassIndependentPipeline->getLayout(), 3u, 1u, &gpuds3_ptr, nullptr);
            commandBuffer->pushConstants(gpuRenderpassIndependentPipeline->getLayout(), video::IGPUSpecializedShader::ESS_FRAGMENT, 0u, gpuMeshBuffer->MAX_PUSH_CONSTANT_BYTESIZE, gpuMeshBuffer->getPushConstantsDataPtr());

            commandBuffer->drawMeshBuffer(gpuMeshBuffer);
        }

        commandBuffer->endRenderPass();
        commandBuffer->end();

        CommonAPI::Submit(engine->logicalDevice.get(),
            engine->swapchain.get(), 
            commandBuffer.get(),
            engine->queues[CommonAPI::InitOutput<1>::EQT_GRAPHICS],
            engine->imageAcquire[engine->resourceIx].get(),
            engine->renderFinished[engine->resourceIx].get(),
            fence.get());
        CommonAPI::Present(engine->logicalDevice.get(), 
            engine->swapchain.get(),
            engine->queues[CommonAPI::InitOutput<1>::EQT_GRAPHICS], engine->renderFinished[engine->resourceIx].get(), engine->acquiredNextFBO);
    }
    bool keepRunning(void* params) override
    {
        nabla* engine = (nabla*)params;
        return engine->windowCb->isWindowOpen();
    }
};

NBL_COMMON_API_MAIN(MeshLoaderApp, MeshLoaderApp::nabla)
//int main(int argc, char** argv)
//{
//    system::path CWD = system::path(argv[0]).parent_path().generic_string() + "/";
//
//    auto window = std::move(initOutput.window);
//    auto gl = std::move(initOutput.apiConnection);
//    auto surface = std::move(initOutput.surface);
//    auto gpuPhysicalDevice = std::move(initOutput.physicalDevice);
//    auto logicalDevice = std::move(initOutput.logicalDevice);
//    auto queues = std::move(initOutput.queues);
//    auto swapchain = std::move(initOutput.swapchain);
//    auto renderpass = std::move(initOutput.renderpass);
//    auto fbos = std::move(initOutput.fbo);
//    auto commandPool = std::move(initOutput.commandPool);
//    auto assetManager = std::move(initOutput.assetManager);
//    auto logger = std::move(initOutput.logger);
//    auto inputSystem = std::move(initOutput.inputSystem);
//    auto system = std::move(initOutput.system);
//    auto windowCallback = std::move(initOutput.windowCb);
//    auto cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
//    auto utilities = std::move(initOutput.utilities);
//
//    auto gpuTransferFence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));
//    auto gpuComputeFence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));
//    
//    nbl::video::IGPUObjectFromAssetConverter cpu2gpu;
//    {
//        cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].fence = &gpuTransferFence;
//        cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].fence = &gpuComputeFence;
//    }
//
//    auto cpu2gpuWaitForFences = [&]() -> void
//    {
//        video::IGPUFence::E_STATUS waitStatus = video::IGPUFence::ES_NOT_READY;
//        while (waitStatus != video::IGPUFence::ES_SUCCESS)
//        {
//            waitStatus = logicalDevice->waitForFences(1u, &gpuTransferFence.get(), false, 999999999ull);
//            if (waitStatus == video::IGPUFence::ES_ERROR)
//                assert(false);
//            else if (waitStatus == video::IGPUFence::ES_TIMEOUT)
//                break;
//        }
//
//        waitStatus = video::IGPUFence::ES_NOT_READY;
//        while (waitStatus != video::IGPUFence::ES_SUCCESS)
//        {
//            waitStatus = logicalDevice->waitForFences(1u, &gpuComputeFence.get(), false, 999999999ull);
//            if (waitStatus == video::IGPUFence::ES_ERROR)
//                assert(false);
//            else if (waitStatus == video::IGPUFence::ES_TIMEOUT)
//                break;
//        }
//    };
//    
//    auto createDescriptorPool = [&](const uint32_t textureCount)
//    {
//        constexpr uint32_t maxItemCount = 256u;
//        {
//            nbl::video::IDescriptorPool::SDescriptorPoolSize poolSize;
//            poolSize.count = textureCount;
//            poolSize.type = nbl::asset::EDT_COMBINED_IMAGE_SAMPLER;
//            return logicalDevice->createDescriptorPool(static_cast<nbl::video::IDescriptorPool::E_CREATE_FLAGS>(0), maxItemCount, 1u, &poolSize);
//        }
//    };
//
//    asset::ICPUMesh* meshRaw = nullptr;
//    const asset::COBJMetadata* metaOBJ = nullptr;
//    {        
//        auto* quantNormalCache = assetManager->getMeshManipulator()->getQuantNormalCache();
//        quantNormalCache->loadCacheFromFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), "../../tmp/normalCache101010.sse");
//
//        system::path archPath = CWD/"../../media/sponza.zip";
//        auto arch = system->openFileArchive(archPath);
//        // test no alias loading (TODO: fix loading from absolute paths)
//        system->mount(std::move(arch));
//        asset::IAssetLoader::SAssetLoadParams loadParams;
//        loadParams.workingDirectory = CWD;
//        loadParams.logger = logger.get();
//        auto meshes_bundle = assetManager->getAsset((CWD/"../../media/sponza.zip/sponza.obj").string(), loadParams);
//        assert(!meshes_bundle.getContents().empty());
//
//        metaOBJ = meshes_bundle.getMetadata()->selfCast<const asset::COBJMetadata>();
//
//        auto cpuMesh = meshes_bundle.getContents().begin()[0];
//        meshRaw = static_cast<asset::ICPUMesh*>(cpuMesh.get());
//
//        quantNormalCache->saveCacheToFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), "../../tmp/normalCache101010.sse");
//    }
//
//    // we can safely assume that all meshbuffers within mesh loaded from OBJ has same DS1 layout (used for camera-specific data)
//    const asset::ICPUMeshBuffer* const firstMeshBuffer = *meshRaw->getMeshBuffers().begin();
//    auto pipelineMetadata = metaOBJ->getAssetSpecificMetadata(firstMeshBuffer->getPipeline());
//
//    // so we can create just one DS
//    const asset::ICPUDescriptorSetLayout* ds1layout = firstMeshBuffer->getPipeline()->getLayout()->getDescriptorSetLayout(1u);
//    uint32_t ds1UboBinding = 0u;
//    for (const auto& bnd : ds1layout->getBindings())
//        if (bnd.type==asset::EDT_UNIFORM_BUFFER)
//        {
//            ds1UboBinding = bnd.binding;
//            break;
//        }
//
//    size_t neededDS1UBOsz = 0ull;
//    {
//        for (const auto& shdrIn : pipelineMetadata->m_inputSemantics)
//            if (shdrIn.descriptorSection.type==asset::IRenderpassIndependentPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set==1u && shdrIn.descriptorSection.uniformBufferObject.binding==ds1UboBinding)
//                neededDS1UBOsz = std::max<size_t>(neededDS1UBOsz, shdrIn.descriptorSection.uniformBufferObject.relByteoffset+shdrIn.descriptorSection.uniformBufferObject.bytesize);
//    }
//
//    core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> gpuds1layout;
//    {
//        auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&ds1layout, &ds1layout + 1, cpu2gpuParams);
//        if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
//            assert(false);
//
//        //cpu2gpuWaitForFences();
//        gpuds1layout = (*gpu_array)[0];
//    }
//
//    auto descriptorPool = createDescriptorPool(1u);
//
//    auto ubomemreq = logicalDevice->getDeviceLocalGPUMemoryReqs();
//    ubomemreq.vulkanReqs.size = neededDS1UBOsz;
//    auto gpuubo = logicalDevice->createGPUBufferOnDedMem(ubomemreq,true);
//    auto gpuds1 = logicalDevice->createGPUDescriptorSet(descriptorPool.get(), std::move(gpuds1layout));
//    {
//        video::IGPUDescriptorSet::SWriteDescriptorSet write;
//        write.dstSet = gpuds1.get();
//        write.binding = ds1UboBinding;
//        write.count = 1u;
//        write.arrayElement = 0u;
//        write.descriptorType = asset::EDT_UNIFORM_BUFFER;
//        video::IGPUDescriptorSet::SDescriptorInfo info;
//        {
//            info.desc = gpuubo;
//            info.buffer.offset = 0ull;
//            info.buffer.size = neededDS1UBOsz;
//        }
//        write.info = &info;
//        logicalDevice->updateDescriptorSets(1u, &write, 0u, nullptr);
//    }
//
//    core::smart_refctd_ptr<video::IGPUMesh> gpumesh;
//    {
//        auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&meshRaw, &meshRaw + 1, cpu2gpuParams);
//        if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
//            assert(false);
//
//        cpu2gpuWaitForFences();
//        gpumesh = (*gpu_array)[0];
//    }
//
//    using RENDERPASS_INDEPENDENT_PIPELINE_ADRESS = size_t;
//    std::map<RENDERPASS_INDEPENDENT_PIPELINE_ADRESS, core::smart_refctd_ptr<video::IGPUGraphicsPipeline>> gpuPipelines;
//    {
//        for (size_t i = 0; i < gpumesh->getMeshBuffers().size(); ++i)
//        {
//            auto gpuIndependentPipeline = gpumesh->getMeshBuffers().begin()[i]->getPipeline();
//
//            nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams;
//            graphicsPipelineParams.renderpassIndependent = core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline>(const_cast<video::IGPURenderpassIndependentPipeline*>(gpuIndependentPipeline));
//            graphicsPipelineParams.renderpass = core::smart_refctd_ptr(renderpass);
//
//            const RENDERPASS_INDEPENDENT_PIPELINE_ADRESS adress = reinterpret_cast<RENDERPASS_INDEPENDENT_PIPELINE_ADRESS>(graphicsPipelineParams.renderpassIndependent.get());
//            gpuPipelines[adress] = logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(graphicsPipelineParams));
//        }
//    }
//
//    CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> mouse;
//    CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;
//
//    core::vectorSIMDf cameraPosition(0, 5, -10);
//    matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60), float(WIN_W) / WIN_H, 0.1, 1000);
//    Camera camera = Camera(cameraPosition, core::vectorSIMDf(0, 0, 0), projectionMatrix, 10.f, 1.f);
//    auto lastTime = std::chrono::system_clock::now();
//
//    constexpr size_t NBL_FRAMES_TO_AVERAGE = 100ull;
//    bool frameDataFilled = false;
//    size_t frame_count = 0ull;
//    double time_sum = 0;
//    double dtList[NBL_FRAMES_TO_AVERAGE] = {};
//    for (size_t i = 0ull; i < NBL_FRAMES_TO_AVERAGE; ++i)
//        dtList[i] = 0.0;
//
//    core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[FRAMES_IN_FLIGHT];
//    logicalDevice->createCommandBuffers(commandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, commandBuffers);
//
//    core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
//    core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
//    core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
//
//    for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
//    {
//        imageAcquire[i] = logicalDevice->createSemaphore();
//        renderFinished[i] = logicalDevice->createSemaphore();
//    }
//
//    constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;
//    uint32_t acquiredNextFBO = {};
//    auto resourceIx = -1;
//
//	while(windowCallback->isWindowOpen())
//	{
//        ++resourceIx;
//        if (resourceIx >= FRAMES_IN_FLIGHT)
//            resourceIx = 0;
//        
//        auto& commandBuffer = commandBuffers[resourceIx];
//        auto& fence = frameComplete[resourceIx];
//
//        if (fence)
//            while (logicalDevice->waitForFences(1u, &fence.get(), false, MAX_TIMEOUT) == video::IGPUFence::ES_TIMEOUT) {}
//        else
//            fence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));
//
//        auto renderStart = std::chrono::system_clock::now();
//        const auto renderDt = std::chrono::duration_cast<std::chrono::milliseconds>(renderStart - lastTime).count();
//        lastTime = renderStart;
//        { // Calculate Simple Moving Average for FrameTime
//            time_sum -= dtList[frame_count];
//            time_sum += renderDt;
//            dtList[frame_count] = renderDt;
//            frame_count++;
//            if (frame_count >= NBL_FRAMES_TO_AVERAGE) 
//			{
//				frameDataFilled = true;
//				frame_count = 0;
//			}
//                
//        }
//        const double averageFrameTime = frameDataFilled ? (time_sum / (double)NBL_FRAMES_TO_AVERAGE) : (time_sum / frame_count);
//
//        #ifdef NBL_MORE_LOGS
//        logger->log("renderDt = %f ------ averageFrameTime = %f", system::ILogger::ELL_INFO, renderDt, averageFrameTime);
//        #endif // NBL_MORE_LOGS
//
//        auto averageFrameTimeDuration = std::chrono::duration<double, std::milli>(averageFrameTime);
//        auto nextPresentationTime = renderStart + averageFrameTimeDuration;
//        auto nextPresentationTimeStamp = std::chrono::duration_cast<std::chrono::microseconds>(nextPresentationTime.time_since_epoch());
//
//        inputSystem->getDefaultMouse(&mouse);
//        inputSystem->getDefaultKeyboard(&keyboard);
//
//        camera.beginInputProcessing(nextPresentationTimeStamp);
//        mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); }, logger.get());
//        keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { camera.keyboardProcess(events); }, logger.get());
//        camera.endInputProcessing(nextPresentationTimeStamp);
//
//        const auto& viewMatrix = camera.getViewMatrix();
//        const auto& viewProjectionMatrix = camera.getConcatenatedMatrix();
//
//        commandBuffer->reset(nbl::video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
//        commandBuffer->begin(0);
//        
//        asset::SViewport viewport;
//        viewport.minDepth = 1.f;
//        viewport.maxDepth = 0.f;
//        viewport.x = 0u;
//        viewport.y = 0u;
//        viewport.width = WIN_W;
//        viewport.height = WIN_H;
//        commandBuffer->setViewport(0u, 1u, &viewport);
//
//        swapchain->acquireNextImage(MAX_TIMEOUT, imageAcquire[resourceIx].get(), nullptr, &acquiredNextFBO);
//
//        nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
//        {
//            VkRect2D area;
//            area.offset = { 0,0 };
//            area.extent = { WIN_W, WIN_H };
//            asset::SClearValue clear[2] = {};
//            clear[0].color.float32[0] = 1.f;
//            clear[0].color.float32[1] = 1.f;
//            clear[0].color.float32[2] = 1.f;
//            clear[0].color.float32[3] = 1.f;
//            clear[1].depthStencil.depth = 0.f;
//
//            beginInfo.clearValueCount = 2u;
//            beginInfo.framebuffer = fbos[acquiredNextFBO];
//            beginInfo.renderpass = renderpass;
//            beginInfo.renderArea = area;
//            beginInfo.clearValues = clear;
//        }
//        
//        commandBuffer->beginRenderPass(&beginInfo, nbl::asset::ESC_INLINE);
//
//        core::matrix3x4SIMD modelMatrix;
//        modelMatrix.setTranslation(nbl::core::vectorSIMDf(0, 0, 0, 0));
//
//        core::matrix4SIMD mvp = core::concatenateBFollowedByA(viewProjectionMatrix, modelMatrix);
//
//        core::vector<uint8_t> uboData(gpuubo->getSize());
//        for (const auto& shdrIn : pipelineMetadata->m_inputSemantics)
//        {
//            if (shdrIn.descriptorSection.type==asset::IRenderpassIndependentPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set==1u && shdrIn.descriptorSection.uniformBufferObject.binding==ds1UboBinding)
//            {
//                switch (shdrIn.type)
//                {
//                    case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW_PROJ:
//                    {
//                        memcpy(uboData.data()+shdrIn.descriptorSection.uniformBufferObject.relByteoffset, mvp.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
//                    } break;
//                    
//                    case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW:
//                    {
//                        memcpy(uboData.data() + shdrIn.descriptorSection.uniformBufferObject.relByteoffset, viewMatrix.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
//                    } break;
//                    
//                    case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW_INVERSE_TRANSPOSE:
//                    {
//                        memcpy(uboData.data()+shdrIn.descriptorSection.uniformBufferObject.relByteoffset, viewMatrix.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
//                    } break;
//                }
//            }
//        }       
//
//        commandBuffer->updateBuffer(gpuubo.get(), 0ull, gpuubo->getSize(), uboData.data());
//
//        for (size_t i = 0; i < gpumesh->getMeshBuffers().size(); ++i)
//        {
//            auto gpuMeshBuffer = gpumesh->getMeshBuffers().begin()[i];
//            auto gpuGraphicsPipeline = gpuPipelines[reinterpret_cast<RENDERPASS_INDEPENDENT_PIPELINE_ADRESS>(gpuMeshBuffer->getPipeline())];
//
//            const video::IGPURenderpassIndependentPipeline* gpuRenderpassIndependentPipeline = gpuMeshBuffer->getPipeline();
//            const video::IGPUDescriptorSet* ds3 = gpuMeshBuffer->getAttachedDescriptorSet();
//            
//            commandBuffer->bindGraphicsPipeline(gpuGraphicsPipeline.get());
//
//            const video::IGPUDescriptorSet* gpuds1_ptr = gpuds1.get();
//            commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuRenderpassIndependentPipeline->getLayout(), 1u, 1u, &gpuds1_ptr, nullptr);
//            const video::IGPUDescriptorSet* gpuds3_ptr = gpuMeshBuffer->getAttachedDescriptorSet();
//            if (gpuds3_ptr)
//                commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuRenderpassIndependentPipeline->getLayout(), 3u, 1u, &gpuds3_ptr, nullptr);
//            commandBuffer->pushConstants(gpuRenderpassIndependentPipeline->getLayout(), video::IGPUSpecializedShader::ESS_FRAGMENT, 0u, gpuMeshBuffer->MAX_PUSH_CONSTANT_BYTESIZE, gpuMeshBuffer->getPushConstantsDataPtr());
//
//            commandBuffer->drawMeshBuffer(gpuMeshBuffer);
//        }
//
//        commandBuffer->endRenderPass();
//        commandBuffer->end();
//
//        CommonAPI::Submit(logicalDevice.get(), swapchain.get(), commandBuffer.get(), queues[decltype(initOutput)::EQT_GRAPHICS], imageAcquire[resourceIx].get(), renderFinished[resourceIx].get(), fence.get());
//        CommonAPI::Present(logicalDevice.get(), swapchain.get(), queues[decltype(initOutput)::EQT_GRAPHICS], renderFinished[resourceIx].get(), acquiredNextFBO);
//    }
//
//    const auto& fboCreationParams = fbos[acquiredNextFBO]->getCreationParameters();
//    auto gpuSourceImageView = fboCreationParams.attachments[0];
//
//    bool status = ext::ScreenShot::createScreenShot(logicalDevice.get(), queues[decltype(initOutput)::EQT_TRANSFER_UP], renderFinished[resourceIx].get(), gpuSourceImageView.get(), assetManager.get(), "ScreenShot.png");
//    assert(status);
//
//    return 0;
//}