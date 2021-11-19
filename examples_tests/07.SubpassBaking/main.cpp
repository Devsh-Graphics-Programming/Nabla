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


class SubpassBakingApp : public ApplicationBase
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
    uint32_t cameraUBOBinding = 0u;
    core::smart_refctd_ptr<video::IGPUBuffer> cameraUBO;
    core::smart_refctd_ptr<video::IGPUDescriptorSet> perCameraDescSet;
    core::smart_refctd_ptr<video::IGPUCommandBuffer> bakedCommandBuffer;
    video::CDumbPresentationOracle oracle;


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
    APP_CONSTRUCTOR(SubpassBakingApp)
    void onAppInitialized_impl() override
    {
        CommonAPI::InitOutput<SC_IMG_COUNT> initOutput;
        initOutput.window = core::smart_refctd_ptr(window);
        initOutput.system = core::smart_refctd_ptr(system);
        CommonAPI::Init<WIN_W, WIN_H, SC_IMG_COUNT>(initOutput, video::EAT_OPENGL, "MeshLoaders", nbl::asset::EF_D32_SFLOAT);
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
        commandPool = std::move(initOutput.commandPool);
        system = std::move(initOutput.system);
        assetManager = std::move(initOutput.assetManager);
        cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
        logger = std::move(initOutput.logger);
        inputSystem = std::move(initOutput.inputSystem);


        {
            cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].fence = &gpuTransferFence;
            cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].fence = &gpuComputeFence;
        }

        asset::ICPUMesh* meshRaw = nullptr;
        const asset::COBJMetadata* metaOBJ = nullptr;
        {
            auto* quantNormalCache = assetManager->getMeshManipulator()->getQuantNormalCache();
            quantNormalCache->loadCacheFromFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), sharedOutputCWD / "normalCache101010.sse");

            system::path archPath = sharedInputCWD / "sponza.zip";
            auto arch = system->openFileArchive(archPath);
            system->mount(std::move(arch), "resources");
            asset::IAssetLoader::SAssetLoadParams loadParams;
#if 0 // @sadiuk unfuck this please
            loadParams.workingDirectory = "resources";
#else
            loadParams.workingDirectory = archPath;
#endif
            loadParams.logger = logger.get();
            auto meshes_bundle = assetManager->getAsset("sponza.obj", loadParams);
            assert(!meshes_bundle.getContents().empty());

            metaOBJ = meshes_bundle.getMetadata()->selfCast<const asset::COBJMetadata>();

            auto cpuMesh = meshes_bundle.getContents().begin()[0];
            meshRaw = static_cast<asset::ICPUMesh*>(cpuMesh.get());

            quantNormalCache->saveCacheToFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), sharedOutputCWD / "normalCache101010.sse");
        }


        {
            const auto meshbuffers = meshRaw->getMeshBuffers();

            core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> gpuds1layout;
            size_t neededDS1UBOsz = 0ull;
            {
                // we can safely assume that all meshbuffers within mesh loaded from OBJ has same DS1 layout (used for camera-specific data)
                const asset::ICPUMeshBuffer* const firstMeshBuffer = *meshbuffers.begin();
                pipelineMetadata = metaOBJ->getAssetSpecificMetadata(firstMeshBuffer->getPipeline());

                // so we can create just one DS
                const asset::ICPUDescriptorSetLayout* ds1layout = firstMeshBuffer->getPipeline()->getLayout()->getDescriptorSetLayout(1u);
                for (const auto& bnd : ds1layout->getBindings())
                    if (bnd.type == asset::EDT_UNIFORM_BUFFER)
                    {
                        cameraUBOBinding = bnd.binding;
                        break;
                    }

                for (const auto& shdrIn : pipelineMetadata->m_inputSemantics)
                    if (shdrIn.descriptorSection.type == asset::IRenderpassIndependentPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set == 1u && shdrIn.descriptorSection.uniformBufferObject.binding == cameraUBOBinding)
                        neededDS1UBOsz = std::max<size_t>(neededDS1UBOsz, shdrIn.descriptorSection.uniformBufferObject.relByteoffset + shdrIn.descriptorSection.uniformBufferObject.bytesize);

                auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&ds1layout, &ds1layout + 1, cpu2gpuParams);
                assert(gpu_array && gpu_array->size() && (*gpu_array)[0]);
                gpuds1layout = (*gpu_array)[0];
                cpu2gpuParams.waitForCreationToComplete();
            }

            descriptorPool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, &gpuds1layout.get(), &gpuds1layout.get() + 1);

            ubomemreq = logicalDevice->getDeviceLocalGPUMemoryReqs();
            ubomemreq.vulkanReqs.size = neededDS1UBOsz;

            video::IGPUBuffer::SCreationParams cameraUBOCreationParams;
            cameraUBOCreationParams.usage = asset::IBuffer::EUF_UNIFORM_BUFFER_BIT;
            cameraUBOCreationParams.canUpdateSubRange = true;
            cameraUBOCreationParams.sharingMode = asset::E_SHARING_MODE::ESM_EXCLUSIVE;
            cameraUBOCreationParams.queueFamilyIndexCount = 0u;
            cameraUBOCreationParams.queueFamilyIndices = nullptr;

            cameraUBO = logicalDevice->createGPUBufferOnDedMem(cameraUBOCreationParams, ubomemreq);
            perCameraDescSet = logicalDevice->createGPUDescriptorSet(descriptorPool.get(), std::move(gpuds1layout));
            {
                video::IGPUDescriptorSet::SWriteDescriptorSet write;
                write.dstSet = perCameraDescSet.get();
                write.binding = cameraUBOBinding;
                write.count = 1u;
                write.arrayElement = 0u;
                write.descriptorType = asset::EDT_UNIFORM_BUFFER;
                video::IGPUDescriptorSet::SDescriptorInfo info;
                {
                    info.desc = cameraUBO;
                    info.buffer.offset = 0ull;
                    info.buffer.size = neededDS1UBOsz;
                }
                write.info = &info;
                logicalDevice->updateDescriptorSets(1u, &write, 0u, nullptr);
            }
        }

        {
            auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&meshRaw, &meshRaw + 1, cpu2gpuParams);
            assert(gpu_array && gpu_array->size() && (*gpu_array)[0]);
            gpumesh = (*gpu_array)[0];
            cpu2gpuParams.waitForCreationToComplete();
        }

        logicalDevice->createCommandBuffers(commandPool.get(), video::IGPUCommandBuffer::EL_SECONDARY, 1u, &bakedCommandBuffer);
        bakedCommandBuffer->begin(video::IGPUCommandBuffer::EU_RENDER_PASS_CONTINUE_BIT | video::IGPUCommandBuffer::EU_SIMULTANEOUS_USE_BIT);
        //#define REFERENCE
        {
            const uint32_t drawCallCount = gpumesh->getMeshBuffers().size();
            core::smart_refctd_ptr<video::CDrawIndirectAllocator<>> drawAllocator;
            {
                video::IDrawIndirectAllocator::ImplicitBufferCreationParameters params;
                params.device = logicalDevice.get();
                params.maxDrawCommandStride = sizeof(asset::DrawElementsIndirectCommand_t);
                params.drawCommandCapacity = drawCallCount;
                params.drawCountCapacity = 0u;
                drawAllocator = video::CDrawIndirectAllocator<>::create(std::move(params));
            }
            video::IDrawIndirectAllocator::Allocation allocation;
            {
                allocation.count = drawCallCount;
                {
                    allocation.multiDrawCommandRangeByteOffsets = new uint32_t[allocation.count];
                    // you absolutely must do this
                    std::fill_n(allocation.multiDrawCommandRangeByteOffsets, allocation.count, video::IDrawIndirectAllocator::invalid_draw_range_begin);
                }
                {
                    auto drawCounts = new uint32_t[allocation.count];
                    std::fill_n(drawCounts, allocation.count, 1u);
                    allocation.multiDrawCommandMaxCounts = drawCounts;
                }
                allocation.setAllCommandStructSizesConstant(sizeof(asset::DrawElementsIndirectCommand_t));
                drawAllocator->allocateMultiDraws(allocation);
                delete[] allocation.multiDrawCommandMaxCounts;
            }

            video::CSubpassKiln kiln;
            constexpr auto kSubpassIx = 0u;

            auto drawCallData = new asset::DrawElementsIndirectCommand_t[drawCallCount];
            {
                auto drawIndexIt = allocation.multiDrawCommandRangeByteOffsets;
                auto drawCallDataIt = drawCallData;
                core::map<const void*, core::smart_refctd_ptr<video::IGPUGraphicsPipeline>> graphicsPipelines;
                for (auto& mb : gpumesh->getMeshBuffers())
                {
                    auto& drawcall = kiln.getDrawcallMetadataVector().emplace_back();
                    memcpy(drawcall.pushConstantData, mb->getPushConstantsDataPtr(), video::IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE);
                    {
                        auto renderpassIndep = mb->getPipeline();
                        auto foundPpln = graphicsPipelines.find(renderpassIndep);
                        if (foundPpln == graphicsPipelines.end())
                        {
                            video::IGPUGraphicsPipeline::SCreationParams params;
                            params.renderpassIndependent = core::smart_refctd_ptr<const video::IGPURenderpassIndependentPipeline>(renderpassIndep);
                            params.renderpass = core::smart_refctd_ptr(renderpass);
                            params.subpassIx = kSubpassIx;
                            foundPpln = graphicsPipelines.emplace_hint(foundPpln, renderpassIndep, logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(params)));
                        }
                        drawcall.pipeline = foundPpln->second;
                    }
                    drawcall.descriptorSets[1] = perCameraDescSet;
                    drawcall.descriptorSets[3] = core::smart_refctd_ptr<const video::IGPUDescriptorSet>(mb->getAttachedDescriptorSet());
                    std::copy_n(mb->getVertexBufferBindings(), video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT, drawcall.vertexBufferBindings);
                    drawcall.indexBufferBinding = mb->getIndexBufferBinding().buffer;
                    drawcall.drawCommandStride = sizeof(asset::DrawElementsIndirectCommand_t);
                    drawcall.indexType = mb->getIndexType();
                    //drawcall.drawCountOffset // leave as invalid
                    drawcall.drawCallOffset = *(drawIndexIt++);
                    drawcall.drawMaxCount = 1u;

                    // TODO: in the far future, just make IMeshBuffer hold a union of `DrawArraysIndirectCommand_t` `DrawElementsIndirectCommand_t`
                    drawCallDataIt->count = mb->getIndexCount();
                    drawCallDataIt->instanceCount = mb->getInstanceCount();
                    switch (drawcall.indexType)
                    {
                    case asset::EIT_32BIT:
                        drawCallDataIt->firstIndex = mb->getIndexBufferBinding().offset / sizeof(uint32_t);
                        break;
                    case asset::EIT_16BIT:
                        drawCallDataIt->firstIndex = mb->getIndexBufferBinding().offset / sizeof(uint16_t);
                        break;
                    default:
                        assert(false);
                        break;
                }
                    drawCallDataIt->baseVertex = mb->getBaseVertex();
                    drawCallDataIt->baseInstance = mb->getBaseInstance();
                    drawCallDataIt++;
#ifdef REFERENCE
                    bakedCommandBuffer->bindGraphicsPipeline(drawcall.pipeline.get());
                    bakedCommandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, drawcall.pipeline->getRenderpassIndependentPipeline()->getLayout(), 1u, 3u, &drawcall.descriptorSets->get() + 1u, nullptr);
                    bakedCommandBuffer->pushConstants(drawcall.pipeline->getRenderpassIndependentPipeline()->getLayout(), video::IGPUSpecializedShader::ESS_FRAGMENT, 0u, video::IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE, drawcall.pushConstantData);
                    bakedCommandBuffer->drawMeshBuffer(mb);
#endif
            }
        }
            // do the transfer of drawcall structs
            {
                video::CPropertyPoolHandler::UpStreamingRequest request;
                request.destination = drawAllocator->getDrawCommandMemoryBlock();
                request.fill = false;
                request.elementSize = sizeof(asset::DrawElementsIndirectCommand_t);
                request.elementCount = drawCallCount;
                request.source.device2device = false;
                request.source.data = drawCallData;
                request.srcAddresses = nullptr; // iota 0,1,2,3,4,etc.
                request.dstAddresses = allocation.multiDrawCommandRangeByteOffsets;
                std::for_each_n(allocation.multiDrawCommandRangeByteOffsets, request.elementCount, [&](auto& handle) {handle /= request.elementSize; });

                auto upQueue = queues[decltype(initOutput)::EQT_TRANSFER_UP];
                auto fence = logicalDevice->createFence(video::IGPUFence::ECF_UNSIGNALED);
                core::smart_refctd_ptr<video::IGPUCommandBuffer> tferCmdBuf;
                logicalDevice->createCommandBuffers(commandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &tferCmdBuf);
                tferCmdBuf->begin(0u); // TODO some one time submit bit or something
                {
                    auto* ppHandler = utilities->getDefaultPropertyPoolHandler();
                    // if we did multiple transfers, we'd reuse the scratch
                    asset::SBufferBinding<video::IGPUBuffer> scratch;
                    {
                        video::IGPUBuffer::SCreationParams scratchParams = {};
                        scratchParams.canUpdateSubRange = true;
                        scratchParams.usage = core::bitflag(video::IGPUBuffer::EUF_TRANSFER_DST_BIT) | video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
                        scratch = { 0ull,logicalDevice->createDeviceLocalGPUBufferOnDedMem(scratchParams,ppHandler->getMaxScratchSize()) };
                        scratch.buffer->setObjectDebugName("Scratch Buffer");
                    }
                    auto* pRequest = &request;
                    uint32_t waitSemaphoreCount = 0u;
                    video::IGPUSemaphore* const* waitSemaphores = nullptr;
                    const asset::E_PIPELINE_STAGE_FLAGS* waitStages = nullptr;
                    ppHandler->transferProperties(
                        utilities->getDefaultUpStreamingBuffer(), tferCmdBuf.get(), fence.get(), upQueue,
                        scratch, pRequest, 1u, waitSemaphoreCount, waitSemaphores, waitStages,
                        logger.get(), std::chrono::high_resolution_clock::time_point::max() // wait forever if necessary, need initialization to finish
                    );
                }
                tferCmdBuf->end();
                {
                    video::IGPUQueue::SSubmitInfo submit = {}; // intializes all semaphore stuff to 0 and nullptr
                    submit.commandBufferCount = 1u;
                    submit.commandBuffers = &tferCmdBuf.get();
                    upQueue->submit(1u, &submit, fence.get());
                }
                logicalDevice->blockForFences(1u, &fence.get());
            }
            delete[] drawCallData;
            // free the draw command index list
            delete[] allocation.multiDrawCommandRangeByteOffsets;

#ifndef REFERENCE
            kiln.bake(bakedCommandBuffer.get(), renderpass.get(), kSubpassIx, drawAllocator->getDrawCommandMemoryBlock().buffer.get(), nullptr);
#endif
    }
        bakedCommandBuffer->end();

        core::vectorSIMDf cameraPosition(0, 5, -10);
        matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60), float(WIN_W) / WIN_H, 2.f, 4000.f);
        camera = Camera(cameraPosition, core::vectorSIMDf(0, 0, 0), projectionMatrix, 10.f, 1.f);

        oracle.reportBeginFrameRecord();

        logicalDevice->createCommandBuffers(commandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, commandBuffers);

        for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
        {
            imageAcquire[i] = logicalDevice->createSemaphore();
            renderFinished[i] = logicalDevice->createSemaphore();
        }
        resourceIx = -1;
    }
    void onAppTerminated_impl() override
    {
        const auto& fboCreationParams = fbo[acquiredNextFBO]->getCreationParameters();
        auto gpuSourceImageView = fboCreationParams.attachments[0];

        bool status = ext::ScreenShot::createScreenShot(logicalDevice.get(), queues[CommonAPI::InitOutput<1>::EQT_TRANSFER_DOWN], renderFinished[resourceIx].get(), gpuSourceImageView.get(), assetManager.get(), "ScreenShot.png");
        assert(status);
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

        //
        commandBuffer->reset(nbl::video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
        commandBuffer->begin(0);

        // late latch input
        const auto nextPresentationTimestamp = oracle.acquireNextImage(swapchain.get(), imageAcquire[resourceIx].get(), nullptr, &acquiredNextFBO);

        // input
        {
            inputSystem->getDefaultMouse(&mouse);
            inputSystem->getDefaultKeyboard(&keyboard);

            camera.beginInputProcessing(nextPresentationTimestamp);
            mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); }, logger.get());
            keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { camera.keyboardProcess(events); }, logger.get());
            camera.endInputProcessing(nextPresentationTimestamp);
        }

        // update camera
        {
            const auto& viewMatrix = camera.getViewMatrix();
            const auto& viewProjectionMatrix = camera.getConcatenatedMatrix();

            core::vector<uint8_t> uboData(cameraUBO->getSize());
            for (const auto& shdrIn : pipelineMetadata->m_inputSemantics)
            {
                if (shdrIn.descriptorSection.type == asset::IRenderpassIndependentPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set == 1u && shdrIn.descriptorSection.uniformBufferObject.binding == cameraUBOBinding)
                {
                    switch (shdrIn.type)
                    {
                    case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW_PROJ:
                    {
                        memcpy(uboData.data() + shdrIn.descriptorSection.uniformBufferObject.relByteoffset, viewProjectionMatrix.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
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
            commandBuffer->updateBuffer(cameraUBO.get(), 0ull, cameraUBO->getSize(), uboData.data());
        }

        // renderpass
        {
            asset::SViewport viewport;
            viewport.minDepth = 1.f;
            viewport.maxDepth = 0.f;
            viewport.x = 0u;
            viewport.y = 0u;
            viewport.width = WIN_W;
            viewport.height = WIN_H;
            commandBuffer->setViewport(0u, 1u, &viewport);

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
            commandBuffer->executeCommands(1u, &bakedCommandBuffer.get());
            commandBuffer->endRenderPass();

            commandBuffer->end();
        }

        CommonAPI::Submit(logicalDevice.get(), swapchain.get(), commandBuffer.get(), queues[CommonAPI::InitOutput<3>::EQT_GRAPHICS], imageAcquire[resourceIx].get(), renderFinished[resourceIx].get(), fence.get());
        CommonAPI::Present(logicalDevice.get(), swapchain.get(), queues[CommonAPI::InitOutput<3>::EQT_GRAPHICS], renderFinished[resourceIx].get(), acquiredNextFBO);
    }

    bool keepRunning() override
    {
        return windowCb->isWindowOpen();
    }
};

NBL_COMMON_API_MAIN(SubpassBakingApp)

