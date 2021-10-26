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
#include "nbl/ext/OIT/OIT.h"

using namespace nbl;
using namespace core;
/*
    Uncomment for more detailed logging
*/

#define LIMIT_TRANSPARENT_FRAGMENTS

// #define NBL_MORE_LOGS

class OITSampleApp : public ApplicationBase
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

    ext::OIT::COIT oit;
    core::vector<core::smart_refctd_ptr<video::IGPUMeshBuffer>> gpu_transMeshes;
    core::vector<core::smart_refctd_ptr<video::IGPUMeshBuffer>> gpu_opaqueMeshes;
    core::vector<core::smart_refctd_ptr<video::IGPUMeshBuffer>> gpu_allMeshes;
    core::smart_refctd_ptr<video::IGPUDescriptorSet> ds2;
    core::smart_refctd_ptr<video::IGPUGraphicsPipeline> oit_resolve_ppln;
    core::smart_refctd_ptr<video::IGPUGraphicsPipeline> oit_pass1_pipeline;


    uint32_t ds1UboBinding = 0;
    int resourceIx;
    uint32_t acquiredNextFBO = {};
    std::chrono::system_clock::time_point lastTime;
    bool frameDataFilled = false;
    size_t frameNum = 0ull;
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
    nbl::ui::IWindow* getWindow() override
    {
        return window.get();
    }
    void setSystem(core::smart_refctd_ptr<nbl::system::ISystem>&& system) override
    {
        system = std::move(system);
    }
    
    APP_CONSTRUCTOR(OITSampleApp)
    void onAppInitialized_impl() override
    {
        CommonAPI::InitOutput<SC_IMG_COUNT> initOutput;
        initOutput.window = core::smart_refctd_ptr(window);
        CommonAPI::Init<WIN_W, WIN_H, SC_IMG_COUNT>(initOutput, video::EAT_OPENGL, "OITSample", nbl::asset::EF_D32_SFLOAT);
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

        bool oit_init = oit.initialize(logicalDevice.get(), WIN_W, WIN_H, cpu2gpuParams);
        assert(oit_init);

        gpuTransferFence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));
        gpuComputeFence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

        core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout> cpu_ds2layout;
        {
            asset::ICPUDescriptorSetLayout::SBinding bnd[ext::OIT::COIT::BindingCount];
            oit.getDSLayoutBindings<asset::ICPUDescriptorSetLayout>(bnd);

            cpu_ds2layout = core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(bnd, bnd + ext::OIT::COIT::BindingCount);
        }
        core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> ds2layout;
        {
            video::IGPUObjectFromAssetConverter cpu2gpu;
            ds2layout = cpu2gpu.getGPUObjectsFromAssets(&cpu_ds2layout.get(), &cpu_ds2layout.get() + 1, cpu2gpuParams)->operator[](0);
        }

        auto ds2pool = logicalDevice->createDescriptorPoolForDSLayouts(static_cast<video::IDescriptorPool::E_CREATE_FLAGS>(0), &ds2layout.get(), &ds2layout.get() + 1);
        {
            ds2 = logicalDevice->createGPUDescriptorSet(ds2pool.get(), core::smart_refctd_ptr(ds2layout));

            video::IGPUDescriptorSet::SDescriptorInfo info[ext::OIT::COIT::BindingCount];
            video::IGPUDescriptorSet::SWriteDescriptorSet w[ext::OIT::COIT::BindingCount];

            oit.getDSWrites(w, info, ds2.get());

            logicalDevice->updateDescriptorSets(3u, w, 0u, nullptr);
        }

        {
            auto layout = logicalDevice->createGPUPipelineLayout(nullptr, nullptr, nullptr, nullptr, core::smart_refctd_ptr(ds2layout), nullptr);

            const auto& proto = oit.getResolveProtoPipeline();

            video::IGPUSpecializedShader* shaders[2]{ proto.vs.get(), proto.fs.get() };
            auto rpindep = logicalDevice->createGPURenderpassIndependentPipeline(nullptr, std::move(layout), shaders, shaders + 2, proto.vtx, proto.blend, proto.primAsm, proto.raster);

            video::IGPUGraphicsPipeline::SCreationParams pplnparams;
            pplnparams.renderpass = renderpass;
            pplnparams.renderpassIndependent = rpindep;
            pplnparams.subpassIx = 0u;
            oit_resolve_ppln = logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(pplnparams));
        }

        nbl::video::IGPUObjectFromAssetConverter cpu2gpu;
        {
            cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].fence = &gpuTransferFence;
            cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].fence = &gpuComputeFence;
        }

        {
            auto* quantNormalCache = assetManager->getMeshManipulator()->getQuantNormalCache();
            quantNormalCache->loadCacheFromFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), "../../tmp/normalCache101010.sse");

            system::path archPath = CWDOnStartup / "../../media/white_oak.zip";
            auto arch = system->openFileArchive(archPath);
            // test no alias loading (TODO: fix loading from absolute paths)
            system->mount(std::move(arch));
            asset::IAssetLoader::SAssetLoadParams loadParams;
            loadParams.workingDirectory = CWDOnStartup;
            loadParams.logger = logger.get();
            auto meshes_bundle = assetManager->getAsset((CWDOnStartup / "../../media/white_oak.zip/white_oak.obj").string(), loadParams);
            assert(!meshes_bundle.getContents().empty());

            metaOBJ = meshes_bundle.getMetadata()->selfCast<const asset::COBJMetadata>();

            auto cpuMesh = meshes_bundle.getContents().begin()[0];
            meshRaw = static_cast<asset::ICPUMesh*>(cpuMesh.get());

            quantNormalCache->saveCacheToFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), "../../tmp/normalCache101010.sse");
        }

        core::vector<core::smart_refctd_ptr<asset::ICPUMeshBuffer>> transparentMeshes;
        core::vector<core::smart_refctd_ptr<asset::ICPUMeshBuffer>> opaqueMeshes;
        asset::ICPURenderpassIndependentPipeline* transPpln;
        {
            for (auto& mb : meshRaw->getMeshBufferVector())
            {
                auto* ppln = mb->getPipeline();
                if (ppln->getBlendParams().blendParams[0].dstColorFactor != asset::EBF_ZERO)
                {
                    transparentMeshes.push_back(mb);
                    transPpln = ppln;
                }
                else
                {
                    opaqueMeshes.push_back(mb);
                }
            }
        }

        {
            auto srcppln = core::smart_refctd_ptr_static_cast<asset::ICPURenderpassIndependentPipeline>(transPpln->clone());

            auto* layout = srcppln->getLayout();
            layout->setDescriptorSetLayout(2u, core::smart_refctd_ptr(cpu_ds2layout));

            asset::IAssetLoader::SAssetLoadParams lparams;
            lparams.workingDirectory = CWDOnStartup;
            auto fs_bundle = assetManager->getAsset("../oit_fill_nodes.frag", lparams);
            assert(!fs_bundle.getContents().empty());
            auto fs = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(fs_bundle.getContents().begin()[0]);

            srcppln->setShaderAtStage(asset::ISpecializedShader::ESS_FRAGMENT, fs.get());

            asset::SVertexInputParams& vtxparams = srcppln->getVertexInputParams();
            asset::SPrimitiveAssemblyParams& primparams = srcppln->getPrimitiveAssemblyParams();
            asset::SBlendParams& blendparams = srcppln->getBlendParams();
            asset::SRasterizationParams& rasterparams = srcppln->getRasterizationParams();

            blendparams.blendParams[0].blendEnable = 0;
            blendparams.blendParams[0].colorWriteMask = 0;
            rasterparams.faceCullingMode = nbl::asset::EFCM_NONE;
            rasterparams.depthCompareOp = nbl::asset::ECO_GREATER;
            rasterparams.depthWriteEnable = 0;
            rasterparams.depthTestEnable = 1;

            core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> rpindep;
            {
                video::IGPUObjectFromAssetConverter c2g;
                auto gpu_bundle = c2g.getGPUObjectsFromAssets(&srcppln.get(), &srcppln.get() + 1, cpu2gpuParams);
                assert(gpu_bundle && gpu_bundle->size() && (*gpu_bundle)[0]);
                rpindep = (*gpu_bundle)[0];
            }

            video::IGPUGraphicsPipeline::SCreationParams pplnparams;
            pplnparams.renderpass = renderpass;
            pplnparams.renderpassIndependent = rpindep;
            pplnparams.subpassIx = 0u;
            oit_pass1_pipeline = logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(pplnparams));
        }
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

            //cpu2gpuWaitForFences();
            gpuds1layout = (*gpu_array)[0];
        }

        descriptorPool = createDescriptorPool(1u);

        ubomemreq = logicalDevice->getDeviceLocalGPUMemoryReqs();
        ubomemreq.vulkanReqs.size = neededDS1UBOsz;

        video::IGPUBuffer::SCreationParams gpuuboCreationParams;
        gpuuboCreationParams.usage = asset::IBuffer::EUF_UNIFORM_BUFFER_BIT;
        gpuuboCreationParams.sharingMode = asset::E_SHARING_MODE::ESM_CONCURRENT;
        gpuuboCreationParams.queueFamilyIndexCount = 0u;
        gpuuboCreationParams.queueFamilyIndices = nullptr;

        gpuubo = logicalDevice->createGPUBufferOnDedMem(gpuuboCreationParams, ubomemreq, true);
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
            auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(transparentMeshes.data(), transparentMeshes.data() + transparentMeshes.size(), cpu2gpuParams);
            if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
                assert(false);
            for (auto& mb : *gpu_array)
                gpu_transMeshes.push_back(mb);
            cpu2gpuWaitForFences();

            gpuTransferFence = nullptr;
            gpuComputeFence = nullptr;

            gpu_array = cpu2gpu.getGPUObjectsFromAssets(opaqueMeshes.data(), opaqueMeshes.data() + opaqueMeshes.size(), cpu2gpuParams);
            if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
                assert(false);
            for (auto& mb : *gpu_array)
                gpu_opaqueMeshes.push_back(mb);
            cpu2gpuWaitForFences();

            gpuTransferFence = nullptr;
            gpuComputeFence = nullptr;

            gpu_allMeshes.insert(gpu_allMeshes.begin(), gpu_transMeshes.begin(), gpu_transMeshes.end());
            gpu_allMeshes.insert(gpu_allMeshes.begin(), gpu_opaqueMeshes.begin(), gpu_opaqueMeshes.end());
        }

        {
            for (size_t i = 0; i < gpu_allMeshes.size(); ++i)
            {
                auto gpuIndependentPipeline = gpu_allMeshes[i]->getPipeline();

                nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams;
                graphicsPipelineParams.renderpassIndependent = core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline>(const_cast<video::IGPURenderpassIndependentPipeline*>(gpuIndependentPipeline));
                graphicsPipelineParams.renderpass = core::smart_refctd_ptr(renderpass);

                const RENDERPASS_INDEPENDENT_PIPELINE_ADRESS adress = reinterpret_cast<RENDERPASS_INDEPENDENT_PIPELINE_ADRESS>(graphicsPipelineParams.renderpassIndependent.get());
                if (gpuPipelines.find(adress) != gpuPipelines.end())
                    continue;
                gpuPipelines[adress] = logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(graphicsPipelineParams));
            }
        }

        core::vectorSIMDf cameraPosition(0, 5, -10);
        matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60), float(WIN_W) / WIN_H, 0.1, 1000);
        camera = Camera(cameraPosition, core::vectorSIMDf(0, 0, 0), projectionMatrix, 10.f, 1.f);
        lastTime = std::chrono::system_clock::now();

        frameDataFilled = false;
        frame_count = 0ull;
        time_sum = 0;
        dtList[NBL_FRAMES_TO_AVERAGE] = {};
        for (size_t i = 0ull; i < NBL_FRAMES_TO_AVERAGE; ++i)
            dtList[i] = 0.0;

        logicalDevice->createCommandBuffers(commandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, commandBuffers);

        for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
        {
            imageAcquire[i] = logicalDevice->createSemaphore();
            renderFinished[i] = logicalDevice->createSemaphore();
        }


    }
    void onAppTerminated_impl() override
    {
        const auto& fboCreationParams = fbo[acquiredNextFBO]->getCreationParameters();
        auto gpuSourceImageView = fboCreationParams.attachments[0];

        bool status = ext::ScreenShot::createScreenShot(logicalDevice.get(), 
            queues[CommonAPI::InitOutput<1>::EQT_TRANSFER_UP], 
            renderFinished[resourceIx].get(),
            gpuSourceImageView.get(),
            assetManager.get(),
            "ScreenShot.png");
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
            while (logicalDevice->waitForFences(1u, &fence.get(), false, MAX_TIMEOUT) == video::IGPUFence::ES_TIMEOUT) {}
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

        swapchain->acquireNextImage(MAX_TIMEOUT, imageAcquire[resourceIx].get(), nullptr, &acquiredNextFBO);

        // clear all pixels to 1.f in first frame,
        // all later frames will clear their pixels at the end of OIT resolve shader
        if (frameNum == 0u)
        {
            oit.invalidateNodesVisibility(commandBuffer.get());
        }

        nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
        {
            VkRect2D area;
            area.offset = { 0,0 };
            area.extent = { WIN_W, WIN_H };
            asset::SClearValue clear[2] = {};
            clear[0].color.float32[0] = 1.f;
            clear[0].color.float32[1] = 1.f;
            clear[0].color.float32[2] = 1.f;
            clear[0].color.float32[3] = 0.f;
            clear[1].depthStencil.depth = 0.f;
            clear[1].depthStencil.stencil = 0;

            beginInfo.clearValueCount = 2u;
            beginInfo.framebuffer = fbo[acquiredNextFBO];
            beginInfo.renderpass = renderpass;
            beginInfo.renderArea = area;
            beginInfo.clearValues = clear;
        }

        commandBuffer->beginRenderPass(&beginInfo, nbl::asset::ESC_INLINE);

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

        //draw opaque
        for (size_t i = 0; i < gpu_opaqueMeshes.size(); ++i)
        {
            auto gpuMeshBuffer = gpu_opaqueMeshes[i].get();
            auto gpuGraphicsPipeline = gpuPipelines[reinterpret_cast<RENDERPASS_INDEPENDENT_PIPELINE_ADRESS>(gpuMeshBuffer->getPipeline())];

            const video::IGPURenderpassIndependentPipeline* gpuRenderpassIndependentPipeline = gpuMeshBuffer->getPipeline();
            const video::IGPUDescriptorSet* ds3 = gpuMeshBuffer->getAttachedDescriptorSet();

            commandBuffer->bindGraphicsPipeline(gpuGraphicsPipeline.get());

            const video::IGPUDescriptorSet* gpuds1_ptr = gpuds1.get();
            commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuRenderpassIndependentPipeline->getLayout(), 1u, 1u, &gpuds1_ptr, nullptr);
            const video::IGPUDescriptorSet* gpuds3_ptr = gpuMeshBuffer->getAttachedDescriptorSet();
            if (gpuds3_ptr)
                commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuRenderpassIndependentPipeline->getLayout(), 3u, 1u, &gpuds3_ptr, nullptr);
            commandBuffer->pushConstants(gpuRenderpassIndependentPipeline->getLayout(), video::IGPUSpecializedShader::ESS_FRAGMENT, 0u, gpuMeshBuffer->MAX_PUSH_CONSTANT_BYTESIZE, gpuMeshBuffer->getPushConstantsDataPtr());

            commandBuffer->drawMeshBuffer(gpuMeshBuffer);
        }

        // OIT 1st pass
        for (size_t i = 0; i < gpu_transMeshes.size(); ++i)
        {
            auto gpuMeshBuffer = gpu_transMeshes[i].get();

            const video::IGPURenderpassIndependentPipeline* gpuRenderpassIndependentPipeline = gpuMeshBuffer->getPipeline();
            const video::IGPUDescriptorSet* ds3 = gpuMeshBuffer->getAttachedDescriptorSet();
            const video::IGPUDescriptorSet* gpuds2_ptr = ds2.get();
            const video::IGPUDescriptorSet* gpuds3_ptr = gpuMeshBuffer->getAttachedDescriptorSet();

            commandBuffer->bindGraphicsPipeline(oit_pass1_pipeline.get());
            const video::IGPUDescriptorSet* gpuds1_ptr = gpuds1.get();
            commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, oit_pass1_pipeline->getRenderpassIndependentPipeline()->getLayout(), 1u, 1u, &gpuds1_ptr, nullptr);
            commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, oit_pass1_pipeline->getRenderpassIndependentPipeline()->getLayout(), 2u, 1u, &ds2.get());
            if (gpuds3_ptr)
                commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, oit_pass1_pipeline->getRenderpassIndependentPipeline()->getLayout(), 3u, 1u, &gpuds3_ptr, nullptr);

            commandBuffer->drawMeshBuffer(gpuMeshBuffer);
        }

        // mem barriers on OIT images
        oit.barrierBetweenPasses(commandBuffer.get(), queues[CommonAPI::InitOutput<1>::EQT_GRAPHICS]->getFamilyIndex());

        //OIT resolve
        oit.resolvePass(commandBuffer.get(), oit_resolve_ppln.get(), ds2.get(), 2u);

        // mem barrier on OIT vis image (written at the end of resolve pass)
        oit.barrierAfterResolve(commandBuffer.get(), queues[CommonAPI::InitOutput<1>::EQT_GRAPHICS]->getFamilyIndex());

        commandBuffer->endRenderPass();
        commandBuffer->end();

        CommonAPI::Submit(logicalDevice.get(), 
            swapchain.get(),
            commandBuffer.get(),
            queues[CommonAPI::InitOutput<1>::EQT_GRAPHICS],
            imageAcquire[resourceIx].get(), 
            renderFinished[resourceIx].get(),
            fence.get());
        CommonAPI::Present(logicalDevice.get(),
            swapchain.get(), 
            queues[CommonAPI::InitOutput<1>::EQT_GRAPHICS], 
            renderFinished[resourceIx].get(), 
            acquiredNextFBO);

        frameNum++;
    }
    bool keepRunning() override
    {
        return windowCb->isWindowOpen();
    }
};

NBL_COMMON_API_MAIN(OITSampleApp, OITSampleApp::nabla)
//
//int main(int argc, char** argv)
//{
//    system::path CWD = system::path(argv[0]).parent_path().generic_string() + "/";
//    constexpr uint32_t WIN_W = 1280;
//    constexpr uint32_t WIN_H = 720;
//    constexpr uint32_t SC_IMG_COUNT = 3u;
//    constexpr uint32_t FRAMES_IN_FLIGHT = 5u;
//    static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);
//
//    CommonAPI::InitOutput<SC_IMG_COUNT> initOutput; 
//    CommonAPI::Init<WIN_W, WIN_H, SC_IMG_COUNT>(initOutput, video::EAT_OPENGL, "OIT", nbl::asset::EF_D32_SFLOAT);
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
//    ext::OIT::COIT oit;
//    bool oit_init = oit.initialize(logicalDevice.get(), WIN_W, WIN_H, cpu2gpuParams);
//    assert(oit_init);
//
//    auto gpuTransferFence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));
//    auto gpuComputeFence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));
//
//    core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout> cpu_ds2layout;
//    {
//        asset::ICPUDescriptorSetLayout::SBinding bnd[ext::OIT::COIT::BindingCount];
//        oit.getDSLayoutBindings<asset::ICPUDescriptorSetLayout>(bnd);
//
//        cpu_ds2layout = core::make_smart_refctd_ptr<asset::ICPUDescriptorSetLayout>(bnd, bnd+ext::OIT::COIT::BindingCount);
//    }
//    core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> ds2layout;
//    {
//        video::IGPUObjectFromAssetConverter cpu2gpu;
//        ds2layout = cpu2gpu.getGPUObjectsFromAssets(&cpu_ds2layout.get(), &cpu_ds2layout.get() + 1, cpu2gpuParams)->operator[](0);
//    }
//
//    auto ds2pool = logicalDevice->createDescriptorPoolForDSLayouts(static_cast<video::IDescriptorPool::E_CREATE_FLAGS>(0), &ds2layout.get(), &ds2layout.get() + 1);
//    core::smart_refctd_ptr<video::IGPUDescriptorSet> ds2;
//    {
//        ds2 = logicalDevice->createGPUDescriptorSet(ds2pool.get(), core::smart_refctd_ptr(ds2layout));
//
//        video::IGPUDescriptorSet::SDescriptorInfo info[ext::OIT::COIT::BindingCount];
//        video::IGPUDescriptorSet::SWriteDescriptorSet w[ext::OIT::COIT::BindingCount];
//        
//        oit.getDSWrites(w, info, ds2.get());
//
//        logicalDevice->updateDescriptorSets(3u, w, 0u, nullptr);
//    }
//
//    core::smart_refctd_ptr<video::IGPUGraphicsPipeline> oit_resolve_ppln;
//    {
//        auto layout = logicalDevice->createGPUPipelineLayout(nullptr, nullptr, nullptr, nullptr, core::smart_refctd_ptr(ds2layout), nullptr);
//
//        const auto& proto = oit.getResolveProtoPipeline();
//
//        video::IGPUSpecializedShader* shaders[2]{ proto.vs.get(), proto.fs.get() };
//        auto rpindep = logicalDevice->createGPURenderpassIndependentPipeline(nullptr, std::move(layout), shaders, shaders + 2, proto.vtx, proto.blend, proto.primAsm, proto.raster);
//
//        video::IGPUGraphicsPipeline::SCreationParams pplnparams;
//        pplnparams.renderpass = renderpass;
//        pplnparams.renderpassIndependent = rpindep;
//        pplnparams.subpassIx = 0u;
//        oit_resolve_ppln = logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(pplnparams));
//    }
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
//        system::path archPath = CWD/"../../media/white_oak.zip";
//        auto arch = system->openFileArchive(archPath);
//        // test no alias loading (TODO: fix loading from absolute paths)
//        system->mount(std::move(arch));
//        asset::IAssetLoader::SAssetLoadParams loadParams;
//        loadParams.workingDirectory = CWD;
//        loadParams.logger = logger.get();
//        auto meshes_bundle = assetManager->getAsset((CWD/"../../media/white_oak.zip/white_oak.obj").string(), loadParams);
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
//    core::vector<core::smart_refctd_ptr<asset::ICPUMeshBuffer>> transparentMeshes;
//    core::vector<core::smart_refctd_ptr<asset::ICPUMeshBuffer>> opaqueMeshes;
//    asset::ICPURenderpassIndependentPipeline* transPpln;
//    {
//        for (auto& mb : meshRaw->getMeshBufferVector())
//        {
//            auto* ppln = mb->getPipeline();
//            if (ppln->getBlendParams().blendParams[0].dstColorFactor != asset::EBF_ZERO)
//            {
//                transparentMeshes.push_back(mb);
//                transPpln = ppln;
//            }
//            else
//            {
//                opaqueMeshes.push_back(mb);
//            }
//        }
//    }
//
//    core::smart_refctd_ptr<video::IGPUGraphicsPipeline> oit_pass1_pipeline;
//    {
//        auto srcppln = core::smart_refctd_ptr_static_cast<asset::ICPURenderpassIndependentPipeline>(transPpln->clone());
//
//        auto* layout = srcppln->getLayout();
//        layout->setDescriptorSetLayout(2u, core::smart_refctd_ptr(cpu_ds2layout));
//
//        asset::IAssetLoader::SAssetLoadParams lparams;
//        lparams.workingDirectory = CWD;
//        auto fs_bundle = assetManager->getAsset("../oit_fill_nodes.frag",lparams);
//        assert(!fs_bundle.getContents().empty());
//        auto fs = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(fs_bundle.getContents().begin()[0]);
//
//        srcppln->setShaderAtStage(asset::ISpecializedShader::ESS_FRAGMENT, fs.get());
//
//        asset::SVertexInputParams& vtxparams = srcppln->getVertexInputParams();
//        asset::SPrimitiveAssemblyParams& primparams = srcppln->getPrimitiveAssemblyParams();
//        asset::SBlendParams& blendparams = srcppln->getBlendParams();
//        asset::SRasterizationParams& rasterparams = srcppln->getRasterizationParams();
//
//        blendparams.blendParams[0].blendEnable = 0;
//        blendparams.blendParams[0].colorWriteMask = 0;
//        rasterparams.faceCullingMode = nbl::asset::EFCM_NONE;
//        rasterparams.depthCompareOp = nbl::asset::ECO_GREATER;
//        rasterparams.depthWriteEnable = 0;
//        rasterparams.depthTestEnable = 1;
//
//        core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> rpindep;
//        {
//            video::IGPUObjectFromAssetConverter c2g;
//            auto gpu_bundle = c2g.getGPUObjectsFromAssets(&srcppln.get(),&srcppln.get()+1, cpu2gpuParams);
//            assert(gpu_bundle&& gpu_bundle->size() && (*gpu_bundle)[0]);
//            rpindep = (*gpu_bundle)[0];
//        }
//
//        video::IGPUGraphicsPipeline::SCreationParams pplnparams;
//        pplnparams.renderpass = renderpass;
//        pplnparams.renderpassIndependent = rpindep;
//        pplnparams.subpassIx = 0u;
//        oit_pass1_pipeline = logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(pplnparams));
//    }
//
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
//
//    video::IGPUBuffer::SCreationParams gpuuboCreationParams;
//    gpuuboCreationParams.usage = asset::IBuffer::EUF_UNIFORM_BUFFER_BIT;
//    gpuuboCreationParams.sharingMode = asset::E_SHARING_MODE::ESM_CONCURRENT;
//    gpuuboCreationParams.queueFamilyIndexCount = 0u;
//    gpuuboCreationParams.queueFamilyIndices = nullptr;
//
//    auto gpuubo = logicalDevice->createGPUBufferOnDedMem(gpuuboCreationParams,ubomemreq,true);
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
//    core::vector<core::smart_refctd_ptr<video::IGPUMeshBuffer>> gpu_transMeshes;
//    core::vector<core::smart_refctd_ptr<video::IGPUMeshBuffer>> gpu_opaqueMeshes;
//    core::vector<core::smart_refctd_ptr<video::IGPUMeshBuffer>> gpu_allMeshes;
//    {
//        auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(transparentMeshes.data(), transparentMeshes.data()+transparentMeshes.size(), cpu2gpuParams);
//        if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
//            assert(false);
//        for (auto& mb : *gpu_array)
//            gpu_transMeshes.push_back(mb);
//        cpu2gpuWaitForFences();
//
//        gpuTransferFence = nullptr;
//        gpuComputeFence = nullptr;
//        
//        gpu_array = cpu2gpu.getGPUObjectsFromAssets(opaqueMeshes.data(), opaqueMeshes.data() + opaqueMeshes.size(), cpu2gpuParams);
//        if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
//            assert(false);
//        for (auto& mb : *gpu_array)
//            gpu_opaqueMeshes.push_back(mb);
//        cpu2gpuWaitForFences();
//
//        gpuTransferFence = nullptr;
//        gpuComputeFence = nullptr;
//
//        gpu_allMeshes.insert(gpu_allMeshes.begin(), gpu_transMeshes.begin(), gpu_transMeshes.end());
//        gpu_allMeshes.insert(gpu_allMeshes.begin(), gpu_opaqueMeshes.begin(), gpu_opaqueMeshes.end());
//    }
//
//    using RENDERPASS_INDEPENDENT_PIPELINE_ADRESS = size_t;
//    std::map<RENDERPASS_INDEPENDENT_PIPELINE_ADRESS, core::smart_refctd_ptr<video::IGPUGraphicsPipeline>> gpuPipelines;
//    {
//        for (size_t i = 0; i < gpu_allMeshes.size(); ++i)
//        {
//            auto gpuIndependentPipeline = gpu_allMeshes[i]->getPipeline();
//
//            nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams;
//            graphicsPipelineParams.renderpassIndependent = core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline>(const_cast<video::IGPURenderpassIndependentPipeline*>(gpuIndependentPipeline));
//            graphicsPipelineParams.renderpass = core::smart_refctd_ptr(renderpass);
//
//            const RENDERPASS_INDEPENDENT_PIPELINE_ADRESS adress = reinterpret_cast<RENDERPASS_INDEPENDENT_PIPELINE_ADRESS>(graphicsPipelineParams.renderpassIndependent.get());
//            if (gpuPipelines.find(adress) != gpuPipelines.end())
//                continue;
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
//    uint32_t frameNum = 0u;
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
//        // clear all pixels to 1.f in first frame,
//        // all later frames will clear their pixels at the end of OIT resolve shader
//        if (frameNum == 0u)
//        {
//            oit.invalidateNodesVisibility(commandBuffer.get());
//        }
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
//            clear[0].color.float32[3] = 0.f;
//            clear[1].depthStencil.depth = 0.f;
//            clear[1].depthStencil.stencil = 0;
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
//        //draw opaque
//        for (size_t i = 0; i < gpu_opaqueMeshes.size(); ++i)
//        {
//            auto gpuMeshBuffer = gpu_opaqueMeshes[i].get();
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
//        // OIT 1st pass
//        for (size_t i = 0; i < gpu_transMeshes.size(); ++i)
//        {
//            auto gpuMeshBuffer = gpu_transMeshes[i].get();
//
//            const video::IGPURenderpassIndependentPipeline* gpuRenderpassIndependentPipeline = gpuMeshBuffer->getPipeline();
//            const video::IGPUDescriptorSet* ds3 = gpuMeshBuffer->getAttachedDescriptorSet();
//            const video::IGPUDescriptorSet* gpuds2_ptr = ds2.get();
//            const video::IGPUDescriptorSet* gpuds3_ptr = gpuMeshBuffer->getAttachedDescriptorSet();
//
//            commandBuffer->bindGraphicsPipeline(oit_pass1_pipeline.get());
//            const video::IGPUDescriptorSet* gpuds1_ptr = gpuds1.get();
//            commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, oit_pass1_pipeline->getRenderpassIndependentPipeline()->getLayout(), 1u, 1u, &gpuds1_ptr, nullptr);
//            commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, oit_pass1_pipeline->getRenderpassIndependentPipeline()->getLayout(), 2u, 1u, &ds2.get());
//            if (gpuds3_ptr)
//                commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, oit_pass1_pipeline->getRenderpassIndependentPipeline()->getLayout(), 3u, 1u, &gpuds3_ptr, nullptr);
//
//            commandBuffer->drawMeshBuffer(gpuMeshBuffer);
//        }
//
//        // mem barriers on OIT images
//        oit.barrierBetweenPasses(commandBuffer.get(), queues[decltype(initOutput)::EQT_GRAPHICS]->getFamilyIndex());
//
//        //OIT resolve
//        oit.resolvePass(commandBuffer.get(), oit_resolve_ppln.get(), ds2.get(), 2u);
//
//        // mem barrier on OIT vis image (written at the end of resolve pass)
//        oit.barrierAfterResolve(commandBuffer.get(), queues[decltype(initOutput)::EQT_GRAPHICS]->getFamilyIndex());
//
//        commandBuffer->endRenderPass();
//        commandBuffer->end();
//
//        CommonAPI::Submit(logicalDevice.get(), swapchain.get(), commandBuffer.get(), queues[decltype(initOutput)::EQT_GRAPHICS], imageAcquire[resourceIx].get(), renderFinished[resourceIx].get(), fence.get());
//        CommonAPI::Present(logicalDevice.get(), swapchain.get(), queues[decltype(initOutput)::EQT_GRAPHICS], renderFinished[resourceIx].get(), acquiredNextFBO);
//
//        frameNum++;
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