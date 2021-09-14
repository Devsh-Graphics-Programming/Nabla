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

int main(int argc, char** argv)
{
    system::path CWD = system::path(argv[0]).parent_path().generic_string() + "/";
    constexpr uint32_t WIN_W = 1280;
    constexpr uint32_t WIN_H = 720;
    constexpr uint32_t SC_IMG_COUNT = 3u;
    constexpr uint32_t FRAMES_IN_FLIGHT = 5u;
    static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

    auto initOutput = CommonAPI::Init<WIN_W, WIN_H, SC_IMG_COUNT>(video::EAT_OPENGL, "MeshLoaders", nbl::asset::EF_D32_SFLOAT);
    auto window = std::move(initOutput.window);
    auto gl = std::move(initOutput.apiConnection);
    auto surface = std::move(initOutput.surface);
    auto gpuPhysicalDevice = std::move(initOutput.physicalDevice);
    auto logicalDevice = std::move(initOutput.logicalDevice);
    auto queues = std::move(initOutput.queues);
    auto swapchain = std::move(initOutput.swapchain);
    auto renderpass = std::move(initOutput.renderpass);
    auto fbos = std::move(initOutput.fbo);
    auto commandPool = std::move(initOutput.commandPool);
    auto assetManager = std::move(initOutput.assetManager);
    auto logger = std::move(initOutput.logger);
    auto inputSystem = std::move(initOutput.inputSystem);
    auto system = std::move(initOutput.system);
    auto windowCallback = std::move(initOutput.windowCb);
    auto cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
    auto utilities = std::move(initOutput.utilities);

    core::smart_refctd_ptr<video::IGPUFence> gpuTransferFence;
    core::smart_refctd_ptr<video::IGPUFence> gpuComputeFence;
    nbl::video::IGPUObjectFromAssetConverter cpu2gpu;
    {
        cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].fence = &gpuTransferFence;
        cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].fence = &gpuComputeFence;
    }

    asset::ICPUMesh* meshRaw = nullptr;
    const asset::COBJMetadata* metaOBJ = nullptr;
    {        
        auto* quantNormalCache = assetManager->getMeshManipulator()->getQuantNormalCache();
        quantNormalCache->loadCacheFromFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), "../../tmp/normalCache101010.sse");

        system::path archPath = CWD/"../../media/sponza.zip";
        auto arch = system->openFileArchive(archPath);
        system->mount(std::move(arch),"resources");
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

        quantNormalCache->saveCacheToFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), "../../tmp/normalCache101010.sse");
    }

    const asset::COBJMetadata::CRenderpassIndependentPipeline* pipelineMetadata = nullptr;
    uint32_t cameraUBOBinding = 0u;
    core::smart_refctd_ptr<video::IGPUBuffer> cameraUBO;
    core::smart_refctd_ptr<video::IGPUDescriptorSet> perCameraDescSet;
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
            if (bnd.type==asset::EDT_UNIFORM_BUFFER)
            {
                cameraUBOBinding = bnd.binding;
                break;
            }

            for (const auto& shdrIn : pipelineMetadata->m_inputSemantics)
            if (shdrIn.descriptorSection.type==asset::IRenderpassIndependentPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set==1u && shdrIn.descriptorSection.uniformBufferObject.binding==cameraUBOBinding)
                neededDS1UBOsz = std::max<size_t>(neededDS1UBOsz,shdrIn.descriptorSection.uniformBufferObject.relByteoffset+shdrIn.descriptorSection.uniformBufferObject.bytesize);

            auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&ds1layout,&ds1layout+1,cpu2gpuParams);
            assert(gpu_array&&gpu_array->size()&&(*gpu_array)[0]);
            gpuds1layout = (*gpu_array)[0];
            cpu2gpuParams.waitForCreationToComplete();
        }

        auto descriptorPool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE,&gpuds1layout.get(),&gpuds1layout.get()+1);

        auto ubomemreq = logicalDevice->getDeviceLocalGPUMemoryReqs();
        ubomemreq.vulkanReqs.size = neededDS1UBOsz;
        cameraUBO = logicalDevice->createGPUBufferOnDedMem(ubomemreq,true);
        perCameraDescSet = logicalDevice->createGPUDescriptorSet(descriptorPool.get(),std::move(gpuds1layout));
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
    
    core::smart_refctd_ptr<video::IGPUMesh> gpumesh;
    {
        auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&meshRaw, &meshRaw + 1, cpu2gpuParams);
        assert(gpu_array&&gpu_array->size()&&(*gpu_array)[0]);
        gpumesh = (*gpu_array)[0];
        cpu2gpuParams.waitForCreationToComplete();
    }

    core::smart_refctd_ptr<video::IGPUCommandBuffer> bakedCommandBuffer;
    logicalDevice->createCommandBuffers(commandPool.get(),video::IGPUCommandBuffer::EL_SECONDARY,1u,&bakedCommandBuffer);
    bakedCommandBuffer->begin(video::IGPUCommandBuffer::EU_RENDER_PASS_CONTINUE_BIT|video::IGPUCommandBuffer::EU_SIMULTANEOUS_USE_BIT);
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
                std::fill_n(allocation.multiDrawCommandRangeByteOffsets,allocation.count,video::IDrawIndirectAllocator::invalid_draw_range_begin);
            }
            {
                auto drawCounts = new uint32_t[allocation.count];
                std::fill_n(drawCounts,allocation.count,1u);
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
            core::map<const void*,core::smart_refctd_ptr<video::IGPUGraphicsPipeline>> graphicsPipelines;
            for (auto& mb : gpumesh->getMeshBuffers())
            {
                auto& drawcall = kiln.getDrawcallMetadataVector().emplace_back();
                memcpy(drawcall.pushConstantData,mb->getPushConstantsDataPtr(),video::IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE);
                {
                    auto renderpassIndep = mb->getPipeline();
                    auto foundPpln = graphicsPipelines.find(renderpassIndep);
                    if (foundPpln==graphicsPipelines.end())
                    {
                        video::IGPUGraphicsPipeline::SCreationParams params;
                        params.renderpassIndependent = core::smart_refctd_ptr<const video::IGPURenderpassIndependentPipeline>(renderpassIndep);
                        params.renderpass = core::smart_refctd_ptr(renderpass);
                        params.subpassIx = kSubpassIx;
                        foundPpln = graphicsPipelines.emplace_hint(foundPpln,renderpassIndep,logicalDevice->createGPUGraphicsPipeline(nullptr,std::move(params)));
                    }
                    drawcall.pipeline = foundPpln->second;
                }
                drawcall.descriptorSets[1] = perCameraDescSet;
                drawcall.descriptorSets[3] = core::smart_refctd_ptr<const video::IGPUDescriptorSet>(mb->getAttachedDescriptorSet());
                std::copy_n(mb->getVertexBufferBindings(),video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT,drawcall.vertexBufferBindings);
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
                        drawCallDataIt->firstIndex = mb->getIndexBufferBinding().offset/sizeof(uint32_t);
                        break;
                    case asset::EIT_16BIT:
                        drawCallDataIt->firstIndex = mb->getIndexBufferBinding().offset/sizeof(uint16_t);
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
                bakedCommandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS,drawcall.pipeline->getRenderpassIndependentPipeline()->getLayout(),1u,3u,&drawcall.descriptorSets->get()+1u,nullptr);
                bakedCommandBuffer->pushConstants(drawcall.pipeline->getRenderpassIndependentPipeline()->getLayout(),video::IGPUSpecializedShader::ESS_FRAGMENT,0u,video::IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE,drawcall.pushConstantData);
                bakedCommandBuffer->drawMeshBuffer(mb);
    #endif
            }
        }
        // do the transfer of drawcall structs
        {
            video::CPropertyPoolHandler::TransferRequest request;
            request.memblock = drawAllocator->getDrawCommandMemoryBlock();
            request.flags = decltype(request)::EF_NONE;
            request.elementSize = sizeof(asset::DrawElementsIndirectCommand_t);
            request.elementCount = drawCallCount;
            request.srcAddresses = nullptr; // iota 0,1,2,3,4,etc.
            std::for_each_n(allocation.multiDrawCommandRangeByteOffsets,request.elementCount,[=](auto& handle){handle/=request.elementSize;});
            request.dstAddresses = allocation.multiDrawCommandRangeByteOffsets;
            request.device2device = false;
            request.source = drawCallData;

            auto fence = logicalDevice->createFence(video::IGPUFence::ECF_UNSIGNALED);
            core::smart_refctd_ptr<video::IGPUCommandBuffer> tferCmdBuf;
            logicalDevice->createCommandBuffers(commandPool.get(),video::IGPUCommandBuffer::EL_PRIMARY,1u,&tferCmdBuf);
            tferCmdBuf->begin(0u); // TODO some one time submit bit or something
            utilities->getDefaultPropertyPoolHandler()->transferProperties(utilities->getDefaultUpStreamingBuffer(),nullptr,tferCmdBuf.get(),fence.get(),&request,&request+1u,logger.get());
            tferCmdBuf->end();
            {
                video::IGPUQueue::SSubmitInfo submit = {}; // intializes all semaphore stuff to 0 and nullptr
                submit.commandBufferCount = 1u;
                submit.commandBuffers = &tferCmdBuf.get();
                queues[decltype(initOutput)::EQT_TRANSFER_UP]->submit(1u,&submit,fence.get());
            }
            logicalDevice->blockForFences(1u,&fence.get());
        }
        delete[] drawCallData;
        // free the draw command index list
        delete[] allocation.multiDrawCommandRangeByteOffsets;

#ifndef REFERENCE
        kiln.bake(bakedCommandBuffer.get(),renderpass.get(),kSubpassIx,drawAllocator->getDrawCommandMemoryBlock().buffer.get(),nullptr);
#endif
    }
    bakedCommandBuffer->end();

    CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> mouse;
    CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

    core::vectorSIMDf cameraPosition(0, 5, -10);
    matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60), float(WIN_W) / WIN_H, 2.f, 4000.f);
    Camera camera = Camera(cameraPosition, core::vectorSIMDf(0, 0, 0), projectionMatrix, 10.f, 1.f);
    
    video::CDumbPresentationOracle oracle;
    oracle.reportBeginFrameRecord();

    core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[FRAMES_IN_FLIGHT];
    logicalDevice->createCommandBuffers(commandPool.get(),video::IGPUCommandBuffer::EL_PRIMARY,FRAMES_IN_FLIGHT,commandBuffers);

    core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
    core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
    core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
    for (uint32_t i=0u; i<FRAMES_IN_FLIGHT; i++)
    {
        imageAcquire[i] = logicalDevice->createSemaphore();
        renderFinished[i] = logicalDevice->createSemaphore();
    }

    uint32_t acquiredNextFBO = {};
    auto resourceIx = -1;
	while(windowCallback->isWindowOpen())
	{
        ++resourceIx;
        if (resourceIx >= FRAMES_IN_FLIGHT)
            resourceIx = 0;
        
        auto& commandBuffer = commandBuffers[resourceIx];
        auto& fence = frameComplete[resourceIx];
        if (fence)
            logicalDevice->blockForFences(1u,&fence.get());
        else
            fence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

        //
        commandBuffer->reset(nbl::video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
        commandBuffer->begin(0);

        // late latch input
        const auto nextPresentationTimestamp = oracle.acquireNextImage(swapchain.get(),imageAcquire[resourceIx].get(),nullptr,&acquiredNextFBO);

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
                if (shdrIn.descriptorSection.type==asset::IRenderpassIndependentPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set==1u && shdrIn.descriptorSection.uniformBufferObject.binding==cameraUBOBinding)
                {
                    switch (shdrIn.type)
                    {
                        case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW_PROJ:
                        {
                            memcpy(uboData.data()+shdrIn.descriptorSection.uniformBufferObject.relByteoffset, viewProjectionMatrix.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
                        } break;
                    
                        case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW:
                        {
                            memcpy(uboData.data() + shdrIn.descriptorSection.uniformBufferObject.relByteoffset, viewMatrix.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
                        } break;
                    
                        case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW_INVERSE_TRANSPOSE:
                        {
                            memcpy(uboData.data()+shdrIn.descriptorSection.uniformBufferObject.relByteoffset, viewMatrix.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
                        } break;
                    }
                }
            }       
            commandBuffer->updateBuffer(cameraUBO.get(),0ull,cameraUBO->getSize(),uboData.data());
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
                beginInfo.framebuffer = fbos[acquiredNextFBO];
                beginInfo.renderpass = renderpass;
                beginInfo.renderArea = area;
                beginInfo.clearValues = clear;
            }

            commandBuffer->beginRenderPass(&beginInfo, nbl::asset::ESC_INLINE);
            commandBuffer->executeCommands(1u,&bakedCommandBuffer.get());
            commandBuffer->endRenderPass();

            commandBuffer->end();
        }

        CommonAPI::Submit(logicalDevice.get(), swapchain.get(), commandBuffer.get(), queues[decltype(initOutput)::EQT_GRAPHICS], imageAcquire[resourceIx].get(), renderFinished[resourceIx].get(), fence.get());
        CommonAPI::Present(logicalDevice.get(), swapchain.get(), queues[decltype(initOutput)::EQT_GRAPHICS], renderFinished[resourceIx].get(), acquiredNextFBO);
    }

    const auto& fboCreationParams = fbos[acquiredNextFBO]->getCreationParameters();
    auto gpuSourceImageView = fboCreationParams.attachments[0];

    bool status = ext::ScreenShot::createScreenShot(logicalDevice.get(), queues[decltype(initOutput)::EQT_TRANSFER_DOWN], renderFinished[resourceIx].get(), gpuSourceImageView.get(), assetManager.get(), "ScreenShot.png");
    assert(status);

    return 0;
}