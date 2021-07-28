// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nabla.h>

#include "../common/CommonAPI.h"
#include "../source/Nabla/CFileSystem.h"

using namespace nbl;
using namespace core;

int main()
{
    constexpr uint32_t WIN_W = 1280;
    constexpr uint32_t WIN_H = 720;
    constexpr uint32_t FBO_COUNT = 1u;

    auto initOutput = CommonAPI::Init<WIN_W, WIN_H, FBO_COUNT>(video::EAT_OPENGL, "MeshLoaders");
    auto window = std::move(initOutput.window);
    auto gl = std::move(initOutput.apiConnection);
    auto surface = std::move(initOutput.surface);
    auto gpuPhysicalDevice = std::move(initOutput.physicalDevice);
    auto logicalDevice = std::move(initOutput.logicalDevice);
    auto queue = std::move(initOutput.queue);
    auto swapchain = std::move(initOutput.swapchain);
    auto renderpass = std::move(initOutput.renderpass);
    auto fbo = std::move(initOutput.fbo[0]);
    auto commandPool = std::move(initOutput.commandPool);
    {
        video::IDriverMemoryBacked::SDriverMemoryRequirements mreq;
        core::smart_refctd_ptr<video::IGPUCommandBuffer> gpuCommandBuffer;
        logicalDevice->createCommandBuffers(commandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &gpuCommandBuffer);
        assert(gpuCommandBuffer);

        gpuCommandBuffer->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);

        asset::SViewport viewport;
        viewport.minDepth = 1.f;
        viewport.maxDepth = 0.f;
        viewport.x = 0u;
        viewport.y = 0u;
        viewport.width = WIN_W;
        viewport.height = WIN_H;
        gpuCommandBuffer->setViewport(0u, 1u, &viewport);

        gpuCommandBuffer->end();

        video::IGPUQueue::SSubmitInfo info;
        info.commandBufferCount = 1u;
        info.commandBuffers = &gpuCommandBuffer.get();
        info.pSignalSemaphores = nullptr;
        info.signalSemaphoreCount = 0u;
        info.pWaitSemaphores = nullptr;
        info.waitSemaphoreCount = 0u;
        info.pWaitDstStageMask = nullptr;
        queue->submit(1u, &info, nullptr);
    }

    core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> commandBuffers[1];
    logicalDevice->createCommandBuffers(commandPool.get(), nbl::video::IGPUCommandBuffer::EL_PRIMARY, 1, commandBuffers);
    auto commandBuffer = commandBuffers[0];

    core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
    {
        nbl::core::smart_refctd_ptr<nbl::io::IFileSystem> fileSystem = nbl::core::make_smart_refctd_ptr<nbl::io::CFileSystem>("");
        assetManager = core::make_smart_refctd_ptr<nbl::asset::IAssetManager>(std::move(fileSystem));
    }

    core::smart_refctd_ptr<nbl::scene::ISceneManager> sceneManager;
    {
        // how to create properly Scene Manager in new API?
        // only interface class provided
    }

    nbl::video::IGPUObjectFromAssetConverter cpu2gpu;
    nbl::video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
    cpu2gpuParams.assetManager = assetManager.get();
    cpu2gpuParams.device = logicalDevice.get();
    cpu2gpuParams.finalQueueFamIx = queue->getFamilyIndex();
    cpu2gpuParams.limits = gpuPhysicalDevice->getLimits();
    cpu2gpuParams.pipelineCache = nullptr;
    cpu2gpuParams.sharingMode = nbl::asset::ESM_EXCLUSIVE;
    cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].queue = queue;
    cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].queue = queue;

    auto createDescriptorPool = [&](const uint32_t textureCount)
    {
        constexpr uint32_t maxItemCount = 256u;
        {
            nbl::video::IDescriptorPool::SDescriptorPoolSize poolSize;
            poolSize.count = textureCount;
            poolSize.type = nbl::asset::EDT_COMBINED_IMAGE_SAMPLER;
            return logicalDevice->createDescriptorPool(static_cast<nbl::video::IDescriptorPool::E_CREATE_FLAGS>(0), maxItemCount, 1u, &poolSize);
        }
    };

    asset::ICPUMesh* meshRaw = nullptr;
    const asset::COBJMetadata* metaOBJ = nullptr;
    {
        auto* fileSystem = assetManager->getFileSystem();

        auto* quantNormalCache = assetManager->getMeshManipulator()->getQuantNormalCache();
        quantNormalCache->loadCacheFromFile<asset::EF_A2B10G10R10_SNORM_PACK32>(fileSystem, "../../tmp/normalCache101010.sse");

        fileSystem->addFileArchive("../../media/sponza.zip");

        asset::IAssetLoader::SAssetLoadParams loadParams;
        auto meshes_bundle = assetManager->getAsset("sponza.obj", loadParams);
        assert(!meshes_bundle.getContents().empty());

        metaOBJ = meshes_bundle.getMetadata()->selfCast<const asset::COBJMetadata>();

        auto cpuMesh = meshes_bundle.getContents().begin()[0];
        meshRaw = static_cast<asset::ICPUMesh*>(cpuMesh.get());

        quantNormalCache->saveCacheToFile<asset::EF_A2B10G10R10_SNORM_PACK32>(fileSystem, "../../tmp/normalCache101010.sse");
    }

    // we can safely assume that all meshbuffers within mesh loaded from OBJ has same DS1 layout (used for camera-specific data)
    asset::ICPUMeshBuffer* const firstMeshBuffer = *meshRaw->getMeshBuffers().begin();
    auto pipelineMetadata = metaOBJ->getAssetSpecificMetadata(firstMeshBuffer->getPipeline());

    // so we can create just one DS
    asset::ICPUDescriptorSetLayout* ds1layout = firstMeshBuffer->getPipeline()->getLayout()->getDescriptorSetLayout(1u);
    uint32_t ds1UboBinding = 0u;
    for (const auto& bnd : ds1layout->getBindings())
        if (bnd.type==asset::EDT_UNIFORM_BUFFER)
        {
            ds1UboBinding = bnd.binding;
            break;
        }

    size_t neededDS1UBOsz = 0ull;
    {
        for (const auto& shdrIn : pipelineMetadata->m_inputSemantics)
            if (shdrIn.descriptorSection.type==asset::IRenderpassIndependentPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set==1u && shdrIn.descriptorSection.uniformBufferObject.binding==ds1UboBinding)
                neededDS1UBOsz = std::max<size_t>(neededDS1UBOsz, shdrIn.descriptorSection.uniformBufferObject.relByteoffset+shdrIn.descriptorSection.uniformBufferObject.bytesize);
    }

    core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> gpuds1layout;
    {
        auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&ds1layout, &ds1layout + 1, cpu2gpuParams);
        if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
            assert(false);

        gpuds1layout = (*gpu_array)[0];
    }

    auto descriptorPool = createDescriptorPool(1u);

    auto gpuubo = logicalDevice->createDeviceLocalGPUBufferOnDedMem(neededDS1UBOsz);
    auto gpuds1 = logicalDevice->createGPUDescriptorSet(descriptorPool.get(), std::move(gpuds1layout));
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

    core::smart_refctd_ptr<video::IGPUMesh> gpumesh;
    {
        auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&meshRaw, &meshRaw + 1, cpu2gpuParams);
        if (!gpu_array || gpu_array->size() < 1u || !(*gpu_array)[0])
            assert(false);

        gpumesh = (*gpu_array)[0];
    }

    std::map<core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline>, core::smart_refctd_ptr<video::IGPUGraphicsPipeline>> gpuPipelines;
    {
        for (size_t i = 0; i < gpumesh->getMeshBuffers().size(); ++i)
        {
            nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams;
            graphicsPipelineParams.renderpassIndependent = core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline>(const_cast<video::IGPURenderpassIndependentPipeline*>(gpumesh->getMeshBuffers().begin()[i]->getPipeline()));
            graphicsPipelineParams.renderpass = core::smart_refctd_ptr(renderpass);

            gpuPipelines[graphicsPipelineParams.renderpassIndependent] = logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(graphicsPipelineParams));
        }
    }

	scene::ICameraSceneNode* camera = sceneManager->addCameraSceneNodeFPS(0,100.0f,0.5f);

	camera->setPosition(core::vector3df(-4,0,0));
	camera->setTarget(core::vector3df(0,0,0));
	camera->setNearValue(1.f);
	camera->setFarValue(5000.0f);

    sceneManager->setActiveCamera(camera);

	while(true)
	{
        commandBuffer->reset(nbl::video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
        commandBuffer->begin(0);

        nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
        nbl::asset::VkRect2D area;
        area.offset = { 0,0 };
        area.extent = { WIN_W, WIN_H };
        nbl::asset::SClearValue clear;
        clear.color.float32[0] = 0.f;
        clear.color.float32[1] = 0.f;
        clear.color.float32[2] = 0.f;
        clear.color.float32[3] = 1.f;
        beginInfo.clearValueCount = 1u;
        beginInfo.framebuffer = fbo;
        beginInfo.renderpass = renderpass;
        beginInfo.renderArea = area;
        beginInfo.clearValues = &clear;

        commandBuffer->beginRenderPass(&beginInfo, nbl::asset::ESC_INLINE);

        core::vector<uint8_t> uboData(gpuubo->getSize());
        for (const auto& shdrIn : pipelineMetadata->m_inputSemantics)
        {
            if (shdrIn.descriptorSection.type==asset::IRenderpassIndependentPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set==1u && shdrIn.descriptorSection.uniformBufferObject.binding==ds1UboBinding)
            {
                switch (shdrIn.type)
                {
                    case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW_PROJ:
                    {
                        core::matrix4SIMD mvp = camera->getConcatenatedMatrix();
                        memcpy(uboData.data()+shdrIn.descriptorSection.uniformBufferObject.relByteoffset, mvp.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
                    } break;
                    
                    case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW:
                    {
                        core::matrix3x4SIMD MV = camera->getViewMatrix();
                        memcpy(uboData.data() + shdrIn.descriptorSection.uniformBufferObject.relByteoffset, MV.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
                    } break;
                    
                    case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW_INVERSE_TRANSPOSE:
                    {
                        core::matrix3x4SIMD MV = camera->getViewMatrix();
                        memcpy(uboData.data()+shdrIn.descriptorSection.uniformBufferObject.relByteoffset, MV.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
                    } break;
                }
            }
        }       

        commandBuffer->updateBuffer(gpuubo.get(), 0ull, gpuubo->getSize(), uboData.data());

        for (size_t i = 0; i < gpumesh->getMeshBuffers().size(); ++i)
        {
            auto gpuMeshBuffer = gpumesh->getMeshBuffers().begin()[i];
            auto gpuGraphicsPipeline = gpuPipelines[core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline>(const_cast<video::IGPURenderpassIndependentPipeline*>(gpuMeshBuffer->getPipeline()))];

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

        commandBuffer->endRenderPass();
        commandBuffer->end();

        auto img_acq_sem = logicalDevice->createSemaphore();
        auto render_finished_sem = logicalDevice->createSemaphore();

        uint32_t imgnum = 0u;
        constexpr uint64_t MAX_TIMEOUT = 99999999999999ull; // ns
        swapchain->acquireNextImage(MAX_TIMEOUT, img_acq_sem.get(), nullptr, &imgnum);

        CommonAPI::Submit(logicalDevice.get(), swapchain.get(), commandBuffer.get(), queue, img_acq_sem.get(), render_finished_sem.get());
        CommonAPI::Present(logicalDevice.get(), swapchain.get(), queue, render_finished_sem.get(), imgnum);
	}

	return 0;
}