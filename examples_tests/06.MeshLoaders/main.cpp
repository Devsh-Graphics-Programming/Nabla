// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nabla.h>

#include "../common/CommonAPI.h"
#include "nbl/ext/ScreenShot/ScreenShot.h"

using namespace nbl;
using namespace core;

int main(int argc, char** argv)
{
    system::path CWD = system::path(argv[0]).parent_path().generic_string() + "/";
    constexpr uint32_t WIN_W = 1280;
    constexpr uint32_t WIN_H = 720;
    constexpr uint32_t FBO_COUNT = 1u;

    auto initOutput = CommonAPI::Init<WIN_W, WIN_H, FBO_COUNT>(video::EAT_OPENGL, "MeshLoaders");
    auto window = std::move(initOutput.window);
    auto gl = std::move(initOutput.apiConnection);
    auto surface = std::move(initOutput.surface);
    auto gpuPhysicalDevice = std::move(initOutput.physicalDevice);
    auto logicalDevice = std::move(initOutput.logicalDevice);
    auto queues = std::move(initOutput.queues);
    auto swapchain = std::move(initOutput.swapchain);
    auto renderpass = std::move(initOutput.renderpass);
    auto fbo = std::move(initOutput.fbo[0]);
    auto commandPool = std::move(initOutput.commandPool);
    auto assetManager = std::move(initOutput.assetManager);
    auto cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
    auto logger = std::move(initOutput.logger);
    nbl::video::IGPUObjectFromAssetConverter cpu2gpu;

    core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> commandBuffers[1];
    logicalDevice->createCommandBuffers(commandPool.get(), nbl::video::IGPUCommandBuffer::EL_PRIMARY, 1, commandBuffers);
    auto commandBuffer = commandBuffers[0];

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
        //auto* fileSystem = assetManager->getFileSystem();

        //auto* quantNormalCache = assetManager->getMeshManipulator()->getQuantNormalCache();
        //quantNormalCache->loadCacheFromFile<asset::EF_A2B10G10R10_SNORM_PACK32>(fileSystem, "../../tmp/normalCache101010.sse"); // Matt what about this?

        //fileSystem->addFileArchive("../../media/sponza.zip"); 

        /*
            To make it work we need to read sponza but not from a zip,
            so remember to unpack sponza.zip to /bin directory upon executable

            TODO: come back to addFileArchive
        */

        asset::IAssetLoader::SAssetLoadParams loadParams;
        loadParams.workingDirectory = CWD;
        loadParams.logger = logger.get();
        auto meshes_bundle = assetManager->getAsset("sponza.obj", loadParams);
        assert(!meshes_bundle.getContents().empty());

        metaOBJ = meshes_bundle.getMetadata()->selfCast<const asset::COBJMetadata>();

        auto cpuMesh = meshes_bundle.getContents().begin()[0];
        meshRaw = static_cast<asset::ICPUMesh*>(cpuMesh.get());

        //quantNormalCache->saveCacheToFile<asset::EF_A2B10G10R10_SNORM_PACK32>(fileSystem, "../../tmp/normalCache101010.sse"); // Matt what about this?
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

    auto ubomemreq = logicalDevice->getDeviceLocalGPUMemoryReqs();
    ubomemreq.vulkanReqs.size = neededDS1UBOsz;
    auto gpuubo = logicalDevice->createGPUBufferOnDedMem(ubomemreq,true);
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

    using RENDERPASS_INDEPENDENT_PIPELINE_ADRESS = size_t;
    std::map<RENDERPASS_INDEPENDENT_PIPELINE_ADRESS, core::smart_refctd_ptr<video::IGPUGraphicsPipeline>> gpuPipelines;
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

    core::vectorSIMDf cameraPosition(-1, 2, -10);
    matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(90), float(WIN_W) / WIN_H, 0.01, 100);
    matrix3x4SIMD viewMatrix = matrix3x4SIMD::buildCameraLookAtMatrixLH(cameraPosition, core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 1, 0));
    auto viewProjectionMatrix = matrix4SIMD::concatenateBFollowedByA(projectionMatrix, matrix4SIMD(viewMatrix));

    constexpr size_t NBL_FRAMES = 1000;

    nbl::core::smart_refctd_ptr<nbl::video::IGPUSemaphore> render_finished_sem;
	for(auto frame = 0; frame < NBL_FRAMES; ++frame)
	{
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

        nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
        nbl::asset::VkRect2D area;
        area.offset = { 0,0 };
        area.extent = { WIN_W, WIN_H };
        nbl::asset::SClearValue clear;
        clear.color.float32[0] = 1.f;
        clear.color.float32[1] = 1.f;
        clear.color.float32[2] = 1.f;
        clear.color.float32[3] = 1.f;
        beginInfo.clearValueCount = 1u;
        beginInfo.framebuffer = fbo;
        beginInfo.renderpass = renderpass;
        beginInfo.renderArea = area;
        beginInfo.clearValues = &clear;

        commandBuffer->beginRenderPass(&beginInfo, nbl::asset::ESC_INLINE);

        core::matrix3x4SIMD modelMatrix;
        modelMatrix.setTranslation(nbl::core::vectorSIMDf(0, 0, 0, 0));

        core::matrix4SIMD mvp = core::concatenateBFollowedByA(viewProjectionMatrix, modelMatrix);

        core::vector<uint8_t> uboData(gpuubo->getSize());
        for (const auto& shdrIn : pipelineMetadata->m_inputSemantics)
        {
            if (shdrIn.descriptorSection.type==asset::IRenderpassIndependentPipelineMetadata::ShaderInput::ET_UNIFORM_BUFFER && shdrIn.descriptorSection.uniformBufferObject.set==1u && shdrIn.descriptorSection.uniformBufferObject.binding==ds1UboBinding)
            {
                switch (shdrIn.type)
                {
                    case asset::IRenderpassIndependentPipelineMetadata::ECSI_WORLD_VIEW_PROJ:
                    {
                        memcpy(uboData.data()+shdrIn.descriptorSection.uniformBufferObject.relByteoffset, mvp.pointer(), shdrIn.descriptorSection.uniformBufferObject.bytesize);
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

        commandBuffer->updateBuffer(gpuubo.get(), 0ull, gpuubo->getSize(), uboData.data());

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
            commandBuffer->pushConstants(gpuRenderpassIndependentPipeline->getLayout(), video::IGPUSpecializedShader::ESS_FRAGMENT, 0u, gpuMeshBuffer->MAX_PUSH_CONSTANT_BYTESIZE, gpuMeshBuffer->getPushConstantsDataPtr());

            commandBuffer->drawMeshBuffer(gpuMeshBuffer);
        }

        commandBuffer->endRenderPass();
        commandBuffer->end();

        auto img_acq_sem = logicalDevice->createSemaphore();
        render_finished_sem = logicalDevice->createSemaphore();

        uint32_t imgnum = 0u;
        constexpr uint64_t MAX_TIMEOUT = 99999999999999ull; // ns
        swapchain->acquireNextImage(MAX_TIMEOUT, img_acq_sem.get(), nullptr, &imgnum);

        CommonAPI::Submit(logicalDevice.get(), swapchain.get(), commandBuffer.get(), queues[decltype(initOutput)::EQT_GRAPHICS], img_acq_sem.get(), render_finished_sem.get());
        CommonAPI::Present(logicalDevice.get(), swapchain.get(), queues[decltype(initOutput)::EQT_GRAPHICS], render_finished_sem.get(), imgnum);
	}

    const auto& fboCreationParams = fbo->getCreationParameters();
    auto gpuSourceImageView = fboCreationParams.attachments[0];

    bool status = ext::ScreenShot::createScreenShot(logicalDevice.get(), queues[decltype(initOutput)::EQT_TRANSFER_UP], render_finished_sem.get(), gpuSourceImageView.get(), assetManager.get(), "ScreenShot.png");
    return status;
}