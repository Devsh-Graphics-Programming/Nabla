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

#define NABLA_QUEUE_COUNT 1             // per family
#define NABLA_RENDER_QUEUE_INDEX 0      // index within family
#define NABLA_TRANSFER_QUEUE_INDEX 0    // index within family
#define NABLA_COMPUTE_QUEUE_INDEX 0     // index within family
#define NABLA_SHARING_MODE ::nbl::asset::ESM_EXCLUSIVE
#define NABLA_SWAPCHAIN_IMAGE_FORMAT ::nbl::asset::EF_R8G8B8A8_SRGB

inline void debugCallback(nbl::video::E_DEBUG_MESSAGE_SEVERITY severity, nbl::video::E_DEBUG_MESSAGE_TYPE type, const char* msg, void* userData)
{
    const char* sev = nullptr;

    switch (severity)
    {
        case video::EDMS_VERBOSE:
        {
            sev = "verbose";
        } break;

        case video::EDMS_INFO:
        {
            sev = "info";
        } break;

        case video::EDMS_WARNING:
        {
            sev = "warning";
        } break;

        case video::EDMS_ERROR:
        {
            sev = "error";
        } break;
    }

    std::cout << "OpenGL " << sev << ": " << msg << std::endl;
}

int main()
{
    constexpr uint32_t WIN_W = 800u;
    constexpr uint32_t WIN_H = 600u;
    constexpr uint32_t SC_IMG_COUNT = 3u;

    auto window = CWindowT::create(WIN_W, WIN_H, ui::IWindow::ECF_NONE);

    video::SDebugCallback debugCallbackObject;
    debugCallbackObject.callback = &debugCallback;
    debugCallbackObject.userData = nullptr;

    auto glApiConnection = video::IAPIConnection::create(video::EAT_OPENGL, 0, "New API Test", debugCallbackObject);
    auto surface = glApiConnection->createSurface(window.get());

    auto physicalDevices = glApiConnection->getPhysicalDevices();
    assert(!physicalDevices.empty());

    nbl::core::smart_refctd_dynamic_array<uint32_t> queueFamilyIxs = nullptr;
    uint32_t queueFamilyIx_render = 0xffffFFFFu;
    uint32_t queueFamilyIx_compute = 0xffffFFFFu;
    uint32_t queueFamilyIx_transfer = 0xffffFFFFu;

    auto gpuPhysicalDevice = physicalDevices.begin()[0];
    {
        auto queueFamilyPropertiesRange = gpuPhysicalDevice->getQueueFamilyProperties();
        for (uint32_t i = 0u; i < queueFamilyPropertiesRange.size(); ++i)
        {
            const auto& queueFamilyProperties = queueFamilyPropertiesRange.begin()[i];

            if (queueFamilyProperties.queueFlags & nbl::video::IPhysicalDevice::EQF_GRAPHICS_BIT)
                queueFamilyIx_render = i;
            if (queueFamilyProperties.queueFlags & nbl::video::IPhysicalDevice::EQF_COMPUTE_BIT)
                queueFamilyIx_compute = i;
            if (queueFamilyProperties.queueFlags & nbl::video::IPhysicalDevice::EQF_TRANSFER_BIT)
                queueFamilyIx_transfer = i;

            if (queueFamilyIx_render < queueFamilyPropertiesRange.size() && queueFamilyIx_compute < queueFamilyPropertiesRange.size() && queueFamilyIx_transfer < queueFamilyPropertiesRange.size())
                break;
            else if (i == queueFamilyPropertiesRange.size() - 1u)
                assert(false);
        }
    }

    nbl::core::set<uint32_t> families = { queueFamilyIx_render, queueFamilyIx_compute, queueFamilyIx_transfer };

    nbl::video::ILogicalDevice::SCreationParams deviceParams;
    deviceParams.queueParamsCount = families.size();
    nbl::core::vector<nbl::video::ILogicalDevice::SQueueCreationParams> queuesParams;
    queuesParams.reserve(families.size());

    float priorities[32];
    std::fill_n(priorities, 32, 1.f);

    for (auto& family : families)
    {
        nbl::video::ILogicalDevice::SQueueCreationParams qp;
        qp.count = NABLA_QUEUE_COUNT;
        qp.familyIndex = family;
        qp.flags = static_cast<nbl::video::IGPUQueue::E_CREATE_FLAGS>(0);
        qp.priorities = priorities;
        queuesParams.push_back(qp);
    }
    deviceParams.queueCreateInfos = queuesParams.data();

    auto logicalDevice = gpuPhysicalDevice->createLogicalDevice(deviceParams);

    if (!logicalDevice)
		return 1; // could not create selected driver.

    core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> commandBuffer[1];
    auto commandPool = logicalDevice->createCommandPool(queueFamilyIx_render, nbl::video::IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
    logicalDevice->createCommandBuffers(commandPool.get(), nbl::video::IGPUCommandBuffer::EL_PRIMARY, 1, commandBuffer);

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
    cpu2gpuParams.finalQueueFamIx = queueFamilyIx_render;
    cpu2gpuParams.limits = gpuPhysicalDevice->getLimits();
    cpu2gpuParams.pipelineCache = nullptr;
    cpu2gpuParams.sharingMode = NABLA_SHARING_MODE;
    cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].queue = logicalDevice->getQueue(queueFamilyIx_transfer, NABLA_TRANSFER_QUEUE_INDEX);
    cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].queue = logicalDevice->getQueue(queueFamilyIx_compute, NABLA_COMPUTE_QUEUE_INDEX);

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

	//auto* driver = logicalDevice->getVideoDriver();
	//auto* smgr = device->getSceneManager();

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

    core::smart_refctd_ptr<video::IGPUGraphicsPipeline> gpuGraphicsPipeline;
    {
        nbl::video::IGPURenderpass::SCreationParams::SAttachmentDescription attachment;
        attachment.initialLayout = nbl::asset::EIL_UNDEFINED;
        attachment.finalLayout = nbl::asset::EIL_UNDEFINED;
        attachment.format = NABLA_SWAPCHAIN_IMAGE_FORMAT;
        attachment.samples = nbl::asset::IImage::ESCF_1_BIT;
        attachment.loadOp = nbl::video::IGPURenderpass::ELO_CLEAR;
        attachment.storeOp = nbl::video::IGPURenderpass::ESO_STORE;
        
        nbl::video::IGPURenderpass::SCreationParams::SSubpassDescription::SAttachmentRef colorAttachmentRef;
        colorAttachmentRef.attachment = 0u;
        colorAttachmentRef.layout = nbl::asset::EIL_UNDEFINED;

        nbl::video::IGPURenderpass::SCreationParams::SSubpassDescription subpassDescription;
        subpassDescription.colorAttachmentCount = 1u;
        subpassDescription.colorAttachments = &colorAttachmentRef;
        subpassDescription.depthStencilAttachment = nullptr;
        subpassDescription.flags = nbl::video::IGPURenderpass::ESDF_NONE;
        subpassDescription.inputAttachmentCount = 0u;
        subpassDescription.inputAttachments = nullptr;
        subpassDescription.preserveAttachmentCount = 0u;
        subpassDescription.preserveAttachments = nullptr;
        subpassDescription.resolveAttachments = nullptr;

        nbl::video::IGPURenderpass::SCreationParams renderpassParams;
        renderpassParams.attachmentCount = 1u;
        renderpassParams.attachments = &attachment;
        renderpassParams.dependencies = nullptr;
        renderpassParams.dependencyCount = 0u;
        renderpassParams.subpasses = &subpassDescription;
        renderpassParams.subpassCount = 1u;

        auto gpuRenderpass = logicalDevice->createGPURenderpass(renderpassParams);

        nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams;
        graphicsPipelineParams.renderpassIndependent = core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline>(const_cast<video::IGPURenderpassIndependentPipeline*>(gpumesh->getMeshBuffers().begin()[0]->getPipeline()));
        graphicsPipelineParams.renderpass = gpuRenderpass;

        gpuGraphicsPipeline = logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(graphicsPipelineParams));
    }

	scene::ICameraSceneNode* camera = sceneManager->addCameraSceneNodeFPS(0,100.0f,0.5f);

	camera->setPosition(core::vector3df(-4,0,0));
	camera->setTarget(core::vector3df(0,0,0));
	camera->setNearValue(1.f);
	camera->setFarValue(5000.0f);

    sceneManager->setActiveCamera(camera);

    nbl::asset::SViewport viewport;
    {
        viewport.maxDepth = 0.f;
        viewport.minDepth = 1.f;
        viewport.x = 0u;
        viewport.y = 0u;
        viewport.width = WIN_W;
        viewport.height = WIN_H;
    }

	while(true)
	{
        // Shouldn't I need to invoke something like beginScene and endScene? Is there any such call on new API?

        commandBuffer[0]->setViewport(0, 1, &viewport);

        const size_t msTimeCount = 69; // device->getTimer()->getTime()).count(), how to get time in new API properly?

		camera->OnAnimate(msTimeCount);
		camera->render();

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

        commandBuffer[0]->updateBuffer(gpuubo.get(), 0ull, gpuubo->getSize(), uboData.data());

        for (auto gpumb : gpumesh->getMeshBuffers())
        {
            const video::IGPURenderpassIndependentPipeline* gpuRenderpassIndependentPipeline = gpumb->getPipeline();
            const video::IGPUDescriptorSet* ds3 = gpumb->getAttachedDescriptorSet();
            
            commandBuffer[0]->bindGraphicsPipeline(gpuGraphicsPipeline.get()); // I'm wondering if I should create gpuGraphicsPipeline for each mesh buffer before

            const video::IGPUDescriptorSet* gpuds1_ptr = gpuds1.get();
            commandBuffer[0]->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuRenderpassIndependentPipeline->getLayout(), 1u, 1u, &gpuds1_ptr, nullptr);
            const video::IGPUDescriptorSet* gpuds3_ptr = gpumb->getAttachedDescriptorSet();
            if (gpuds3_ptr)
                commandBuffer[0]->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuRenderpassIndependentPipeline->getLayout(), 3u, 1u, &gpuds3_ptr, nullptr);
            commandBuffer[0]->pushConstants(gpuRenderpassIndependentPipeline->getLayout(), video::IGPUSpecializedShader::ESS_FRAGMENT, 0u, gpumb->MAX_PUSH_CONSTANT_BYTESIZE, gpumb->getPushConstantsDataPtr());

            // commandBuffer[0]->draw(vertexCount, instanceCount, firstVertex, firstInstance); // TODO! Damn, do I need an access to VAO and vertex stuff?
        }
	}

	return 0;
}