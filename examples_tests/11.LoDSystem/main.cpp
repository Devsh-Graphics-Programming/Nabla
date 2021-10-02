// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>
#include "nbl/scene/ICullingLoDSelectionSystem.h"

#include "../common/Camera.hpp"
#include "../common/CommonAPI.h"
#include "nbl/ext/ScreenShot/ScreenShot.h"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;

#include <random>

int main()
{
	constexpr uint32_t WIN_W = 1280;
	constexpr uint32_t WIN_H = 720;
    constexpr uint32_t FBO_COUNT = 1u;
	constexpr uint32_t FRAMES_IN_FLIGHT = 5u;
	static_assert(FRAMES_IN_FLIGHT>FBO_COUNT);

	auto initOutput = CommonAPI::Init<WIN_W, WIN_H, FBO_COUNT>(video::EAT_OPENGL, "Level of Detail System", asset::EF_D32_SFLOAT);
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
    

    //auto lodLibrary = scene::ILevelOfDetailLibrary::create();
    auto lodLibraryDSLayout = scene::ILevelOfDetailLibrary::createDescriptorSetLayout(logicalDevice.get()); // TODO: scope better

    core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> customCullingDSLayout; // TODO: scope better
    {
        // TODO: figure out what should be here
        constexpr auto BindingCount = 1u;
        video::IGPUDescriptorSetLayout::SBinding bindings[BindingCount];
        for (auto i=0u; i<BindingCount; i++)
        {
            bindings[i].binding = i;
            bindings[i].type = asset::EDT_STORAGE_BUFFER;
            bindings[i].count = 1u;
            bindings[i].stageFlags = asset::ISpecializedShader::ESS_COMPUTE;
            bindings[i].samplers = nullptr;
        }
        customCullingDSLayout = logicalDevice->createGPUDescriptorSetLayout(bindings,bindings+BindingCount);
    }
    auto cullingSystem = core::make_smart_refctd_ptr<scene::ICullingLoDSelectionSystem>(logicalDevice.get(),core::smart_refctd_ptr(customCullingDSLayout));

    core::smart_refctd_ptr<video::IGPUDescriptorSet> cullingDescriptorSets[4u];
    {
        auto inputDSLayout = scene::ICullingLoDSelectionSystem::createInputDescriptorSetLayout(logicalDevice.get());
        auto outputDSLayout = scene::ICullingLoDSelectionSystem::createOutputDescriptorSetLayout(logicalDevice.get());
        const video::IGPUDescriptorSetLayout* layouts[] = {lodLibraryDSLayout.get(),inputDSLayout.get(),outputDSLayout.get(),customCullingDSLayout.get()};
        const core::SRange<const video::IGPUDescriptorSetLayout*> layoutRange = {layouts,layouts+sizeof(layouts)/sizeof(void*)};

        auto pool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE,layoutRange.begin(),layoutRange.end());
        logicalDevice->createGPUDescriptorSets(pool.get(),layoutRange,cullingDescriptorSets);
    }
    {
        //
    }


    core::smart_refctd_ptr<video::IGPUFence> gpuTransferFence;
    core::smart_refctd_ptr<video::IGPUFence> gpuComputeFence;
    nbl::video::IGPUObjectFromAssetConverter cpu2gpu;
    {
        cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].fence = &gpuTransferFence;
        cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].fence = &gpuComputeFence;
    }

    core::smart_refctd_ptr<ICPUSpecializedShader> shaders[2];
    {
        IAssetLoader::SAssetLoadParams lp;
        lp.workingDirectory = std::filesystem::current_path();
        lp.logger = logger.get();
        auto vertexShaderBundle = assetManager->getAsset("../mesh.vert", lp);
        auto fragShaderBundle = assetManager->getAsset("../mesh.frag", lp);
        shaders[0] = IAsset::castDown<ICPUSpecializedShader>(*vertexShaderBundle.getContents().begin());
        shaders[1] = IAsset::castDown<ICPUSpecializedShader>(*fragShaderBundle.getContents().begin());
    }
    

    // TODO: refactor
    constexpr auto MaxInstanceCount = 8u;
    core::smart_refctd_ptr<video::IGPUBuffer> perViewPerInstanceDataScratch;
    {
        video::IGPUBuffer::SCreationParams params;
        params.usage = core::bitflag(asset::IBuffer::EUF_STORAGE_BUFFER_BIT);
        perViewPerInstanceDataScratch = logicalDevice->createDeviceLocalGPUBufferOnDedMem(params,sizeof(core::matrix4SIMD)*MaxInstanceCount);
        auto mreqs = logicalDevice->getDeviceLocalGPUMemoryReqs();
        mreqs.vulkanReqs.size = sizeof(core::matrix4SIMD)*MaxInstanceCount;
        perViewPerInstanceDataScratch = logicalDevice->createGPUBufferOnDedMem(params,mreqs,true);
    }


    core::smart_refctd_ptr<video::IGPUDescriptorSet> perViewDS;
    core::smart_refctd_ptr<ICPUDescriptorSetLayout> cpuPerViewDSLayout;
    {
        constexpr auto BindingCount = 1;
        ICPUDescriptorSetLayout::SBinding cpuBindings[BindingCount];
        for (auto i=0; i<BindingCount; i++)
        {
            cpuBindings[i].binding = i;
            cpuBindings[i].count = 1u;
            cpuBindings[i].stageFlags = ISpecializedShader::ESS_VERTEX;
            cpuBindings[i].samplers = nullptr;
        }
        cpuBindings[0].type = EDT_STORAGE_BUFFER;
        cpuPerViewDSLayout = core::make_smart_refctd_ptr<ICPUDescriptorSetLayout>(cpuBindings,cpuBindings+BindingCount);

        auto bindings = reinterpret_cast<video::IGPUDescriptorSetLayout::SBinding*>(cpuBindings);
        auto perViewDSLayout = logicalDevice->createGPUDescriptorSetLayout(bindings,bindings+BindingCount);
        auto dsPool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE,&perViewDSLayout.get(),&perViewDSLayout.get()+1u);
        perViewDS = logicalDevice->createGPUDescriptorSet(dsPool.get(),std::move(perViewDSLayout));
        {
            video::IGPUDescriptorSet::SWriteDescriptorSet writes[BindingCount];
            video::IGPUDescriptorSet::SDescriptorInfo infos[BindingCount];
            for (auto i=0; i<BindingCount; i++)
            {
                writes[i].dstSet = perViewDS.get();
                writes[i].binding = i;
                writes[i].arrayElement = 0u;
                writes[i].count = 1u;
                writes[i].info = infos+i;
            }
            writes[0].descriptorType = EDT_STORAGE_BUFFER;
            infos[0].desc = perViewPerInstanceDataScratch;
            infos[0].buffer = {0u,video::IGPUDescriptorSet::SDescriptorInfo::SBufferInfo::WholeBuffer};
            logicalDevice->updateDescriptorSets(BindingCount,writes,0u,nullptr);
        }
    }

    core::smart_refctd_ptr<video::IGPUBuffer> perInstancePointersScratch;
    //
    core::smart_refctd_ptr<video::IGPUCommandBuffer> bakedCommandBuffer;
    logicalDevice->createCommandBuffers(commandPool.get(),video::IGPUCommandBuffer::EL_SECONDARY,1u,&bakedCommandBuffer);
    bakedCommandBuffer->begin(video::IGPUCommandBuffer::EU_RENDER_PASS_CONTINUE_BIT|video::IGPUCommandBuffer::EU_SIMULTANEOUS_USE_BIT);
    {
        video::IDrawIndirectAllocator::ImplicitBufferCreationParameters drawAllocatorParams;
        drawAllocatorParams.device = logicalDevice.get();
        drawAllocatorParams.maxDrawCommandStride = sizeof(asset::DrawElementsIndirectCommand_t);
        drawAllocatorParams.drawCommandCapacity = 32u;
        drawAllocatorParams.drawCountCapacity = 1u;
        auto drawIndirectAllocator = video::CDrawIndirectAllocator<>::create(std::move(drawAllocatorParams));

        uint32_t maxInstancedDrawcalls = 0u;
        {
            auto* geometryCreator = assetManager->getGeometryCreator();
            auto* meshManipulator = assetManager->getMeshManipulator();
            auto* qnc = meshManipulator->getQuantNormalCache();
            //loading cache from file
            const system::path cachePath = std::filesystem::current_path()/"../../tmp/normalCache101010.sse";
            if (!qnc->loadCacheFromFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(),cachePath))
                logger->log("%s",ILogger::ELL_ERROR,"Failed to load cache.");
            auto tmp = 0;
            core::vector<core::smart_refctd_ptr<ICPUMeshBuffer>> cpumeshes;
            core::smart_refctd_ptr<ICPURenderpassIndependentPipeline> cpupipeline;
            for (uint32_t poly=4u; poly<=256; poly<<=1)
            {
                auto sphereData = geometryCreator->createSphereMesh(2.f,poly,poly,meshManipulator);
                // we'll stick instance data refs in the last attribute binding
                assert((sphereData.inputParams.enabledBindingFlags>>15u)==0u);

                sphereData.inputParams.enabledAttribFlags |= 0x1u<<15;
                sphereData.inputParams.enabledBindingFlags |= 0x1u<<15;
                sphereData.inputParams.attributes[15].binding = 15u;
                sphereData.inputParams.attributes[15].relativeOffset = 0u;
                sphereData.inputParams.attributes[15].format = asset::EF_R32G32_UINT;
                sphereData.inputParams.bindings[15].inputRate = asset::EVIR_PER_INSTANCE;
                sphereData.inputParams.bindings[15].stride = asset::getTexelOrBlockBytesize(asset::EF_R32G32_UINT);

                if (!cpupipeline)
                {
                    auto pipelinelayout = core::make_smart_refctd_ptr<ICPUPipelineLayout>(nullptr,nullptr,nullptr,std::move(cpuPerViewDSLayout));
                    cpupipeline = core::make_smart_refctd_ptr<ICPURenderpassIndependentPipeline>(std::move(pipelinelayout),&shaders->get(),&shaders->get()+2u,sphereData.inputParams,SBlendParams{},sphereData.assemblyParams,SRasterizationParams{});
                }
                constexpr auto indicesPerBatch = 1023u;
                auto i = 0u;
                for (; i<sphereData.indexCount; i+=indicesPerBatch)
                {
                    auto& mb = cpumeshes.emplace_back(
                        core::make_smart_refctd_ptr<ICPUMeshBuffer>()
                    );
                    mb->setPipeline(core::smart_refctd_ptr(cpupipeline));
                    for (auto j=0u; j<ICPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT; j++)
                        mb->setVertexBufferBinding(asset::SBufferBinding(sphereData.bindings[j]),j);
                    mb->setIndexType(sphereData.indexType);
                    mb->setIndexCount(core::min(sphereData.indexCount-i,indicesPerBatch));
                    auto indexBinding = sphereData.indexBuffer;
                    switch (sphereData.indexType)
                    {
                        case EIT_16BIT:
                            indexBinding.offset += sizeof(uint16_t)*i;
                            break;
                        case EIT_32BIT:
                            indexBinding.offset += sizeof(uint32_t)*i;
                            break;
                        default:
                            assert(false);
                            break;
                    }
                    mb->setIndexBufferBinding(std::move(indexBinding));

                    meshManipulator->recalculateBoundingBox(mb.get());
                    // TODO: undo this
                    mb->setInstanceCount(1u);
                    mb->setBaseInstance(cpumeshes.size()-1u);
                }
                maxInstancedDrawcalls = i*MaxInstanceCount;
                tmp++;
            }
            auto gpumeshes = cpu2gpu.getGPUObjectsFromAssets(cpumeshes.data(),cpumeshes.data()+cpumeshes.size(),cpu2gpuParams);
            //! cache results -- speeds up mesh generation on second run
            qnc->saveCacheToFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(),cachePath);
            cpu2gpuParams.waitForCreationToComplete();
            
            // TODO: refactor some more?
            perInstancePointersScratch = scene::ICullingLoDSelectionSystem::createInstanceRedirectBuffer(logicalDevice.get(),maxInstancedDrawcalls);

            auto renderpassindependent = core::smart_refctd_ptr_dynamic_cast<video::IGPURenderpassIndependentPipeline>(assetManager->findGPUObject(cpupipeline.get()));
            video::IGPUGraphicsPipeline::SCreationParams params;
            params.renderpass = renderpass;
            params.renderpassIndependent = renderpassindependent;
            params.subpassIx = 0u;
            auto pipeline = logicalDevice->createGPUGraphicsPipeline(nullptr,std::move(params));
            bakedCommandBuffer->bindGraphicsPipeline(pipeline.get());
            const video::IGPUDescriptorSet* descriptorSets[1] = {perViewDS.get()};
            bakedCommandBuffer->bindDescriptorSets(EPBP_GRAPHICS,renderpassindependent->getLayout(),1u,1u,descriptorSets);
            for (auto& mb : *gpumeshes)
            {
                mb->setVertexBufferBinding({0ull,perInstancePointersScratch},15);
                bakedCommandBuffer->drawMeshBuffer(mb.get());
            }
        }
    }
    bakedCommandBuffer->end();


    CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> mouse;
    CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

    core::vectorSIMDf cameraPosition(0, 5, -10);
    matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60), float(WIN_W) / WIN_H, 2.f, 4000.f);
    Camera camera = Camera(cameraPosition, core::vectorSIMDf(0, 0, 0), projectionMatrix, 2.f, 1.f);
    
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

        // update camera, TODO: redo
        {
            const auto& viewProjectionMatrix = camera.getConcatenatedMatrix();
            std::array<core::matrix4SIMD,8u> data;
            for (auto i=0; i<data.size(); i++)
            {
                data[i].setTranslation(core::vectorSIMDf(0.f,i,0.f)*6.f);
                data[i] = core::concatenateBFollowedByA(viewProjectionMatrix,data[i]);
            }
            commandBuffer->updateBuffer(perViewPerInstanceDataScratch.get(),0ull,perViewPerInstanceDataScratch->getSize(),data.data());
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
            commandBuffer->setViewport(0u,1u,&viewport);

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

            commandBuffer->beginRenderPass(&beginInfo,nbl::asset::ESC_INLINE);
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

#if 0
int main2()
{

    refctd_dynamic_array<ModelData_t>* dummy0 = nullptr;
    refctd_dynamic_array<DrawData_t>* dummy1;
    
    auto instanceData = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ModelData_t>>(kInstanceCount);
    auto mbuff = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<core::smart_refctd_ptr<video::IGPUMeshBuffer> > >(kInstanceCount);
    
    //
    SBufferBinding<video::IGPUBuffer> globalVertexBindings[SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT];
    core::smart_refctd_ptr<video::IGPUBuffer> globalIndexBuffer,perDrawDataSSBO,indirectDrawSSBO,perInstanceDataSSBO;

    
    core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> gpuDrawDirectPipeline,gpuDrawIndirectPipeline;
	{
        DrawElementsIndirectCommand_t indirectDrawData[kInstanceCount];

        {
            size_t vertexSize = 0;
            std::vector<uint8_t> vertexData;
            std::vector<uint32_t> indexData;

            std::uniform_int_distribution<uint32_t> dist(16, 4*1024);
            for (size_t i=0; i<kInstanceCount; i++)
            {
                float poly = sqrtf(dist(mt))+0.5f;

                //
                indirectDrawData[i].count = sphereData.indexCount;
                indirectDrawData[i].instanceCount = 1;
                indirectDrawData[i].firstIndex = indexData.size();
                indirectDrawData[i].baseVertex = vertexData.size()/vertexSize;
                indirectDrawData[i].baseInstance = 0;

                //
                auto vdata = reinterpret_cast<const uint8_t*>(databuf->buffer->getPointer());
                vertexData.insert(vertexData.end(),vdata,vdata+vdatasize);

                auto idata = reinterpret_cast<const uint32_t*>(sphereData.indexBuffer.buffer->getPointer());
                indexData.insert(indexData.end(),idata,idata+sphereData.indexCount);
            }
            indirectDrawSSBO = driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(indirectDrawData), indirectDrawData);
            
            globalIndexBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(indexData.size()*sizeof(uint32_t),indexData.data());
            indexData.clear();

            globalVertexBindings[0] = { 0u,driver->createFilledDeviceLocalGPUBufferOnDedMem(vertexData.size(),vertexData.data()) };
            vertexData.clear();
        }
        
        //
        gpuDrawDirectPipeline = driver->getGPUObjectsFromAssets(&cpuDrawDirectPipeline.get(),&cpuDrawDirectPipeline.get()+1)->operator[](0);
        gpuDrawIndirectPipeline = driver->getGPUObjectsFromAssets(&cpuDrawIndirectPipeline.get(),&cpuDrawIndirectPipeline.get()+1)->operator[](0);

        std::uniform_real_distribution<float> dist3D(0.f,400.f);
        for (size_t i=0; i<kInstanceCount; i++)
        {
            auto& meshbuffer = mbuff->operator[](i);
            meshbuffer = core::make_smart_refctd_ptr<video::IGPUMeshBuffer>(
                core::smart_refctd_ptr(gpuDrawDirectPipeline),
                nullptr,
                globalVertexBindings,
                SBufferBinding<video::IGPUBuffer>{indirectDrawData[i].firstIndex*sizeof(uint32_t),core::smart_refctd_ptr(globalIndexBuffer)}
            );

            meshbuffer->setBaseVertex(indirectDrawData[i].baseVertex);
            meshbuffer->setIndexCount(indirectDrawData[i].count);
            meshbuffer->setIndexType(asset::EIT_32BIT);

            auto& instance = instanceData->operator[](i);
            meshbuffer->setBoundingBox({instance.bbox[0].getAsVector3df(),instance.bbox[1].getAsVector3df()});

            {
                float scale = dist3D(mt)*0.0025f+1.f;
                instance.worldMatrix.setScale(core::vectorSIMDf(scale,scale,scale));
            }
            instance.worldMatrix.setTranslation(core::vectorSIMDf(dist3D(mt),dist3D(mt),dist3D(mt)));
            instance.worldMatrix.getSub3x3InverseTranspose(instance.normalMatrix);
        }

        perInstanceDataSSBO = driver->createFilledDeviceLocalGPUBufferOnDedMem(instanceData->bytesize(),instanceData->data());
	}
    
	auto perDrawData = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<DrawData_t>>(kInstanceCount);
	perDrawDataSSBO = driver->createDeviceLocalGPUBufferOnDedMem(perDrawData->bytesize());
    
    // TODO: get rid of the `const_cast`s
    auto drawDirectLayout = const_cast<video::IGPUPipelineLayout*>(gpuDrawDirectPipeline->getLayout());
    auto drawIndirectLayout = const_cast<video::IGPUPipelineLayout*>(gpuDrawIndirectPipeline->getLayout());
    auto cullLayout = const_cast<video::IGPUPipelineLayout*>(gpuCullPipeline->getLayout());
    auto drawDirectDescriptorLayout = const_cast<video::IGPUDescriptorSetLayout*>(drawDirectLayout->getDescriptorSetLayout(1));
    auto drawIndirectDescriptorLayout = const_cast<video::IGPUDescriptorSetLayout*>(drawIndirectLayout->getDescriptorSetLayout(1));
    auto cullDescriptorLayout = const_cast<video::IGPUDescriptorSetLayout*>(cullLayout->getDescriptorSetLayout(1));
    auto drawDirectDescriptorSet = driver->createGPUDescriptorSet(core::smart_refctd_ptr<video::IGPUDescriptorSetLayout>(drawDirectDescriptorLayout));
    auto drawIndirectDescriptorSet = driver->createGPUDescriptorSet(core::smart_refctd_ptr<video::IGPUDescriptorSetLayout>(drawIndirectDescriptorLayout));
    auto cullDescriptorSet = driver->createGPUDescriptorSet(core::smart_refctd_ptr<video::IGPUDescriptorSetLayout>(cullDescriptorLayout));
    {
        constexpr auto BindingCount = 3u;
        video::IGPUDescriptorSet::SWriteDescriptorSet writes[BindingCount];
        video::IGPUDescriptorSet::SDescriptorInfo infos[BindingCount];
        for (auto i=0; i<BindingCount; i++)
        {
            writes[i].binding = i;
            writes[i].arrayElement = 0u;
            writes[i].count = 1u;
            writes[i].descriptorType = asset::EDT_STORAGE_BUFFER;
            writes[i].info = infos+i;
        }
        infos[0].desc = perDrawDataSSBO;
        infos[0].buffer = { 0u,perDrawDataSSBO->getSize() };
        infos[1].desc = indirectDrawSSBO;
        infos[1].buffer = { 0u,indirectDrawSSBO->getSize() };
        infos[2].desc = perInstanceDataSSBO;
        infos[2].buffer = { 0u,perInstanceDataSSBO->getSize() };

        writes[0].dstSet = drawDirectDescriptorSet.get();
        driver->updateDescriptorSets(1u,writes,0u,nullptr);

        writes[0].dstSet = drawIndirectDescriptorSet.get();
        driver->updateDescriptorSets(1u,writes,0u,nullptr);

        writes[0].dstSet = cullDescriptorSet.get();
        writes[1].dstSet = cullDescriptorSet.get();
        writes[2].dstSet = cullDescriptorSet.get();
        driver->updateDescriptorSets(BindingCount,writes,0u,nullptr);
    }
    

    auto smgr = device->getSceneManager();

	scene::ICameraSceneNode* camera =
		smgr->addCameraSceneNodeFPS(0,100.0f,0.01f);
	camera->setPosition(core::vector3df(-4,0,0));
	camera->setTarget(core::vector3df(0,0,0));
	camera->setNearValue(0.01f);
	camera->setFarValue(250.0f);
    smgr->setActiveCamera(camera);

    device->getCursorControl()->setVisible(false);




        
        core::matrix3x4SIMD normalMatrix;
        camera->getViewMatrix().getSub3x3InverseTranspose(normalMatrix);
        if (useDrawIndirect)
        {
            CullShaderData_t pc;
            pc.viewProjMatrix = camera->getConcatenatedMatrix();
            pc.viewInverseTransposeMatrix = normalMatrix;
            pc.maxDrawCount = kInstanceCount;
            pc.cull = doCulling ? 1u:0u;

            driver->bindComputePipeline(gpuCullPipeline.get());
            driver->bindDescriptorSets(video::EPBP_COMPUTE, gpuCullPipeline->getLayout(), 1u, 1u, &cullDescriptorSet.get(), nullptr);
            driver->pushConstants(gpuCullPipeline->getLayout(), asset::ICPUSpecializedShader::ESS_COMPUTE, 0u, sizeof(CullShaderData_t), &pc);
            driver->dispatch((kInstanceCount+kCullWorkgroupSize-1)/kCullWorkgroupSize,1u,1u);
            video::COpenGLExtensionHandler::extGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT|GL_COMMAND_BARRIER_BIT);

            driver->bindGraphicsPipeline(gpuDrawIndirectPipeline.get());
            driver->bindDescriptorSets(video::EPBP_GRAPHICS, gpuDrawIndirectPipeline->getLayout(), 1u, 1u, &drawIndirectDescriptorSet.get(), nullptr);
            driver->drawIndexedIndirect(globalVertexBindings,asset::EPT_TRIANGLE_LIST,asset::EIT_32BIT, globalIndexBuffer.get(),indirectDrawSSBO.get(),0,kInstanceCount,sizeof(DrawElementsIndirectCommand_t));
        }
        else
        {
            uint32_t unculledNum = 0u;
            uint32_t mb2draw[kInstanceCount];
            for (uint32_t i=0; i<kInstanceCount; i++)
            {
                const auto& instance = instanceData->operator[](i);

                auto& draw = perDrawData->operator[](i);
                draw.modelViewProjMatrix = core::concatenateBFollowedByA(camera->getConcatenatedMatrix(), instance.worldMatrix);
                if (doCulling)
                {
                    core::aabbox3df bbox(instance.bbox[0].getAsVector3df(), instance.bbox[1].getAsVector3df());
                    if (!draw.modelViewProjMatrix.isBoxInFrustum(bbox))
                        continue;
                }

                draw.normalMatrix = core::concatenateBFollowedByA(normalMatrix,instance.normalMatrix);
                mb2draw[unculledNum++] = i;
            }
            driver->updateBufferRangeViaStagingBuffer(perDrawDataSSBO.get(),0u,perDrawData->bytesize(),perDrawData->data());

            driver->bindGraphicsPipeline(gpuDrawDirectPipeline.get());
            driver->bindDescriptorSets(video::EPBP_GRAPHICS, gpuDrawDirectPipeline->getLayout(), 1u, 1u, &drawDirectDescriptorSet.get(), nullptr);
            for (uint32_t i=0; i<unculledNum; i++)
            {
                driver->pushConstants(gpuDrawDirectPipeline->getLayout(),asset::ICPUSpecializedShader::ESS_VERTEX,0u,sizeof(uint32_t),mb2draw+i);
                driver->drawMeshBuffer(mbuff->operator[](mb2draw[i]).get());
            }
        }

#endif