// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/Camera.hpp"
#include "../common/CommonAPI.h"
#include "nbl/ext/ScreenShot/ScreenShot.h"

using namespace nbl;
using namespace core;
using namespace system;

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

    core::smart_refctd_ptr<video::IGPUFence> gpuTransferFence;
    core::smart_refctd_ptr<video::IGPUFence> gpuComputeFence;
    nbl::video::IGPUObjectFromAssetConverter cpu2gpu;
    {
        cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].fence = &gpuTransferFence;
        cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].fence = &gpuComputeFence;
    }

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
#if 0
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
#endif
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
            //commandBuffer->executeCommands(1u,&bakedCommandBuffer.get());
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


    auto* am = device->getAssetManager();
    video::IVideoDriver* driver = device->getVideoDriver();

    IAssetLoader::SAssetLoadParams lp;
    auto cullShaderBundle = am->getAsset("../boxFrustCull.comp", lp);
    auto vertexShaderBundleMDI = am->getAsset("../meshGPU.vert", lp);
    auto vertexShaderBundle = am->getAsset("../meshCPU.vert", lp);
    auto fragShaderBundle = am->getAsset("../mesh.frag", lp);

    CShaderIntrospector introspector(am->getGLSLCompiler());
    const auto extensions = driver->getSupportedGLSLExtensions();
    auto cpuCullPipeline = introspector.createApproximateComputePipelineFromIntrospection(IAsset::castDown<ICPUSpecializedShader>(cullShaderBundle.getContents().begin()->get()), extensions->begin(), extensions->end());
    auto gpuCullPipeline = driver->getGPUObjectsFromAssets(&cpuCullPipeline.get(),&cpuCullPipeline.get()+1)->operator[](0);

    ICPUSpecializedShader* shaders[2] = { IAsset::castDown<ICPUSpecializedShader>(vertexShaderBundle.getContents().begin()->get()),IAsset::castDown<ICPUSpecializedShader>(fragShaderBundle.getContents().begin()->get()) };
    auto cpuDrawDirectPipeline = introspector.createApproximateRenderpassIndependentPipelineFromIntrospection(shaders, shaders+2, extensions->begin(), extensions->end());
    shaders[0] = IAsset::castDown<ICPUSpecializedShader>(vertexShaderBundleMDI.getContents().begin()->get());
    auto cpuDrawIndirectPipeline = introspector.createApproximateRenderpassIndependentPipelineFromIntrospection(shaders, shaders+2, extensions->begin(), extensions->end());

    auto* fs = am->getFileSystem();
    
    auto* qnc = am->getMeshManipulator()->getQuantNormalCache();
    //loading cache from file
    if (!qnc->loadCacheFromFile<asset::EF_A2B10G10R10_SNORM_PACK32>(fs, "../../tmp/normalCache101010.sse"))
        os::Printer::log("Failed to load cache.");

    constexpr auto kInstanceCount = 8192;
    constexpr auto  kTotalTriangleLimit = 64*1024*1024;

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

        std::random_device rd;
        std::mt19937 mt(rd());
        {
            size_t vertexSize = 0;
            std::vector<uint8_t> vertexData;
            std::vector<uint32_t> indexData;

            std::uniform_int_distribution<uint32_t> dist(16, 4*1024);
            for (size_t i=0; i<kInstanceCount; i++)
            {
                float poly = sqrtf(dist(mt))+0.5f;
                const auto& sphereData = am->getGeometryCreator()->createSphereMesh(2.f,poly,poly);

                //some assumptions about generated mesh
                assert(sphereData.assemblyParams.primitiveType==asset::EPT_TRIANGLE_LIST);
                assert(sphereData.indexType==asset::EIT_32BIT);
                assert(sphereData.indexBuffer.offset==0);

                assert(sphereData.inputParams.enabledBindingFlags&0x1u); //helpful assumption

                auto& instance = instanceData->operator[](i);
                instance.bbox[0].set(sphereData.bbox.MinEdge);
                instance.bbox[1].set(sphereData.bbox.MaxEdge);

                const SBufferBinding<ICPUBuffer>* databuf = nullptr;
                // TODO: add asserts about the vertex attributes and bindings
                for (size_t j=0; j<SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; j++)
                if ((sphereData.inputParams.enabledBindingFlags>>j)&0x1u)
                {
                    if (databuf)
                        assert(databuf->operator==(sphereData.bindings[j])); // all sphere vertex data will be packed in the same buffer
                    else
                        databuf = sphereData.bindings+j;

                    if (vertexSize)
                        assert(sphereData.inputParams.bindings[j].stride == vertexSize); //all data in the same buffer == same vertex stride for all attributes
                    else
                        vertexSize = sphereData.inputParams.bindings[j].stride;
                }
                //
                if (i==0ull)
                {
                    auto enabledAttribBackupDirect = cpuDrawDirectPipeline->getVertexInputParams().enabledAttribFlags;
                    auto enabledAttribBackupIndirect = cpuDrawIndirectPipeline->getVertexInputParams().enabledAttribFlags;
                    cpuDrawDirectPipeline->getVertexInputParams() = sphereData.inputParams;
                    cpuDrawIndirectPipeline->getVertexInputParams() = sphereData.inputParams;
                    cpuDrawDirectPipeline->getVertexInputParams().enabledAttribFlags = enabledAttribBackupDirect;
                    cpuDrawIndirectPipeline->getVertexInputParams().enabledAttribFlags = enabledAttribBackupIndirect;

                    cpuDrawDirectPipeline->getPrimitiveAssemblyParams() = sphereData.assemblyParams;
                    cpuDrawIndirectPipeline->getPrimitiveAssemblyParams() = sphereData.assemblyParams;
                }

                //
                auto vdatasize = core::roundUp(databuf->buffer->getSize(),vertexSize);

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
        
        //! cache results -- speeds up mesh generation on second run
        qnc->saveCacheToFile<asset::EF_A2B10G10R10_SNORM_PACK32>(fs, "../../tmp/normalCache101010.sse");
        
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


	uint64_t lastFPSTime = 0;
	float lastFastestMeshFrameNr = -1.f;

	while(device->run() && receiver.keepOpen())
	//if (device->isWindowActive())
	{
		driver->beginScene(true, true, video::SColor(255,255,255,255) );

        //! This animates (moves) the camera and sets the transforms
        camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
        camera->render();
        
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

		driver->endScene();

		// display frames per second in window title
		uint64_t time = device->getTimer()->getRealTime();
		if (time-lastFPSTime > 1000)
		{
			std::wostringstream str;
			str << L"MultiDrawIndirect Benchmark - Irrlicht Engine [" << driver->getName() << "] FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(str.str());
			lastFPSTime = time;
		}
	}

	return 0;
}
#endif