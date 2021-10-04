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

enum E_GEOM_TYPE
{
    EGT_CUBE,
    EGT_SPHERE,
    EGT_CONE
};
template<E_GEOM_TYPE geom, uint32_t LoDLevels>
auto addLoDTable(
    IAssetManager* assetManager, core::smart_refctd_ptr<ICPUDescriptorSetLayout>&& cpuPerViewDSLayout,
    const core::smart_refctd_ptr<ICPUSpecializedShader>* shaders,
    nbl::video::IGPUObjectFromAssetConverter& cpu2gpu,
    nbl::video::IGPUObjectFromAssetConverter::SParams& cpu2gpuParams,
    const SBufferRange<video::IGPUBuffer>& perInstanceRedirectAttribs,
    const core::smart_refctd_ptr<video::IGPURenderpass>& renderpass
)
{
    constexpr auto perInstanceRedirectAttrID = 15u;
    auto* const geometryCreator = assetManager->getGeometryCreator();
    auto* const meshManipulator = assetManager->getMeshManipulator();

    core::smart_refctd_ptr<ICPURenderpassIndependentPipeline> cpupipeline;
    core::smart_refctd_ptr<ICPUMeshBuffer> cpumeshes[LoDLevels];
    for (uint32_t poly=4u,lod=0u; lod<LoDLevels; lod++)
    {
        IGeometryCreator::return_type geomData;
        switch (geom)
        {
            case EGT_CUBE:
                geomData = geometryCreator->createCubeMesh(core::vector3df(2.f));
                break;
            case EGT_SPHERE:
                geomData = geometryCreator->createSphereMesh(2.f,poly,poly,meshManipulator);
                break;
            case EGT_CONE:
                geomData = geometryCreator->createConeMesh(2.f,2.f,poly,0x0u,0x0u,0.f,meshManipulator);
                break;
            default:
                assert(false);
                break;
        }
        // we'll stick instance data refs in the last attribute binding
        assert((geomData.inputParams.enabledBindingFlags>>perInstanceRedirectAttrID)==0u);

        geomData.inputParams.enabledAttribFlags |= 0x1u<<perInstanceRedirectAttrID;
        geomData.inputParams.enabledBindingFlags |= 0x1u<<perInstanceRedirectAttrID;
        geomData.inputParams.attributes[perInstanceRedirectAttrID].binding = perInstanceRedirectAttrID;
        geomData.inputParams.attributes[perInstanceRedirectAttrID].relativeOffset = 0u;
        geomData.inputParams.attributes[perInstanceRedirectAttrID].format = asset::EF_R32G32_UINT;
        geomData.inputParams.bindings[perInstanceRedirectAttrID].inputRate = asset::EVIR_PER_INSTANCE;
        geomData.inputParams.bindings[perInstanceRedirectAttrID].stride = asset::getTexelOrBlockBytesize(asset::EF_R32G32_UINT);

        if (!cpupipeline)
        {
            auto pipelinelayout = core::make_smart_refctd_ptr<ICPUPipelineLayout>(nullptr,nullptr,nullptr,std::move(cpuPerViewDSLayout));
            cpupipeline = core::make_smart_refctd_ptr<ICPURenderpassIndependentPipeline>(
                std::move(pipelinelayout),&shaders->get(),&shaders->get()+2u,
                geomData.inputParams,SBlendParams{},geomData.assemblyParams,SRasterizationParams{}
            );
        }
        cpumeshes[lod] = core::make_smart_refctd_ptr<ICPUMeshBuffer>();
        cpumeshes[lod]->setPipeline(core::smart_refctd_ptr(cpupipeline));
        cpumeshes[lod]->setIndexType(geomData.indexType);
        cpumeshes[lod]->setIndexCount(geomData.indexCount);
        cpumeshes[lod]->setIndexBufferBinding(std::move(geomData.indexBuffer));
        for (auto j=0u; j<ICPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT; j++)
            cpumeshes[lod]->setVertexBufferBinding(asset::SBufferBinding(geomData.bindings[j]),j);

        poly <<= 1u;
    }
#if 0
            {
                constexpr auto indicesPerBatch = 1023u;
                auto i = 0u;
                for (; i<sphereData.indexCount; i+=indicesPerBatch)
                {
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


                }
                maxInstancedDrawcalls = i*MaxInstanceCount;
            }
#endif
    auto gpumeshes = cpu2gpu.getGPUObjectsFromAssets(cpumeshes,cpumeshes+LoDLevels,cpu2gpuParams);
    std::pair<core::smart_refctd_ptr<video::IGPUGraphicsPipeline>,core::vector<core::smart_refctd_ptr<video::IGPUMeshBuffer>>> retval;
    for (auto gpumb : *gpumeshes)
    {
        {
            auto& mb = retval.second.emplace_back();
            mb = core::make_smart_refctd_ptr<video::IGPUMeshBuffer>();
            mb->setPipeline(core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline>(const_cast<video::IGPURenderpassIndependentPipeline*>(gpumb->getPipeline())));
            mb->setIndexType(gpumb->getIndexType());
            mb->setIndexCount(gpumb->getIndexCount());
            SBufferBinding<video::IGPUBuffer> indexBinding = {gpumb->getIndexBufferBinding().offset,gpumb->getIndexBufferBinding().buffer};
            mb->setIndexBufferBinding(std::move(indexBinding));
            for (auto j=0u; j<video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT; j++)
            {
                auto vertexBinding = gpumb->getVertexBufferBindings()[j];
                mb->setVertexBufferBinding(std::move(vertexBinding),j);
            }
            mb->setVertexBufferBinding({perInstanceRedirectAttribs.offset,perInstanceRedirectAttribs.buffer},perInstanceRedirectAttrID);
            // TODO
            //IMeshManipulator::recalculateBoundingBox(mb.get());

            // TODO: undo this
            mb->setInstanceCount(1u);
            mb->setBaseInstance(retval.second.size()-1u);
        }
    }
    {
        video::IGPUGraphicsPipeline::SCreationParams params;
        params.renderpass = renderpass;
        params.renderpassIndependent = core::smart_refctd_ptr_dynamic_cast<video::IGPURenderpassIndependentPipeline>(assetManager->findGPUObject(cpupipeline.get()));
        params.subpassIx = 0u;
        retval.first = cpu2gpuParams.device->createGPUGraphicsPipeline(nullptr,std::move(params));
    }
    cpu2gpuParams.waitForCreationToComplete();
    return retval;
}

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


    //
    constexpr auto MaxPipelines = 2u;
    constexpr auto MaxDrawCalls = 2048u;
    constexpr auto MaxInstanceCount = 8u;
    constexpr auto MaxTotalDrawcallInstances = 2048u;
    
    core::smart_refctd_ptr<video::CDrawIndirectAllocator<>> drawIndirectAllocator;
    {
        video::IDrawIndirectAllocator::ImplicitBufferCreationParameters drawAllocatorParams;
        drawAllocatorParams.device = logicalDevice.get();
        drawAllocatorParams.maxDrawCommandStride = sizeof(asset::DrawElementsIndirectCommand_t);
        drawAllocatorParams.drawCommandCapacity = MaxDrawCalls;
        drawAllocatorParams.drawCountCapacity = MaxPipelines;
        drawIndirectAllocator = video::CDrawIndirectAllocator<>::create(std::move(drawAllocatorParams));
    }

    using lod_library_t = scene::ILevelOfDetailLibrary;
    //auto lodLibrary = lod_library_t::create();

    using culling_system_t = scene::ICullingLoDSelectionSystem;
    core::smart_refctd_ptr<culling_system_t> cullingSystem;
    culling_system_t::Params cullingParams;
    {
        constexpr auto LayoutCount = 4u;
        core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> layouts[LayoutCount] =
        {
            scene::ILevelOfDetailLibrary::createDescriptorSetLayout(logicalDevice.get()),
            culling_system_t::createInputDescriptorSetLayout(logicalDevice.get()),
            culling_system_t::createOutputDescriptorSetLayout(logicalDevice.get(),true),
            [&]() -> core::smart_refctd_ptr<video::IGPUDescriptorSetLayout>
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
                return logicalDevice->createGPUDescriptorSetLayout(bindings,bindings+BindingCount);
            }()
        };
        auto pool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE,&layouts->get(),&layouts->get()+LayoutCount);
        
        cullingSystem = core::make_smart_refctd_ptr<culling_system_t>(logicalDevice.get(),core::smart_refctd_ptr(layouts[3]));

        cullingParams.indirectDispatchParams = {0ull,culling_system_t::createDispatchIndirectBuffer(utilities.get(),queues[decltype(initOutput)::EQT_TRANSFER_UP])};
        // TODO: add the rest of the buffers
        {
            video::IGPUBuffer::SCreationParams params;
            params.usage = asset::IBuffer::EUF_STORAGE_BUFFER_BIT;
            cullingParams.instanceList = {0ull,~0ull,logicalDevice->createDeviceLocalGPUBufferOnDedMem(params,sizeof(culling_system_t::InstanceToCull)*MaxInstanceCount)};
        }
        cullingParams.scratchBufferRanges = culling_system_t::createScratchBuffer(logicalDevice.get(),MaxInstanceCount,MaxTotalDrawcallInstances);
        cullingParams.drawCalls = drawIndirectAllocator->getDrawCommandMemoryBlock();
        cullingParams.perViewPerInstance = {0ull,~0ull,culling_system_t::createPerViewPerInstanceDataBuffer<core::matrix4SIMD>(logicalDevice.get(),MaxTotalDrawcallInstances)}; // TODO: perView type
        cullingParams.perInstanceRedirectAttribs = {0ul,~0ull,culling_system_t::createInstanceRedirectBuffer(logicalDevice.get(),MaxTotalDrawcallInstances)};
        const auto drawCountsBlock = drawIndirectAllocator->getDrawCountMemoryBlock();
        if (drawCountsBlock)
            cullingParams.drawCounts = *drawCountsBlock;

        cullingParams.lodLibraryDS = nullptr;
        cullingParams.transientInputDS = culling_system_t::createInputDescriptorSet(
            logicalDevice.get(),pool.get(),std::move(layouts[1]),
            cullingParams.indirectDispatchParams,
            cullingParams.instanceList,
            cullingParams.scratchBufferRanges
        );
        cullingParams.transientOutputDS = culling_system_t::createOutputDescriptorSet(
            logicalDevice.get(),pool.get(),std::move(layouts[2]),
            cullingParams.drawCalls,
            cullingParams.perViewPerInstance,
            cullingParams.perInstanceRedirectAttribs,
            cullingParams.drawCounts
        );
        cullingParams.customDS = logicalDevice->createGPUDescriptorSet(pool.get(),std::move(layouts[3]));

        cullingParams.indirectDispatchParams.buffer->setObjectDebugName("CullingIndirect");
        cullingParams.drawCalls.buffer->setObjectDebugName("DrawCallPool");
        cullingParams.scratchBufferRanges.pvsInstanceDraws.buffer->setObjectDebugName("PotentiallyVisibleDrawInstances");
        cullingParams.perInstanceRedirectAttribs.buffer->setObjectDebugName("PerInstanceInputAttribs");
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

    //TODO: delete this
    struct PotentiallyVisibleInstanceDraw
    {
        uint32_t drawBaseInstanceDWORDOffset;
        uint32_t instanceID;
        uint32_t instanceGUID;
        uint32_t perViewPerInstanceID;
    };
    core::vector<PotentiallyVisibleInstanceDraw> pvsContents(1u);
    auto& pvsCount = pvsContents[0].drawBaseInstanceDWORDOffset;


    uint32_t multiDrawCommandRangeByteOffsets[MaxPipelines];
    uint32_t multiDrawCommandMaxCounts[MaxPipelines] = { 0u };
    uint32_t multiDrawCommandCounts[MaxPipelines];
    video::IDrawIndirectAllocator::Allocation mdiAlloc;
    mdiAlloc.count = 0u;
    mdiAlloc.multiDrawCommandRangeByteOffsets = multiDrawCommandRangeByteOffsets;
    mdiAlloc.multiDrawCommandMaxCounts = multiDrawCommandMaxCounts;
    mdiAlloc.multiDrawCommandCounts = multiDrawCommandCounts;
    mdiAlloc.setAllCommandStructSizesConstant(sizeof(asset::DrawElementsIndirectCommand_t));
    core::vector<asset::DrawElementsIndirectCommand_t> drawCallData;

    // TODO: decouple
    core::smart_refctd_ptr<video::IGPUCommandBuffer> bakedCommandBuffer;
    {
        // TODO: turn into a lambda (so we can have multiple geometry types, sphere (multi lod, multi meshlet), cone (multi lod, single meshlet), cube (no lod), etc.)
        const uint32_t pplnIndex = mdiAlloc.count++;
        mdiAlloc.multiDrawCommandRangeByteOffsets[pplnIndex] = video::IDrawIndirectAllocator::invalid_draw_range_begin;
        mdiAlloc.multiDrawCommandCounts[pplnIndex] = video::IDrawIndirectAllocator::invalid_draw_count_ix;

        logicalDevice->createCommandBuffers(commandPool.get(),video::IGPUCommandBuffer::EL_SECONDARY,1u,&bakedCommandBuffer);
        bakedCommandBuffer->begin(video::IGPUCommandBuffer::EU_RENDER_PASS_CONTINUE_BIT|video::IGPUCommandBuffer::EU_SIMULTANEOUS_USE_BIT);
        //
        {
            auto* qnc = assetManager->getMeshManipulator()->getQuantNormalCache();
            //loading cache from file
            const system::path cachePath = std::filesystem::current_path()/"../../tmp/normalCache101010.sse";
            if (!qnc->loadCacheFromFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(),cachePath))
                logger->log("%s",ILogger::ELL_ERROR,"Failed to load cache.");

            auto retval = addLoDTable<EGT_SPHERE,7>(
                assetManager.get(),core::smart_refctd_ptr(cpuPerViewDSLayout),shaders,
                cpu2gpu,cpu2gpuParams,cullingParams.perInstanceRedirectAttribs,renderpass
            );
#if 0
                    drawCallData.emplace_back(
                        mb->getIndexCount(),
                        mb->getInstanceCount(),
                        mb->getIndexBufferBinding().offset/sizeofIndex,
                        mb->getBaseVertex(),
                        mb->getBaseInstance()
                    );
                    multiDrawCommandMaxCounts[pplnIndex]++;
#endif
            //! cache results -- speeds up mesh generation on second run
            qnc->saveCacheToFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(),cachePath);

            // TODO: refactor
            bakedCommandBuffer->bindGraphicsPipeline(retval.first.get());
            auto layout = retval.first->getRenderpassIndependentPipeline()->getLayout();
            const video::IGPUDescriptorSet* descriptorSets[1] = {perViewDS.get()};
            bakedCommandBuffer->bindDescriptorSets(EPBP_GRAPHICS,layout,1u,1u,descriptorSets);
            for (auto& mb : retval.second)
                bakedCommandBuffer->drawMeshBuffer(mb.get());
        }
        bakedCommandBuffer->end();

        const bool success = drawIndirectAllocator->allocateMultiDraws(mdiAlloc);
        assert(success);
        // TODO: get rid of this
    }


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

            cullingParams.cmdbuf = commandBuffer.get();
            cullingSystem->processInstancesAndFillIndirectDraws(cullingParams);
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