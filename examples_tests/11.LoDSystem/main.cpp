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
using namespace asset;
using namespace ui;

using lod_library_t = scene::CLevelOfDetailLibrary<>;
using culling_system_t = scene::ICullingLoDSelectionSystem;

struct LoDLibraryData
{
    core::vector<uint32_t> drawCallOffsetsIn20ByteStrides;
    core::vector<uint32_t> drawCountOffsets;
    core::vector<asset::DrawElementsIndirectCommand_t> drawCallData;
    core::vector<uint32_t> drawCountData;
    core::vector<uint32_t> lodInfoDstUvec2s;
    core::vector<uint32_t> lodTableDstUvec4s;
    scene::ILevelOfDetailLibrary::InfoContainerAdaptor<lod_library_t::LoDInfo> lodInfoData;
    scene::ILevelOfDetailLibrary::InfoContainerAdaptor<scene::ILevelOfDetailLibrary::LoDTableInfo> lodTableData;
};
enum E_GEOM_TYPE
{
    EGT_CUBE,
    EGT_SPHERE,
    EGT_CYLINDER,
    EGT_COUNT
};

template<E_GEOM_TYPE geom, uint32_t LoDLevels>
void addLoDTable(
    IAssetManager* assetManager,
    const core::smart_refctd_ptr<ICPUDescriptorSetLayout>& cpuTransformTreeDSLayout,
    const core::smart_refctd_ptr<ICPUDescriptorSetLayout>& cpuPerViewDSLayout,
    const core::smart_refctd_ptr<ICPUSpecializedShader>* shaders,
    nbl::video::IGPUObjectFromAssetConverter::SParams& cpu2gpuParams,
    LoDLibraryData& lodLibraryData,
    video::CDrawIndirectAllocator<>* drawIndirectAllocator,
    lod_library_t* lodLibrary,
    core::vector<video::CSubpassKiln::DrawcallInfo>& drawcallInfos,
    const SBufferRange<video::IGPUBuffer>& perInstanceRedirectAttribs,
    const core::smart_refctd_ptr<video::IGPURenderpass>& renderpass,
    const video::IGPUDescriptorSet* transformTreeDS,
    const core::smart_refctd_ptr<video::IGPUDescriptorSet>& perViewDS
)
{
    constexpr auto perInstanceRedirectAttrID = 15u;
    auto* const geometryCreator = assetManager->getGeometryCreator();
    auto* const meshManipulator = assetManager->getMeshManipulator();

    core::smart_refctd_ptr<ICPURenderpassIndependentPipeline> cpupipeline;
    core::smart_refctd_ptr<ICPUMeshBuffer> cpumeshes[LoDLevels];
    for (uint32_t poly = 4u, lod = 0u; lod < LoDLevels; lod++)
    {
        IGeometryCreator::return_type geomData;
        switch (geom)
        {
        case EGT_CUBE:
            geomData = geometryCreator->createCubeMesh(core::vector3df(2.f));
            break;
        case EGT_SPHERE:
            geomData = geometryCreator->createSphereMesh(2.f, poly, poly, meshManipulator);
            break;
        case EGT_CYLINDER:
            geomData = geometryCreator->createCylinderMesh(1.f, 4.f, poly, 0x0u, meshManipulator);
            break;
        default:
            assert(false);
            break;
        }
        // we'll stick instance data refs in the last attribute binding
        assert((geomData.inputParams.enabledBindingFlags >> perInstanceRedirectAttrID) == 0u);

        geomData.inputParams.enabledAttribFlags |= 0x1u << perInstanceRedirectAttrID;
        geomData.inputParams.enabledBindingFlags |= 0x1u << perInstanceRedirectAttrID;
        geomData.inputParams.attributes[perInstanceRedirectAttrID].binding = perInstanceRedirectAttrID;
        geomData.inputParams.attributes[perInstanceRedirectAttrID].relativeOffset = 0u;
        geomData.inputParams.attributes[perInstanceRedirectAttrID].format = asset::EF_R32G32_UINT;
        geomData.inputParams.bindings[perInstanceRedirectAttrID].inputRate = asset::EVIR_PER_INSTANCE;
        geomData.inputParams.bindings[perInstanceRedirectAttrID].stride = asset::getTexelOrBlockBytesize(asset::EF_R32G32_UINT);

        if (!cpupipeline)
        {
            auto pipelinelayout = core::make_smart_refctd_ptr<ICPUPipelineLayout>(
                nullptr, nullptr,
                core::smart_refctd_ptr(cpuTransformTreeDSLayout),
                core::smart_refctd_ptr(cpuPerViewDSLayout)
                );
            SRasterizationParams rasterParams = {};
            if (geom == EGT_CYLINDER)
                rasterParams.faceCullingMode = asset::EFCM_NONE;
            cpupipeline = core::make_smart_refctd_ptr<ICPURenderpassIndependentPipeline>(
                std::move(pipelinelayout), &shaders->get(), &shaders->get() + 2u,
                geomData.inputParams, SBlendParams{}, geomData.assemblyParams, rasterParams
                );
        }
        cpumeshes[lod] = core::make_smart_refctd_ptr<ICPUMeshBuffer>();
        cpumeshes[lod]->setPipeline(core::smart_refctd_ptr(cpupipeline));
        cpumeshes[lod]->setIndexType(geomData.indexType);
        cpumeshes[lod]->setIndexCount(geomData.indexCount);
        cpumeshes[lod]->setIndexBufferBinding(std::move(geomData.indexBuffer));
        for (auto j = 0u; j < ICPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT; j++)
            cpumeshes[lod]->setVertexBufferBinding(asset::SBufferBinding(geomData.bindings[j]), j);

        poly <<= 1u;
    }
    cpu2gpuParams.beginCommandBuffers();
    auto gpumeshes = video::CAssetPreservingGPUObjectFromAssetConverter().getGPUObjectsFromAssets(cpumeshes, cpumeshes+LoDLevels, cpu2gpuParams);

    core::smart_refctd_ptr<video::IGPUGraphicsPipeline> pipeline;
    {
        video::IGPUGraphicsPipeline::SCreationParams params;
        params.renderpass = renderpass;
        params.renderpassIndependent = core::smart_refctd_ptr_dynamic_cast<video::IGPURenderpassIndependentPipeline>(assetManager->findGPUObject(cpupipeline.get()));
        params.subpassIx = 0u;
        pipeline = cpu2gpuParams.device->createGPUGraphicsPipeline(nullptr, std::move(params));
    }

    auto drawcallInfosOutIx = drawcallInfos.size();
    drawcallInfos.resize(drawcallInfos.size() + gpumeshes->size());
    core::aabbox3df aabb(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
    lod_library_t::LoDInfo prevInfo;
    for (auto lod = 0u; lod < gpumeshes->size(); lod++)
    {
        auto gpumb = gpumeshes->operator[](lod);

        auto& di = drawcallInfos[drawcallInfosOutIx++];
        memcpy(di.pushConstantData, gpumb->getPushConstantsDataPtr(), video::IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE);
        di.pipeline = pipeline;
        std::fill_n(di.descriptorSets, video::IGPUPipelineLayout::DESCRIPTOR_SET_COUNT, nullptr);
        di.descriptorSets[0] = core::smart_refctd_ptr<const video::IGPUDescriptorSet>(transformTreeDS);
        di.descriptorSets[1] = perViewDS;
        di.indexType = gpumb->getIndexType();
        std::copy_n(gpumb->getVertexBufferBindings(), perInstanceRedirectAttrID, di.vertexBufferBindings);
        di.vertexBufferBindings[perInstanceRedirectAttrID] = { perInstanceRedirectAttribs.offset,perInstanceRedirectAttribs.buffer };
        di.indexBufferBinding = gpumb->getIndexBufferBinding().buffer;
        di.drawCommandStride = sizeof(asset::DrawElementsIndirectCommand_t);
        di.drawCountOffset = video::IDrawIndirectAllocator::invalid_draw_count_ix;
        di.drawCallOffset = video::IDrawIndirectAllocator::invalid_draw_range_begin;
        di.drawMaxCount = 0u;

        video::IDrawIndirectAllocator::Allocation mdiAlloc;
        mdiAlloc.count = 1u;
        mdiAlloc.multiDrawCommandRangeByteOffsets = &di.drawCallOffset;
        mdiAlloc.multiDrawCommandMaxCounts = &di.drawMaxCount;
        mdiAlloc.multiDrawCommandCountOffsets = &di.drawCountOffset;
        mdiAlloc.setAllCommandStructSizesConstant(di.drawCommandStride);

        // small enough batches that they could use 16-bit indices
        // the dispatcher gains above 4k triangles per batch are asymptotic
        // we could probably use smaller batches if we implemented drawcall compaction
        // (we wouldn't pay for the extra drawcalls generated by batches that have no instances)
        // if the culling system was to be used together with occlusion culling, we could use smaller batch sizes
        constexpr auto indicesPerBatch = 3u << 12u;
        const auto indexCount = gpumb->getIndexCount();
        const auto batchCount = di.drawMaxCount = (indexCount - 1u) / indicesPerBatch + 1u;
        lodLibraryData.drawCountData.emplace_back(di.drawMaxCount);

        const bool success = drawIndirectAllocator->allocateMultiDraws(mdiAlloc);
        assert(success);
        lodLibraryData.drawCountOffsets.emplace_back(di.drawCountOffset);
        di.drawCountOffset *= sizeof(uint32_t);

        //
        auto& lodInfo = lodLibraryData.lodInfoData.emplace_back(batchCount);
        if (lod)
        {
            lodInfo = lod_library_t::LoDInfo(batchCount, { 129600.f / exp2f(lod << 1) });
            if (!lodInfo.isValid(prevInfo))
            {
                assert(false && "THE LEVEL OF DETAIL CHOICE PARAMS NEED TO BE MONOTONICALLY DECREASING");
                exit(0x45u);
            }
        }
        else
            lodInfo = lod_library_t::LoDInfo(batchCount, { 2250000.f });
        prevInfo = lodInfo;
        //
        size_t indexSize;
        switch (gpumb->getIndexType())
        {
        case EIT_16BIT:
            indexSize = sizeof(uint16_t);
            break;
        case EIT_32BIT:
            indexSize = sizeof(uint32_t);
            break;
        default:
            assert(false);
            break;
        }
        auto batchID = 0u;
        for (auto i = 0u; i < indexCount; i += indicesPerBatch, batchID++)
        {
            auto& drawCallData = lodLibraryData.drawCallData.emplace_back();

            drawCallData.count = core::min(indexCount-i,indicesPerBatch);
            drawCallData.instanceCount = 0u;
            drawCallData.firstIndex = gpumb->getIndexBufferBinding().offset/indexSize+i;
            drawCallData.baseVertex = 0u;
            drawCallData.baseInstance = 0xdeadbeefu; // set to garbage to test the prefix sum
           
            lodLibraryData.drawCallOffsetsIn20ByteStrides.emplace_back(di.drawCallOffset / di.drawCommandStride + batchID);

            core::aabbox3df batchAABB;
            {
                // temporarily change the base vertex and index count to make AABB computation easier
                auto mb = cpumeshes[lod].get();
                auto oldBinding = mb->getIndexBufferBinding();
                const auto oldIndexCount = mb->getIndexCount();
                mb->setIndexBufferBinding({ oldBinding.offset + i * indexSize,oldBinding.buffer });
                mb->setIndexCount(drawCallData.count);
                batchAABB = IMeshManipulator::calculateBoundingBox(mb);
                mb->setIndexCount(oldIndexCount);
                mb->setIndexBufferBinding(std::move(oldBinding));
            }
            aabb.addInternalBox(batchAABB);

            const uint32_t drawCallDWORDOffset = (di.drawCallOffset + batchID * di.drawCommandStride) / sizeof(uint32_t);
            lodInfo.drawcallInfos[batchID] = scene::ILevelOfDetailLibrary::DrawcallInfo(
                drawCallDWORDOffset, batchAABB
            );
        }
    }
    auto& lodTable = lodLibraryData.lodTableData.emplace_back(LoDLevels);
    lodTable = scene::ILevelOfDetailLibrary::LoDTableInfo(LoDLevels, aabb);
    std::fill_n(lodTable.leveInfoUvec2Offsets, LoDLevels, scene::ILevelOfDetailLibrary::invalid);
    {
        lod_library_t::Allocation::LevelInfoAllocation lodLevelAllocations[1] =
        {
            lodTable.leveInfoUvec2Offsets,
            lodLibraryData.drawCountData.data() + lodLibraryData.drawCountData.size() - LoDLevels
        };
        uint32_t lodTableOffsets[1u] = { scene::ILevelOfDetailLibrary::invalid };
        const uint32_t lodLevelCounts[1u] = { LoDLevels };
        //
        lod_library_t::Allocation alloc = {};
        {
            alloc.count = 1u;
            alloc.tableUvec4Offsets = lodTableOffsets;
            alloc.levelCounts = lodLevelCounts;
            alloc.levelAllocations = lodLevelAllocations;
        }
        const bool success = lodLibrary->allocateLoDs(alloc);
        assert(success);
        for (auto i = 0u; i < scene::ILevelOfDetailLibrary::LoDTableInfo::getSizeInAlignmentUnits(LoDLevels); i++)
            lodLibraryData.lodTableDstUvec4s.push_back(lodTableOffsets[0] + i);
        for (auto lod = 0u; lod < LoDLevels; lod++)
        {
            const auto drawcallCount = lodLevelAllocations[0].drawcallCounts[lod];
            const auto offset = lodLevelAllocations[0].levelUvec2Offsets[lod];
            for (auto i = 0u; i < lod_library_t::LoDInfo::getSizeInAlignmentUnits(drawcallCount); i++)
                lodLibraryData.lodInfoDstUvec2s.push_back(offset + i);
        }
    }
    cpu2gpuParams.waitForCreationToComplete();
}

#include <random>
#include "assets/common.glsl"

class LoDSystemApp : public ApplicationBase
{
    _NBL_STATIC_INLINE_CONSTEXPR uint32_t WIN_W = 1600;
    _NBL_STATIC_INLINE_CONSTEXPR uint32_t WIN_H = 900;
    _NBL_STATIC_INLINE_CONSTEXPR uint32_t FBO_COUNT = 1u;
    _NBL_STATIC_INLINE_CONSTEXPR uint32_t FRAMES_IN_FLIGHT = 5u;
    static_assert(FRAMES_IN_FLIGHT > FBO_COUNT);

    // lod table entries
    _NBL_STATIC_INLINE_CONSTEXPR auto MaxDrawables = EGT_COUNT;
    // all the lod infos from all lod entries
    _NBL_STATIC_INLINE_CONSTEXPR auto MaxTotalLoDs = 8u * MaxDrawables;
    // how many contiguous ranges of drawcalls with explicit draw counts
    _NBL_STATIC_INLINE_CONSTEXPR auto MaxMDIs = 16u;
    // how many drawcalls (meshlets)
    _NBL_STATIC_INLINE_CONSTEXPR auto MaxDrawCalls = 256u;
    // how many instances
    _NBL_STATIC_INLINE_CONSTEXPR auto MaxInstanceCount = 1677721u; // absolute max for Intel HD Graphics on Windows (to keep within 128MB SSBO limit)
    // maximum visible instances of a drawcall (should be a sum of MaxLoDDrawcalls[t]*MaxInstances[t] where t iterates over all LoD Tables)
    _NBL_STATIC_INLINE_CONSTEXPR auto MaxTotalVisibleDrawcallInstances = MaxInstanceCount + (MaxInstanceCount >> 8u); // This is literally my worst case guess of how many batch-draw-instances there will be on screen at the same time

    public:
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
            return gl.get();
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
                fbos[i] = core::smart_refctd_ptr(f[i]);
            }
        }
        void setSwapchain(core::smart_refctd_ptr<video::ISwapchain>&& s) override
        {
            swapchain = std::move(s);
        }
        uint32_t getSwapchainImageCount() override
        {
            return FBO_COUNT;
        }
        virtual nbl::asset::E_FORMAT getDepthFormat() override
        {
            return nbl::asset::EF_D32_SFLOAT;
        }
        APP_CONSTRUCTOR(LoDSystemApp) 
        void onAppInitialized_impl() override
        {
            initOutput.window = core::smart_refctd_ptr(window);
            initOutput.system = core::smart_refctd_ptr(system);
            
            const auto swapchainImageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_COLOR_ATTACHMENT_BIT | asset::IImage::EUF_TRANSFER_DST_BIT);
            const video::ISurface::SFormat surfaceFormat(asset::EF_B8G8R8A8_SRGB, asset::ECP_COUNT, asset::EOTF_UNKNOWN);

            CommonAPI::InitWithDefaultExt(initOutput, video::EAT_OPENGL, "Level of Detail System", WIN_W, WIN_H, FBO_COUNT, swapchainImageUsage, surfaceFormat, asset::EF_D32_SFLOAT);
            window = std::move(initOutput.window);
            gl = std::move(initOutput.apiConnection);
            surface = std::move(initOutput.surface);
            gpuPhysicalDevice = std::move(initOutput.physicalDevice);
            logicalDevice = std::move(initOutput.logicalDevice);
            queues = std::move(initOutput.queues);
            swapchain = std::move(initOutput.swapchain);
            renderpass = std::move(initOutput.renderpass);
            fbos = std::move(initOutput.fbo);
            commandPools = std::move(initOutput.commandPools);
            assetManager = std::move(initOutput.assetManager);
            logger = std::move(initOutput.logger);
            inputSystem = std::move(initOutput.inputSystem);
            system = std::move(initOutput.system);
            windowCallback = std::move(initOutput.windowCb);
            cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
            utilities = std::move(initOutput.utilities);

            transferUpQueue = queues[CommonAPI::InitOutput::EQT_TRANSFER_UP];

            ttm = scene::ITransformTreeManager::create(utilities.get(),transferUpQueue);
            tt = scene::ITransformTreeWithNormalMatrices::create(logicalDevice.get(),MaxInstanceCount);
            const auto* ctt = tt.get(); // fight compiler, hard
            const video::IPropertyPool* nodePP = ctt->getNodePropertyPool();
            asset::SBufferRange<video::IGPUBuffer> nodeList;

            // Drawcall Allocator
            {
                video::IDrawIndirectAllocator::ImplicitBufferCreationParameters drawAllocatorParams;
                drawAllocatorParams.device = logicalDevice.get();
                drawAllocatorParams.maxDrawCommandStride = sizeof(asset::DrawElementsIndirectCommand_t);
                drawAllocatorParams.drawCommandCapacity = MaxDrawCalls;
                drawAllocatorParams.drawCountCapacity = MaxMDIs;
                drawIndirectAllocator = video::CDrawIndirectAllocator<>::create(std::move(drawAllocatorParams));
            }

            // LoD Library
            lodLibrary = lod_library_t::create({ logicalDevice.get(),MaxDrawables,MaxTotalLoDs,MaxDrawCalls });

            // Culling System
            core::smart_refctd_ptr<video::IDescriptorPool> cullingDSPool;
            cullPushConstants.instanceCount = 0u;
            {
                constexpr auto LayoutCount = 4u;
                core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> layouts[LayoutCount] =
                {
                    scene::ILevelOfDetailLibrary::createDescriptorSetLayout(logicalDevice.get()),
                    culling_system_t::createInputDescriptorSetLayout(logicalDevice.get()),
                    culling_system_t::createOutputDescriptorSetLayout(logicalDevice.get()),
                    [&]() -> core::smart_refctd_ptr<video::IGPUDescriptorSetLayout>
                    {
                        // TODO: figure out what should be here
                        constexpr auto BindingCount = 1u;
                        video::IGPUDescriptorSetLayout::SBinding bindings[BindingCount];
                        for (auto i = 0u; i < BindingCount; i++)
                        {
                            bindings[i].binding = i;
                            bindings[i].type = asset::EDT_STORAGE_BUFFER;
                            bindings[i].count = 1u;
                            bindings[i].stageFlags = asset::IShader::ESS_COMPUTE;
                            bindings[i].samplers = nullptr;
                        }
                        return logicalDevice->createGPUDescriptorSetLayout(bindings,bindings + BindingCount);
                    }()
                };
                cullingDSPool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, &layouts->get(), &layouts->get() + LayoutCount);

                const asset::SPushConstantRange range = { asset::IShader::ESS_COMPUTE,0u,sizeof(CullPushConstants_t) };
                cullingSystem = culling_system_t::create(
                    core::smart_refctd_ptr<video::CScanner>(utilities->getDefaultScanner()), &range, &range + 1u, core::smart_refctd_ptr(layouts[3]),
                    localInputCWD, "\n#include \"common.glsl\"\n", "\n#include \"cull_overrides.glsl\"\n"
                );

                cullingParams.indirectDispatchParams = { 0ull,culling_system_t::createDispatchIndirectBuffer(utilities.get(),transferUpQueue) };
                {
                    video::IGPUBuffer::SCreationParams params;
                    params.usage = asset::IBuffer::EUF_STORAGE_BUFFER_BIT;
                    
                    nodeList = {0ull,~0ull,logicalDevice->createDeviceLocalGPUBufferOnDedMem(params,sizeof(uint32_t)+sizeof(scene::ITransformTree::node_t)*MaxInstanceCount)};
                    nodeList.buffer->setObjectDebugName("transformTreeNodeList");
                    cullingParams.instanceList = {0ull,~0ull,logicalDevice->createDeviceLocalGPUBufferOnDedMem(params,sizeof(culling_system_t::InstanceToCull)*MaxInstanceCount)};
                }
                cullingParams.scratchBufferRanges = culling_system_t::createScratchBuffer(utilities->getDefaultScanner(), MaxInstanceCount, MaxTotalVisibleDrawcallInstances);
                cullingParams.drawCalls = drawIndirectAllocator->getDrawCommandMemoryBlock();
                cullingParams.perViewPerInstance = { 0ull,~0ull,culling_system_t::createPerViewPerInstanceDataBuffer<PerViewPerInstance_t>(logicalDevice.get(),MaxInstanceCount) };
                cullingParams.perInstanceRedirectAttribs = { 0ul,~0ull,culling_system_t::createInstanceRedirectBuffer(logicalDevice.get(),MaxTotalVisibleDrawcallInstances) };
                const auto drawCountsBlock = drawIndirectAllocator->getDrawCountMemoryBlock();
                if (drawCountsBlock)
                    cullingParams.drawCounts = *drawCountsBlock;

                cullingParams.lodLibraryDS = core::smart_refctd_ptr<video::IGPUDescriptorSet>(lodLibrary->getDescriptorSet());
                cullingParams.transientOutputDS = culling_system_t::createOutputDescriptorSet(
                    logicalDevice.get(), cullingDSPool.get(), std::move(layouts[2]),
                    cullingParams.drawCalls,
                    cullingParams.perViewPerInstance,
                    cullingParams.perInstanceRedirectAttribs,
                    cullingParams.drawCounts
                );
                cullingParams.customDS = logicalDevice->createGPUDescriptorSet(cullingDSPool.get(), std::move(layouts[3]));
                {
                    video::IGPUDescriptorSet::SWriteDescriptorSet write;
                    video::IGPUDescriptorSet::SDescriptorInfo info(nodePP->getPropertyMemoryBlock(scene::ITransformTree::global_transform_prop_ix));
                    write.dstSet = cullingParams.customDS.get();
                    write.binding = 0u;
                    write.arrayElement = 0u;
                    write.count = 1u;
                    write.descriptorType = EDT_STORAGE_BUFFER;
                    write.info = &info;
                    logicalDevice->updateDescriptorSets(1u, &write, 0u, nullptr);
                }

                cullingParams.indirectDispatchParams.buffer->setObjectDebugName("CullingIndirect");
                cullingParams.drawCalls.buffer->setObjectDebugName("DrawCallPool");
                cullingParams.perInstanceRedirectAttribs.buffer->setObjectDebugName("PerInstanceInputAttribs");
                if (cullingParams.drawCounts.buffer)
                    cullingParams.drawCounts.buffer->setObjectDebugName("DrawCountPool");
                cullingParams.perViewPerInstance.buffer->setObjectDebugName("DrawcallInstanceRedirects");
                cullingParams.indirectInstanceCull = false;
            }

            core::smart_refctd_ptr<ICPUSpecializedShader> shaders[2];
            {
                IAssetLoader::SAssetLoadParams lp;
                lp.workingDirectory = localInputCWD;
                lp.logger = logger.get();
                auto vertexShaderBundle = assetManager->getAsset("mesh.vert", lp);
                auto fragShaderBundle = assetManager->getAsset("mesh.frag", lp);
                shaders[0] = IAsset::castDown<ICPUSpecializedShader>(*vertexShaderBundle.getContents().begin());
                shaders[1] = IAsset::castDown<ICPUSpecializedShader>(*fragShaderBundle.getContents().begin());
            }

            core::smart_refctd_ptr<video::IGPUDescriptorSet> perViewDS;
            core::smart_refctd_ptr<ICPUDescriptorSetLayout> cpuPerViewDSLayout;
            {
                constexpr auto BindingCount = 1;
                ICPUDescriptorSetLayout::SBinding cpuBindings[BindingCount];
                for (auto i = 0; i < BindingCount; i++)
                {
                    cpuBindings[i].binding = i;
                    cpuBindings[i].count = 1u;
                    cpuBindings[i].stageFlags = IShader::ESS_VERTEX;
                    cpuBindings[i].samplers = nullptr;
                }
                cpuBindings[0].type = EDT_STORAGE_BUFFER;
                cpuPerViewDSLayout = core::make_smart_refctd_ptr<ICPUDescriptorSetLayout>(cpuBindings, cpuBindings + BindingCount);

                auto bindings = reinterpret_cast<video::IGPUDescriptorSetLayout::SBinding*>(cpuBindings);
                auto perViewDSLayout = logicalDevice->createGPUDescriptorSetLayout(bindings, bindings + BindingCount);
                auto dsPool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, &perViewDSLayout.get(), &perViewDSLayout.get() + 1u);
                perViewDS = logicalDevice->createGPUDescriptorSet(dsPool.get(), std::move(perViewDSLayout));
                {
                    video::IGPUDescriptorSet::SWriteDescriptorSet writes[BindingCount];
                    video::IGPUDescriptorSet::SDescriptorInfo infos[BindingCount];
                    for (auto i = 0; i < BindingCount; i++)
                    {
                        writes[i].dstSet = perViewDS.get();
                        writes[i].binding = i;
                        writes[i].arrayElement = 0u;
                        writes[i].count = 1u;
                        writes[i].info = infos + i;
                    }
                    writes[0].descriptorType = EDT_STORAGE_BUFFER;
                    infos[0].desc = cullingParams.perViewPerInstance.buffer;
                    infos[0].buffer = { 0u,video::IGPUDescriptorSet::SDescriptorInfo::SBufferInfo::WholeBuffer };
                    logicalDevice->updateDescriptorSets(BindingCount, writes, 0u, nullptr);
                }
            }

            // descset for global tform updates
            {
                auto layout = ttm->createRecomputeGlobalTransformsDescriptorSetLayout(logicalDevice.get());
                auto pool = logicalDevice->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE,&layout.get(),&layout.get()+1u);
                recomputeGlobalTransformsDS = logicalDevice->createGPUDescriptorSet(pool.get(),std::move(layout));
                ttm->updateRecomputeGlobalTransformsDescriptorSet(logicalDevice.get(),recomputeGlobalTransformsDS.get(),{0ull,nodeList.buffer});
            }

            std::mt19937 mt(0x45454545u);
            std::uniform_int_distribution<uint32_t> typeDist(0, EGT_COUNT - 1u);
            std::uniform_real_distribution<float> rotationDist(0, 2.f * core::PI<float>());
            std::uniform_real_distribution<float> posDist(-1200.f, 1200.f);
            {
                video::CSubpassKiln kiln;
                {
                    LoDLibraryData lodLibraryData;
                    uint32_t lodTables[EGT_COUNT];
                    // create all the LoDs of drawables
                    {
                        auto* qnc = assetManager->getMeshManipulator()->getQuantNormalCache();
                        //loading cache from file
                        const system::path cachePath = sharedOutputCWD / "normalCache101010.sse";
                        if (!qnc->loadCacheFromFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), cachePath))
                            int a = 0;// logger->log("%s", ILogger::ELL_ERROR, "Couldn't load cache.");

                        //
                        core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout> cpuTransformTreeDSLayout = scene::ITransformTreeWithNormalMatrices::createRenderDescriptorSetLayout();

                        // populating `lodTables` is a bit messy, I know
                        size_t lodTableIx = lodLibraryData.lodTableDstUvec4s.size();
                        addLoDTable<EGT_CUBE, 1>(
                            assetManager.get(), cpuTransformTreeDSLayout, cpuPerViewDSLayout, shaders, cpu2gpuParams,
                            lodLibraryData, drawIndirectAllocator.get(), lodLibrary.get(), kiln.getDrawcallMetadataVector(),
                            cullingParams.perInstanceRedirectAttribs, renderpass, ctt->getRenderDescriptorSet(), perViewDS
                            );
                        lodTables[EGT_CUBE] = lodLibraryData.lodTableDstUvec4s[lodTableIx];
                        lodTableIx = lodLibraryData.lodTableDstUvec4s.size();
                        addLoDTable<EGT_SPHERE, 7>(
                            assetManager.get(), cpuTransformTreeDSLayout, cpuPerViewDSLayout, shaders, cpu2gpuParams,
                            lodLibraryData, drawIndirectAllocator.get(), lodLibrary.get(), kiln.getDrawcallMetadataVector(),
                            cullingParams.perInstanceRedirectAttribs, renderpass, ctt->getRenderDescriptorSet(), perViewDS
                            );
                        lodTables[EGT_SPHERE] = lodLibraryData.lodTableDstUvec4s[lodTableIx];
                        lodTableIx = lodLibraryData.lodTableDstUvec4s.size();
                        addLoDTable<EGT_CYLINDER, 6>(
                            assetManager.get(), cpuTransformTreeDSLayout, cpuPerViewDSLayout, shaders, cpu2gpuParams,
                            lodLibraryData, drawIndirectAllocator.get(), lodLibrary.get(), kiln.getDrawcallMetadataVector(),
                            cullingParams.perInstanceRedirectAttribs, renderpass, ctt->getRenderDescriptorSet(), perViewDS
                            );
                        lodTables[EGT_CYLINDER] = lodLibraryData.lodTableDstUvec4s[lodTableIx];

                        //! cache results -- speeds up mesh generation on second run
                        qnc->saveCacheToFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), cachePath);
                    }
                    constexpr auto MaxTransfers = 8u;
                    video::CPropertyPoolHandler::UpStreamingRequest upstreamRequests[MaxTransfers];
                    // set up the instance list
                    core::vector<scene::ITransformTree::node_t> instanceGUIDs(
                        std::uniform_int_distribution<uint32_t>(MaxInstanceCount >> 1u, MaxInstanceCount)(mt), // Instance Count
                        scene::ITransformTree::invalid_node
                    );
                    const uint32_t objectCount = instanceGUIDs.size();
                    core::vector<core::matrix3x4SIMD> instanceTransforms(objectCount);
                    for (auto& tform : instanceTransforms)
                    {
                        tform.setRotation(core::quaternion(rotationDist(mt), rotationDist(mt), rotationDist(mt)));
                        tform.setTranslation(core::vectorSIMDf(posDist(mt), posDist(mt), posDist(mt)));
                    }
                    {
                        tt->allocateNodes({instanceGUIDs.data(),instanceGUIDs.data()+objectCount});

                        utilities->updateBufferRangeViaStagingBuffer(transferUpQueue,{0u,sizeof(uint32_t),nodeList.buffer},&objectCount);
                        utilities->updateBufferRangeViaStagingBuffer(transferUpQueue,{sizeof(uint32_t),objectCount*sizeof(scene::ITransformTree::node_t),nodeList.buffer},instanceGUIDs.data());

                        scene::ITransformTreeManager::UpstreamRequest request;
                        request.tree = tt.get();
                        request.parents = {}; // no parents
                        request.relativeTransforms.device2device = false;
                        request.relativeTransforms.data = instanceTransforms.data();
                        // TODO: make an `UpstreamRequest` which allows for sourcing the node list from GPUBuffer
                        request.nodes = {instanceGUIDs.data(),instanceGUIDs.data()+objectCount};
                        ttm->setupTransfers(request,upstreamRequests);

                        core::vector<culling_system_t::InstanceToCull> instanceList; instanceList.reserve(objectCount);
                        for (auto instanceGUID : instanceGUIDs)
                        {
                            auto& instance = instanceList.emplace_back();
                            instance.instanceGUID = instanceGUID;
                            instance.lodTableUvec4Offset = lodTables[typeDist(mt)];
                        }
                        utilities->updateBufferRangeViaStagingBuffer(transferUpQueue, { 0u,instanceList.size()*sizeof(culling_system_t::InstanceToCull),cullingParams.instanceList.buffer }, instanceList.data());

                        cullPushConstants.instanceCount += instanceList.size();
                    }
                    cullingParams.drawcallCount = lodLibraryData.drawCallData.size();
                    // do the transfer of drawcall and LoD data
                    {
                        constexpr auto TTMTransfers = scene::ITransformTreeManager::TransferCount;
                        for (auto i=TTMTransfers; i<MaxTransfers; i++)
                        {
                            upstreamRequests[i].fill = false;
                            upstreamRequests[i].source.device2device = false;
                            upstreamRequests[i].srcAddresses = nullptr; // iota 0,1,2,3,4,etc.
                        }
                        upstreamRequests[TTMTransfers + 0].destination = drawIndirectAllocator->getDrawCommandMemoryBlock();
                        upstreamRequests[TTMTransfers + 0].elementSize = sizeof(asset::DrawElementsIndirectCommand_t);
                        upstreamRequests[TTMTransfers + 0].elementCount = cullingParams.drawcallCount;
                        upstreamRequests[TTMTransfers + 0].source.data = lodLibraryData.drawCallData.data();
                        upstreamRequests[TTMTransfers + 0].dstAddresses = lodLibraryData.drawCallOffsetsIn20ByteStrides.data();
                        upstreamRequests[TTMTransfers + 1].destination = lodLibrary->getLoDInfoBinding();
                        upstreamRequests[TTMTransfers + 1].elementSize = alignof(lod_library_t::LoDInfo);
                        upstreamRequests[TTMTransfers + 1].elementCount = lodLibraryData.lodInfoDstUvec2s.size();
                        upstreamRequests[TTMTransfers + 1].source.data = lodLibraryData.lodInfoData.data();
                        upstreamRequests[TTMTransfers + 1].dstAddresses = lodLibraryData.lodInfoDstUvec2s.data();
                        upstreamRequests[TTMTransfers + 2].destination = lodLibrary->getLodTableInfoBinding();
                        upstreamRequests[TTMTransfers + 2].elementSize = alignof(scene::ILevelOfDetailLibrary::LoDTableInfo);
                        upstreamRequests[TTMTransfers + 2].elementCount = lodLibraryData.lodTableDstUvec4s.size();
                        upstreamRequests[TTMTransfers + 2].source.data = lodLibraryData.lodTableData.data();
                        upstreamRequests[TTMTransfers + 2].dstAddresses = lodLibraryData.lodTableDstUvec4s.data();
                        auto requestCount = TTMTransfers + 3u;
                        if (drawIndirectAllocator->getDrawCountMemoryBlock())
                        {
                            upstreamRequests[requestCount].destination = *drawIndirectAllocator->getDrawCountMemoryBlock();
                            upstreamRequests[requestCount].elementSize = sizeof(uint32_t);
                            upstreamRequests[requestCount].elementCount = lodLibraryData.drawCountOffsets.size();
                            upstreamRequests[requestCount].source.data = lodLibraryData.drawCountData.data();
                            upstreamRequests[requestCount].dstAddresses = lodLibraryData.drawCountOffsets.data();
                            requestCount++;
                        }

                        core::smart_refctd_ptr<video::IGPUCommandBuffer> tferCmdBuf;
                        logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_TRANSFER_UP].get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &tferCmdBuf);
                        auto fence = logicalDevice->createFence(video::IGPUFence::ECF_UNSIGNALED);
                        tferCmdBuf->begin(0u); // TODO some one time submit bit or something
                        {
                            auto ppHandler = utilities->getDefaultPropertyPoolHandler();
                            asset::SBufferBinding<video::IGPUBuffer> scratch;
                            {
                                video::IGPUBuffer::SCreationParams scratchParams = {};
                                scratchParams.canUpdateSubRange = true;
                                scratchParams.usage = core::bitflag(video::IGPUBuffer::EUF_TRANSFER_DST_BIT) | video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
                                scratch = { 0ull,logicalDevice->createDeviceLocalGPUBufferOnDedMem(scratchParams,ppHandler->getMaxScratchSize()) };
                                scratch.buffer->setObjectDebugName("Scratch Buffer");
                            }
                            auto* pRequests = upstreamRequests;
                            uint32_t waitSemaphoreCount = 0u;
                            video::IGPUSemaphore* const* waitSemaphores = nullptr;
                            const asset::E_PIPELINE_STAGE_FLAGS* waitStages = nullptr;
                            ppHandler->transferProperties(
                                utilities->getDefaultUpStreamingBuffer(), tferCmdBuf.get(), fence.get(), transferUpQueue, scratch,
                                pRequests, requestCount, waitSemaphoreCount, waitSemaphores, waitStages,
                                logger.get(), std::chrono::high_resolution_clock::time_point::max() // must finish
                            );
                        }
                        // also clear the scratch
                        cullingSystem->clearScratch(tferCmdBuf.get(),cullingParams.scratchBufferRanges.lodDrawCallCounts,cullingParams.scratchBufferRanges.prefixSumScratch);
                        tferCmdBuf->end();
                        {
                            video::IGPUQueue::SSubmitInfo submit = {}; // intializes all semaphore stuff to 0 and nullptr
                            submit.commandBufferCount = 1u;
                            submit.commandBuffers = &tferCmdBuf.get();
                            transferUpQueue->submit(1u, &submit, fence.get());
                        }
                        logicalDevice->blockForFences(1u, &fence.get());
                    }
                    // set up the remaining descriptor sets of the culling system
                    {
                        auto& drawCallOffsetsInDWORDs = lodLibraryData.drawCallOffsetsIn20ByteStrides;
                        for (auto i = 0u; i < cullingParams.drawcallCount; i++)
                            drawCallOffsetsInDWORDs[i] = lodLibraryData.drawCallOffsetsIn20ByteStrides[i] * sizeof(asset::DrawElementsIndirectCommand_t) / sizeof(uint32_t);
                        cullingParams.transientInputDS = culling_system_t::createInputDescriptorSet(
                            logicalDevice.get(), cullingDSPool.get(),
                            culling_system_t::createInputDescriptorSetLayout(logicalDevice.get()),
                            cullingParams.indirectDispatchParams,
                            cullingParams.instanceList,
                            cullingParams.scratchBufferRanges,
                            { 0ull,~0ull,utilities->createFilledDeviceLocalGPUBufferOnDedMem(transferUpQueue,cullingParams.drawcallCount * sizeof(uint32_t),drawCallOffsetsInDWORDs.data()) },
                            { 0ull,~0ull,utilities->createFilledDeviceLocalGPUBufferOnDedMem(transferUpQueue,lodLibraryData.drawCountOffsets.size() * sizeof(uint32_t),lodLibraryData.drawCountOffsets.data()) }
                        );
                    }
                }
                // prerecord the secondary cmdbuffer
                {
                    logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_GRAPHICS].get(), video::IGPUCommandBuffer::EL_SECONDARY, 1u, &bakedCommandBuffer);
                    bakedCommandBuffer->begin(video::IGPUCommandBuffer::EU_RENDER_PASS_CONTINUE_BIT | video::IGPUCommandBuffer::EU_SIMULTANEOUS_USE_BIT);
                    // TODO: handle teh offsets
                    auto drawCountBlock = drawIndirectAllocator->getDrawCountMemoryBlock();
                    kiln.bake(bakedCommandBuffer.get(), renderpass.get(), 0u, drawIndirectAllocator->getDrawCommandMemoryBlock().buffer.get(), drawIndirectAllocator->supportsMultiDrawIndirectCount() ? drawCountBlock->buffer.get():nullptr);
                    bakedCommandBuffer->end();
                }
            }

            core::vectorSIMDf cameraPosition(0, 5, -10);
            matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60.0f), float(WIN_W) / WIN_H, 2.f, 4000.f);
            {
                cullPushConstants.fovDilationFactor = decltype(lod_library_t::LoDInfo::choiceParams)::getFoVDilationFactor(projectionMatrix);
                // dilate by resolution as well, because the LoD distances were tweaked @ 720p
                cullPushConstants.fovDilationFactor *= float(window->getWidth() * window->getHeight()) / float(1280u * 720u);
            }
            camera = Camera(cameraPosition, core::vectorSIMDf(0, 0, 0), projectionMatrix, 2.f, 1.f);

            oracle.reportBeginFrameRecord();
            logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_GRAPHICS].get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, commandBuffers);

            for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
            {
                imageAcquire[i] = logicalDevice->createSemaphore();
                renderFinished[i] = logicalDevice->createSemaphore();
            }
        }

        void onAppTerminated_impl() override
        {
            lodLibrary->clear();
            drawIndirectAllocator->clear();

            const auto& fboCreationParams = fbos[acquiredNextFBO]->getCreationParameters();
            auto gpuSourceImageView = fboCreationParams.attachments[0];

            bool status = ext::ScreenShot::createScreenShot(
                logicalDevice.get(),
                queues[CommonAPI::InitOutput::EQT_TRANSFER_DOWN],
                renderFinished[resourceIx].get(),
                gpuSourceImageView.get(),
                assetManager.get(),
                "ScreenShot.png",
                asset::EIL_PRESENT_SRC,
                static_cast<asset::E_ACCESS_FLAGS>(0u));

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

            // update transforms
            {
                // buffers to barrier
                video::IGPUCommandBuffer::SBufferMemoryBarrier barriers[scene::ITransformTreeManager::SBarrierSuggestion::MaxBufferCount];
                auto setBufferBarrier = [&barriers,commandBuffer](const uint32_t ix, const asset::SBufferRange<video::IGPUBuffer>& range, const asset::SMemoryBarrier& barrier)
                {
                    barriers[ix].barrier = barrier;
                    barriers[ix].dstQueueFamilyIndex = barriers[ix].srcQueueFamilyIndex = commandBuffer->getQueueFamilyIndex();
                    barriers[ix].buffer = range.buffer;
                    barriers[ix].offset = range.offset;
                    barriers[ix].size = range.size;
                };
                
                // not going to barrier between last frame culling and rendering and TTM recompute, because its not in the same submit
                scene::ITransformTreeManager::BaseParams baseParams;
                baseParams.cmdbuf = commandBuffer.get();
                baseParams.tree = tt.get();
                baseParams.logger = logger.get();
                scene::ITransformTreeManager::DispatchParams dispatchParams;
                dispatchParams.indirect.buffer = nullptr;
                dispatchParams.direct.nodeCount = cullPushConstants.instanceCount;
                ttm->recomputeGlobalTransforms(baseParams,dispatchParams,recomputeGlobalTransformsDS.get());
                // barrier between TTM recompute and TTM recompute+culling system 
                {
                    const auto* ctt = tt.get();
                    auto node_pp = ctt->getNodePropertyPool();

                    auto sugg = scene::ITransformTreeManager::barrierHelper(scene::ITransformTreeManager::SBarrierSuggestion::EF_POST_GLOBAL_TFORM_RECOMPUTE);
                    sugg.dstStageMask |= asset::EPSF_VERTEX_SHADER_BIT; // also rendering shader (to read the normal matrix transforms)
                    uint32_t barrierCount = 0u;
                    setBufferBarrier(barrierCount++,node_pp->getPropertyMemoryBlock(scene::ITransformTree::global_transform_prop_ix),sugg.globalTransforms);
                    setBufferBarrier(barrierCount++,node_pp->getPropertyMemoryBlock(scene::ITransformTree::recomputed_stamp_prop_ix),sugg.recomputedTimestamps);
                    setBufferBarrier(barrierCount++,node_pp->getPropertyMemoryBlock(scene::ITransformTreeWithNormalMatrices::normal_matrix_prop_ix),sugg.normalMatrices);
                    commandBuffer->pipelineBarrier(sugg.srcStageMask,sugg.dstStageMask,asset::EDF_NONE,0u,nullptr,barrierCount,barriers,0u,nullptr);
                }
            }

            // cull, choose LoDs, and fill our draw indirects
            {
                const auto* layout = cullingSystem->getInstanceCullAndLoDSelectLayout();
                cullPushConstants.viewProjMat = camera.getConcatenatedMatrix();
                std::copy_n(camera.getPosition().pointer, 3u, cullPushConstants.camPos.comp);
                commandBuffer->pushConstants(layout, asset::IShader::ESS_COMPUTE, 0u, sizeof(cullPushConstants), &cullPushConstants);
                cullingParams.cmdbuf = commandBuffer.get();
                cullingSystem->processInstancesAndFillIndirectDraws(cullingParams,cullPushConstants.instanceCount);
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
                commandBuffer->executeCommands(1u, &bakedCommandBuffer.get());
                commandBuffer->endRenderPass();

                commandBuffer->end();
            }

            CommonAPI::Submit(logicalDevice.get(), swapchain.get(), commandBuffer.get(), queues[CommonAPI::InitOutput::EQT_GRAPHICS], imageAcquire[resourceIx].get(), renderFinished[resourceIx].get(), fence.get());
            CommonAPI::Present(logicalDevice.get(), swapchain.get(), queues[CommonAPI::InitOutput::EQT_GRAPHICS], renderFinished[resourceIx].get(), acquiredNextFBO);
        }

        bool keepRunning() override
        {
            return windowCallback->isWindowOpen();
        }

    private:

        CommonAPI::InitOutput initOutput;
        nbl::core::smart_refctd_ptr<nbl::ui::IWindow> window;
        nbl::core::smart_refctd_ptr<nbl::video::IAPIConnection> gl;
        nbl::core::smart_refctd_ptr<nbl::video::ISurface> surface;
        nbl::video::IPhysicalDevice* gpuPhysicalDevice;
        nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
        std::array<video::IGPUQueue*, CommonAPI::InitOutput::MaxQueuesCount> queues;
        nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
        nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
        std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, CommonAPI::InitOutput::MaxSwapChainImageCount> fbos;
    
        std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, CommonAPI::InitOutput::MaxQueuesCount> commandPools;
        nbl::core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
        nbl::core::smart_refctd_ptr<nbl::system::ILogger> logger;
        nbl::core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;
        nbl::core::smart_refctd_ptr<nbl::system::ISystem> system;
        nbl::core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCallback;
        nbl::video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
        nbl::core::smart_refctd_ptr<nbl::video::IUtilities> utilities;  

        nbl::video::IGPUQueue* transferUpQueue = nullptr;
        nbl::core::smart_refctd_ptr<nbl::scene::ITransformTreeManager> ttm;
        nbl::core::smart_refctd_ptr<nbl::scene::ITransformTree> tt;
        nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> recomputeGlobalTransformsDS;

        Camera camera = Camera(vectorSIMDf(0, 0, 0), vectorSIMDf(0, 0, 0), matrix4SIMD());
        CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> mouse;
        CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

        core::smart_refctd_ptr<lod_library_t> lodLibrary;
        core::smart_refctd_ptr<culling_system_t> cullingSystem;
        CullPushConstants_t cullPushConstants;
        culling_system_t::Params cullingParams;
        core::smart_refctd_ptr<video::CDrawIndirectAllocator<>> drawIndirectAllocator;

        core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[FRAMES_IN_FLIGHT];
        core::smart_refctd_ptr<video::IGPUCommandBuffer> bakedCommandBuffer;
        core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
        core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
        core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };

        video::CDumbPresentationOracle oracle;
        uint32_t acquiredNextFBO = {};
        int32_t resourceIx = -1;
};

NBL_COMMON_API_MAIN(LoDSystemApp)