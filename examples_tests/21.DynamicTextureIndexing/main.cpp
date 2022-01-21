// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

//! I advise to check out this file, its a basic input handler
#include "../common/Camera.hpp"
#include "../common/CommonAPI.h"

#include "nbl/asset/utils/CCPUMeshPackerV1.h"

using namespace nbl;
using namespace core;
using namespace asset;
using namespace video;

class DynamicTextureIndexingApp : public ApplicationBase
{
    constexpr static uint32_t WIN_W = 1280;
    constexpr static uint32_t WIN_H = 720;
    constexpr static uint32_t SC_IMG_COUNT = 3u;
    constexpr static uint32_t FRAMES_IN_FLIGHT = 5u;
    constexpr static uint64_t MAX_TIMEOUT = 99999999999999ull;
    constexpr static size_t NBL_FRAMES_TO_AVERAGE = 100ull;

    static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);


public:
    struct CustomIndirectCommand : DrawElementsIndirectCommand_t
    {
        uint32_t diffuseTexBinding = std::numeric_limits<uint32_t>::max();
        uint32_t bumpTexBinding = std::numeric_limits<uint32_t>::max();
    };
    using MeshPacker = asset::CCPUMeshPackerV1<CustomIndirectCommand>;

    nbl::core::smart_refctd_ptr<nbl::ui::IWindowManager> windowManager;
    nbl::core::smart_refctd_ptr<nbl::ui::IWindow> window;
    nbl::core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCb;
    nbl::core::smart_refctd_ptr<nbl::video::IAPIConnection> gl;
    nbl::core::smart_refctd_ptr<nbl::video::ISurface> surface;
    nbl::core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
    nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
    nbl::video::IPhysicalDevice* gpuPhysicalDevice;
    std::array<nbl::video::IGPUQueue*, CommonAPI::InitOutput<SC_IMG_COUNT>::EQT_COUNT> queues = { nullptr, nullptr, nullptr, nullptr };
    nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
    nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
    std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, SC_IMG_COUNT> fbo;
    nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool> commandPool;
    nbl::core::smart_refctd_ptr<nbl::system::ISystem> system;
    nbl::core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
    nbl::video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
    nbl::core::smart_refctd_ptr<nbl::system::ILogger> logger;
    nbl::core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;
    
    nbl::core::smart_refctd_ptr<video::IGPUFence> gpuTransferFence;
    nbl::core::smart_refctd_ptr<video::IGPUFence> gpuComputeFence;
    nbl::video::IGPUObjectFromAssetConverter cpu2gpu;
    
    asset::ICPUMesh* meshRaw = nullptr;
    const asset::COBJMetadata* metaOBJ = nullptr;
    
    core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
    core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
    core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
    core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[FRAMES_IN_FLIGHT];

    using GpuDescriptoSetPair = std::pair<core::smart_refctd_ptr<IGPUDescriptorSet>, core::smart_refctd_ptr<IGPUDescriptorSet>>;
    core::vector<GpuDescriptoSetPair> desc;

    CommonAPI::InputSystem::ChannelReader<IMouseEventChannel> mouse;
    CommonAPI::InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;
    Camera camera = Camera(vectorSIMDf(0, 0, 0), vectorSIMDf(0, 0, 0), matrix4SIMD());
    
    core::smart_refctd_ptr<video::IDescriptorPool> descriptorPool;

    core::smart_refctd_ptr<IGPUPipelineLayout> gpuPipelineLayout;
    core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> gpuPipeline;
    core::smart_refctd_ptr<video::IGPUGraphicsPipeline> gpuGraphicsPipeline;
    core::smart_refctd_ptr<IGPUDescriptorSetLayout> ds0layout;
    core::smart_refctd_ptr<IGPUDescriptorSetLayout> ds1layout;

    struct DrawIndexedIndirectInput
    {
        asset::SBufferBinding<video::IGPUBuffer> vtxBindings[video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
        const video::IGPUBuffer* vtxBindingsBuffers[video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
        size_t vtxBindingsOffsets[video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
        asset::E_PRIMITIVE_TOPOLOGY mode = asset::EPT_TRIANGLE_LIST;
        asset::E_INDEX_TYPE indexType = asset::EIT_16BIT;
        core::smart_refctd_ptr<video::IGPUBuffer> indexBuff = nullptr;
        core::smart_refctd_ptr<video::IGPUBuffer> indirectDrawBuff = nullptr;
        size_t offset = 0u;
        size_t maxCount = 0u;
        size_t stride = 0u;
        core::smart_refctd_ptr<video::IGPUBuffer> countBuffer = nullptr;
        size_t countOffset = 0u;
    };

    core::vector<MeshPacker::PackerDataStore> packedMeshBuffer;
    core::vector<DrawIndexedIndirectInput> mdiCallParams;
    core::vector<core::smart_refctd_ptr<IGPUBuffer>> gpuIndirectDrawBuffer;

    uint32_t ds1UboBinding = 0;
    int resourceIx;
    uint32_t acquiredNextFBO = {};
    std::chrono::system_clock::time_point lastTime;
    bool frameDataFilled = false;
    size_t frame_count = 0ull;
    double time_sum = 0;
    double dtList[NBL_FRAMES_TO_AVERAGE] = {};
    
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

    APP_CONSTRUCTOR(DynamicTextureIndexingApp)

    void onAppInitialized_impl() override
    {
        CommonAPI::InitOutput<SC_IMG_COUNT> initOutput;
        initOutput.window = core::smart_refctd_ptr(window);
        CommonAPI::Init<WIN_W, WIN_H, SC_IMG_COUNT>(initOutput, video::EAT_OPENGL, "DynamicTextureIndexingApp", nbl::asset::EF_D32_SFLOAT);
        window = std::move(initOutput.window);
        windowCb = std::move(initOutput.windowCb);
        gl = std::move(initOutput.apiConnection);
        surface = std::move(initOutput.surface);
        utilities = std::move(initOutput.utilities);
        logicalDevice = std::move(initOutput.logicalDevice);
        gpuPhysicalDevice = initOutput.physicalDevice;
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

        descriptorPool = createDescriptorPool(1u);

        gpuTransferFence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));
        gpuComputeFence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

        nbl::video::IGPUObjectFromAssetConverter cpu2gpu;
        {
            cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].fence = &gpuTransferFence;
            cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].fence = &gpuComputeFence;
        }

        {
            auto* quantNormalCache = assetManager->getMeshManipulator()->getQuantNormalCache();
            quantNormalCache->loadCacheFromFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), "../../tmp/normalCache101010.sse");

            system::path archPath = sharedInputCWD / "sponza.zip";
            auto arch = system->openFileArchive(archPath);
            // test no alias loading (TODO: fix loading from absolute paths)
            system->mount(std::move(arch));
            asset::IAssetLoader::SAssetLoadParams loadParams;
            loadParams.workingDirectory = sharedInputCWD;
            loadParams.logger = logger.get();
            auto meshes_bundle = assetManager->getAsset((sharedInputCWD / "sponza.zip/sponza.obj").string(), loadParams);
            assert(!meshes_bundle.getContents().empty());

            metaOBJ = meshes_bundle.getMetadata()->selfCast<const asset::COBJMetadata>();

            auto cpuMesh = meshes_bundle.getContents().begin()[0];
            meshRaw = static_cast<asset::ICPUMesh*>(cpuMesh.get());

            quantNormalCache->saveCacheToFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), "../../tmp/normalCache101010.sse");
        }

        const auto meshbuffers = meshRaw->getMeshBuffers();
        core::vector<ICPUMeshBuffer*> meshBuffers(meshbuffers.begin(), meshbuffers.end());

        //divide mesh buffers into sets, where sum of textures used by meshes in one set is <= 16
        struct MBRangeTexturesPair
        {
            core::SRange<void, core::vector<ICPUMeshBuffer*>::iterator> mbRanges;
            core::unordered_map<ICPUImageView*, uint32_t> textures;
        };

        core::vector<MBRangeTexturesPair> mbRangesTex;
        {
            auto rangeBegin = meshBuffers.begin();
            core::unordered_map<ICPUImageView*, uint32_t> textures;
            textures.reserve(16u);
            uint32_t texBinding = 0u;

            for (auto it = meshBuffers.begin(); it < meshBuffers.end(); it++)
            {
                auto ds3 = (*it)->getAttachedDescriptorSet();
                ICPUImageView* tex = dynamic_cast<ICPUImageView*>(ds3->getDescriptors(0u).begin()->desc.get());
                ICPUImageView* texBump = dynamic_cast<ICPUImageView*>(ds3->getDescriptors(6u).begin()->desc.get());

                auto a = textures.insert(std::make_pair(tex, texBinding));
                if (a.second == true)
                    texBinding++;

                auto b = textures.insert(std::make_pair(texBump, texBinding));
                if (b.second == true)
                    texBinding++;

                if (texBinding >= 15u)
                {
                    mbRangesTex.push_back({ {rangeBegin, it + 1}, textures });
                    rangeBegin = it + 1;
                    texBinding = 0;
                    textures.clear();
                }

                if (it + 1 == meshBuffers.end())
                {
                    mbRangesTex.push_back({ {rangeBegin, meshBuffers.end()}, textures });
                    break;
                }
            }
        }

        packedMeshBuffer = core::vector<MeshPacker::PackerDataStore>(mbRangesTex.size());
        mdiCallParams = core::vector<DrawIndexedIndirectInput>(mbRangesTex.size());
        gpuIndirectDrawBuffer = core::vector<core::smart_refctd_ptr<IGPUBuffer>>(mbRangesTex.size());
        for (size_t i = 0u; i < mbRangesTex.size(); i++)
        {
            asset::SVertexInputParams& inputParams = meshRaw->getMeshBuffers().begin()[0]->getPipeline()->getVertexInputParams();
            MeshPacker::AllocationParams allocationParams;

            auto& mbRange = mbRangesTex[i].mbRanges;
            const ptrdiff_t meshBuffersInRangeCnt = std::distance(mbRange.begin(), mbRange.end());

            allocationParams.indexBuffSupportedCnt = 20000000u;
            allocationParams.indexBufferMinAllocCnt = 5000u;
            allocationParams.vertexBuffSupportedByteSize = 2147483648u;
            allocationParams.vertexBufferMinAllocByteSize = 500u;
            allocationParams.MDIDataBuffSupportedCnt = 20000u;
            allocationParams.vertexBufferMinAllocByteSize = 1u;
            allocationParams.perInstanceVertexBuffSupportedByteSize = 0u;
            allocationParams.perInstanceVertexBufferMinAllocByteSize = 0u;

            //pack mesh buffers
            MeshPacker packer(inputParams, allocationParams, std::numeric_limits<uint16_t>::max() / 3u, std::numeric_limits<uint16_t>::max() / 3u);

            MeshPacker::ReservedAllocationMeshBuffers resData;
            MeshPacker::PackedMeshBufferData pmbData;

            resData = packer.alloc(mbRange.begin(), mbRange.end());
            if (!resData.isValid())
            {
                logger->log("Allocation failed.", system::ILogger::ELL_ERROR);
                assert(false);
            }

            packer.instantiateDataStorage();

            pmbData = packer.commit(mbRange.begin(), mbRange.end(), resData, nullptr);
            if (!pmbData.isValid())
            {
                logger->log("Commit failed.", system::ILogger::ELL_ERROR);
                assert(false);
            }
            assert(pmbData.isValid());

            packedMeshBuffer[i] = packer.getPackerDataStore();
            assert(packedMeshBuffer[i].isValid());

            //fill ssbo with correct values
            auto& textures = mbRangesTex[i].textures;
            CustomIndirectCommand* ssboBuffer = static_cast<CustomIndirectCommand*>(packedMeshBuffer[i].MDIDataBuffer->getPointer());
            const size_t mdiBuffSz = packedMeshBuffer[i].MDIDataBuffer->getSize();

            uint32_t mbIdx = 0u;
            for (auto it = mbRange.begin(); it != mbRange.end(); it++)
            {
                auto ds3 = (*it)->getAttachedDescriptorSet();
                ICPUImageView* tex = dynamic_cast<ICPUImageView*>(ds3->getDescriptors(0u).begin()->desc.get());
                ICPUImageView* texBump = dynamic_cast<ICPUImageView*>(ds3->getDescriptors(6u).begin()->desc.get());

                auto texBindDiffuseIt = textures.find(tex);
                auto texBindBumpIt = textures.find(texBump);
                assert(texBindDiffuseIt != textures.end());
                assert(texBindBumpIt != textures.end());

                const uint32_t texBindDiffuse = (*texBindDiffuseIt).second;
                const uint32_t texBindBump = (*texBindBumpIt).second;

                auto mdiCmdForMb = ssboBuffer + mbIdx++;
                mdiCmdForMb->diffuseTexBinding = texBindDiffuse;
                mdiCmdForMb->bumpTexBinding = texBindBump;
            }

            assert(pmbData.mdiParameterCount == meshBuffersInRangeCnt);

            //create draw call inputs
            mdiCallParams[i].indexBuff = utilities->createFilledDeviceLocalGPUBufferOnDedMem(queues[CommonAPI::InitOutput<SC_IMG_COUNT>::EQT_TRANSFER_UP], packedMeshBuffer[i].indexBuffer.buffer->getSize(), packedMeshBuffer[i].indexBuffer.buffer->getPointer());

            auto& cpuVtxBuff = packedMeshBuffer[i].vertexBufferBindings[0].buffer;

            gpuIndirectDrawBuffer[i] = utilities->createFilledDeviceLocalGPUBufferOnDedMem(queues[CommonAPI::InitOutput<SC_IMG_COUNT>::EQT_TRANSFER_UP], sizeof(CustomIndirectCommand) * pmbData.mdiParameterCount, packedMeshBuffer[i].MDIDataBuffer->getPointer());
            mdiCallParams[i].indirectDrawBuff = core::smart_refctd_ptr(gpuIndirectDrawBuffer[i]);

            auto gpuVtxBuff = utilities->createFilledDeviceLocalGPUBufferOnDedMem(queues[CommonAPI::InitOutput<SC_IMG_COUNT>::EQT_TRANSFER_UP], cpuVtxBuff->getSize(), cpuVtxBuff->getPointer());

            for (uint32_t j = 0u; j < video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT; j++)
            {
                mdiCallParams[i].vtxBindings[j] = { packedMeshBuffer[i].vertexBufferBindings[j].offset, std::move(gpuVtxBuff) };
                mdiCallParams[i].vtxBindingsBuffers[j] = mdiCallParams[i].vtxBindings[j].buffer.get();
                mdiCallParams[i].vtxBindingsOffsets[j] = mdiCallParams[i].vtxBindings[j].offset;
            }
                

            mdiCallParams[i].stride = sizeof(CustomIndirectCommand);
            mdiCallParams[i].maxCount = pmbData.mdiParameterCount;
        }

        //create pipeline
        IAssetLoader::SAssetLoadParams lp;
        auto vertexShaderBundle = assetManager->getAsset("../mesh.vert", lp);
        auto fragShaderBundle = assetManager->getAsset("../mesh.frag", lp);
        ICPUSpecializedShader* shaders[2] =
        {
            IAsset::castDown<ICPUSpecializedShader>(vertexShaderBundle.getContents().begin()->get()),
            IAsset::castDown<ICPUSpecializedShader>(fragShaderBundle.getContents().begin()->get())
        };

        ISampler::SParams sp;
        sp.TextureWrapU = ISampler::E_TEXTURE_CLAMP::ETC_REPEAT;
        sp.TextureWrapV = ISampler::E_TEXTURE_CLAMP::ETC_REPEAT;
        sp.MinFilter = ISampler::E_TEXTURE_FILTER::ETF_LINEAR;
        sp.MaxFilter = ISampler::E_TEXTURE_FILTER::ETF_LINEAR;
        auto sampler = logicalDevice->createGPUSampler(sp);
        {
            asset::SPushConstantRange range[1] = { asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD) };

            core::smart_refctd_ptr<IGPUSampler> samplerArray[16u] = { nullptr };
            for (uint32_t i = 0u; i < 16u; i++)
                samplerArray[i] = core::smart_refctd_ptr(sampler);

            //set layout
            {
                video::IGPUDescriptorSetLayout::SBinding b[1];
                b[0].binding = 1u;
                b[0].count = 16u;
                b[0].type = EDT_COMBINED_IMAGE_SAMPLER;
                b[0].stageFlags = ISpecializedShader::ESS_FRAGMENT;
                b[0].samplers = samplerArray;

                ds0layout = logicalDevice->createGPUDescriptorSetLayout(b, b + 1);
            }
            {
                video::IGPUDescriptorSetLayout::SBinding b[1];
                b[0].binding = 0u;
                b[0].count = 1u;
                b[0].type = EDT_STORAGE_BUFFER;
                b[0].stageFlags = ISpecializedShader::ESS_FRAGMENT;

                ds1layout = logicalDevice->createGPUDescriptorSetLayout(b, b + 1);
            }

            auto gpuShaders = cpu2gpu.getGPUObjectsFromAssets(shaders, shaders + 2, cpu2gpuParams);
            //cpu2gpuParams.waitForCreationToComplete();
            IGPUSpecializedShader* shaders[2] = { gpuShaders->operator[](0).get(), gpuShaders->operator[](1).get() };

            gpuPipelineLayout = logicalDevice->createGPUPipelineLayout(range, range + 1, core::smart_refctd_ptr(ds0layout), core::smart_refctd_ptr(ds1layout));
            gpuPipeline = logicalDevice->createGPURenderpassIndependentPipeline(nullptr, core::smart_refctd_ptr(gpuPipelineLayout), shaders, shaders + 2u, packedMeshBuffer[0].vertexInputParams, asset::SBlendParams(), asset::SPrimitiveAssemblyParams(), SRasterizationParams());

            nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams;
            graphicsPipelineParams.renderpassIndependent = core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline>(const_cast<video::IGPURenderpassIndependentPipeline*>(gpuPipeline.get()));
            graphicsPipelineParams.renderpass = core::smart_refctd_ptr(renderpass);

            gpuGraphicsPipeline = logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(graphicsPipelineParams));
        }

        desc = core::vector<GpuDescriptoSetPair>(mbRangesTex.size());
        for (uint32_t i = 0u; i < mbRangesTex.size(); i++)
        {
            auto& texDesc = desc[i].first;
            auto& ssboDesc = desc[i].second;

            auto& range = mbRangesTex[i].mbRanges;
            auto& textures = mbRangesTex[i].textures;
            const uint32_t texCnt = textures.size();

            texDesc = logicalDevice->createGPUDescriptorSet(descriptorPool.get(), core::smart_refctd_ptr(ds0layout));
            ssboDesc = logicalDevice->createGPUDescriptorSet(descriptorPool.get(),  core::smart_refctd_ptr(ds1layout));
            video::IGPUDescriptorSet::SDescriptorInfo info[16u];
            video::IGPUDescriptorSet::SWriteDescriptorSet w[16u];

            for (auto& texBind : textures)
            {
                auto texture = texBind.first;
                const uint32_t bind = texBind.second;


                auto gpuTexture = cpu2gpu.getGPUObjectsFromAssets(&texture, &texture + 1, cpu2gpuParams)->front();
                cpu2gpuParams.waitForCreationToComplete();

                info[bind].image.imageLayout = asset::EIL_UNDEFINED;
                info[bind].image.sampler = core::smart_refctd_ptr(sampler);
                info[bind].desc = core::smart_refctd_ptr(gpuTexture);

                w[bind].binding = 1u;
                w[bind].arrayElement = bind;
                w[bind].count = 1u;
                w[bind].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
                w[bind].dstSet = texDesc.get();
                w[bind].info = &info[bind];
            }
            logicalDevice->updateDescriptorSets(texCnt, w, 0u, nullptr);

            {
                video::IGPUDescriptorSet::SDescriptorInfo _info;
                _info.buffer.offset = 0u;
                _info.buffer.size = gpuIndirectDrawBuffer[i]->getSize();
                _info.desc = core::smart_refctd_ptr(gpuIndirectDrawBuffer[i]);

                video::IGPUDescriptorSet::SWriteDescriptorSet _w;
                _w.binding = 0u;
                _w.arrayElement = 0u;
                _w.count = 1u;
                _w.descriptorType = EDT_STORAGE_BUFFER;
                _w.dstSet = ssboDesc.get();
                _w.info = &_info;

                logicalDevice->updateDescriptorSets(1u, &_w, 0u, nullptr);
            }
        }

        core::vectorSIMDf cameraPosition(-4, 0, 0);
        matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60.0f), float(WIN_W) / WIN_H, 0.1, 1000);
        camera = Camera(cameraPosition, core::vectorSIMDf(0, 0, 0), projectionMatrix, 10.f, 1.f);

        uint64_t lastFPSTime = 0;

        for (size_t i = 0ull; i < NBL_FRAMES_TO_AVERAGE; ++i)
            dtList[i] = 0.0;

        logicalDevice->createCommandBuffers(commandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, commandBuffers);

        for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
        {
            imageAcquire[i] = logicalDevice->createSemaphore();
            renderFinished[i] = logicalDevice->createSemaphore();
        }

        constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;
        uint32_t acquiredNextFBO = {};
        resourceIx = -1;
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

        core::matrix3x4SIMD modelMatrix;
        modelMatrix.setTranslation(nbl::core::vectorSIMDf(0, 0, 0, 0));

        core::matrix4SIMD mvp = core::concatenateBFollowedByA(viewProjectionMatrix, modelMatrix);

        commandBuffer->bindGraphicsPipeline(gpuGraphicsPipeline.get());

        for (uint32_t i = 0u; i < mdiCallParams.size(); i++)
        {
            commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuPipeline->getLayout(), 0u, 2u, &desc[i].first.get(), nullptr);
            commandBuffer->bindIndexBuffer(mdiCallParams[i].indexBuff.get(), 0ull, mdiCallParams[i].indexType);
            commandBuffer->bindVertexBuffers(0u, 1u, mdiCallParams[i].vtxBindingsBuffers, &mdiCallParams[i].vtxBindingsOffsets[0]);
            commandBuffer->bindVertexBuffers(2u, 1u, mdiCallParams[i].vtxBindingsBuffers, &mdiCallParams[i].vtxBindingsOffsets[2]);
            commandBuffer->bindVertexBuffers(3u, 1u, mdiCallParams[i].vtxBindingsBuffers, &mdiCallParams[i].vtxBindingsOffsets[3]);
            commandBuffer->pushConstants(gpuPipeline->getLayout(), video::IGPUSpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD), camera.getConcatenatedMatrix().pointer());

            commandBuffer->drawIndexedIndirect(mdiCallParams[i].indirectDrawBuff.get(), mdiCallParams[i].offset, mdiCallParams[i].maxCount, mdiCallParams[i].stride);
        }

        /*core::vector<uint8_t> uboData(gpuubo->getSize());
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
        }*/

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
            queues[CommonAPI::InitOutput<1>::EQT_GRAPHICS], renderFinished[resourceIx].get(), acquiredNextFBO);
    }

    bool keepRunning() override
    {
        return windowCb->isWindowOpen();
    }
};

NBL_COMMON_API_MAIN(DynamicTextureIndexingApp, DynamicTextureIndexingApp::Nabla)

/*
driver->bindGraphicsPipeline(gpuPipeline.get());

driver->beginScene(true, true, video::SColor(255, 0, 0, 255));

//! This animates (moves) the camera and sets the transforms
camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
camera->render();

driver->pushConstants(gpuPipelineLayout.get(), asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD), camera->getConcatenatedMatrix().pointer());

for (uint32_t i = 0u; i < mdiCallParams.size(); i++)
{
    driver->bindDescriptorSets(video::EPBP_GRAPHICS, gpuPipeline->getLayout(), 0u, 2u, &desc[i].first.get(), nullptr);
    driver->drawIndexedIndirect(mdiCallParams[i].vtxBindings, mdiCallParams[i].mode, mdiCallParams[i].indexType, mdiCallParams[i].indexBuff.get(), mdiCallParams[i].indirectDrawBuff.get(), mdiCallParams[i].offset, mdiCallParams[i].maxCount, mdiCallParams[i].stride);
}

driver->endScene();
*/