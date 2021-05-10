// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

//! I advise to check out this file, its a basic input handler
#include "../common/QToQuitEventReceiver.h"

using namespace nbl;
using namespace core;
using namespace asset;
using namespace video;

int main()
{
    // create device with full flexibility over creation parameters
    // you can add more parameters if desired, check nbl::SIrrlichtCreationParameters
    nbl::SIrrlichtCreationParameters params;
    params.Bits = 24; //may have to set to 32bit for some platforms
    params.ZBufferBits = 24; //we'd like 32bit here
    params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
    params.WindowSize = dimension2d<uint32_t>(1280, 720);
    params.Fullscreen = false;
    params.Vsync = true; //! If supported by target platform
    params.Doublebuffer = true;
    params.Stencilbuffer = false; //! This will not even be a choice soon
    auto device = createDeviceEx(params);

    if (!device)
        return 1; // could not create selected driver.

    //! disable mouse cursor, since camera will force it to the middle
    //! and we don't want a jittery cursor in the middle distracting us
    device->getCursorControl()->setVisible(false);

    //! Since our cursor will be enslaved, there will be no way to close the window
    //! So we listen for the "Q" key being pressed and exit the application
    QToQuitEventReceiver receiver;
    device->setEventReceiver(&receiver);


    auto* driver = device->getVideoDriver();
    auto* smgr = device->getSceneManager();
    auto* am = device->getAssetManager();
    auto* fs = am->getFileSystem();

    //
    auto* qnc = am->getMeshManipulator()->getQuantNormalCache();
    //loading cache from file
    qnc->loadCacheFromFile<asset::EF_A2B10G10R10_SNORM_PACK32>(fs, "../../tmp/normalCache101010.sse");

    // register the zip
    device->getFileSystem()->addFileArchive("../../media/sponza.zip");

    asset::IAssetLoader::SAssetLoadParams lp;
    auto meshes_bundle = am->getAsset("sponza.obj", lp);
    assert(!meshes_bundle.getContents().empty());
    auto mesh = meshes_bundle.getContents().begin()[0];
    auto mesh_raw = static_cast<asset::ICPUMesh*>(mesh.get());

    //saving cache to file
    qnc->saveCacheToFile<asset::EF_A2B10G10R10_SNORM_PACK32>(fs, "../../tmp/normalCache101010.sse");

    const auto meta = meshes_bundle.getMetadata()->selfCast<const COBJMetadata>();
    
    const auto meshbuffers = mesh_raw->getMeshBuffers();
    core::vector<ICPUMeshBuffer*> meshBuffers(meshbuffers.begin(),meshbuffers.end());

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
    
    struct CustomIndirectCommand : DrawElementsIndirectCommand_t
    {
        uint32_t diffuseTexBinding = std::numeric_limits<uint32_t>::max();
        uint32_t bumpTexBinding = std::numeric_limits<uint32_t>::max();
    };

    struct DrawIndexedIndirectInput
    {
        asset::SBufferBinding<video::IGPUBuffer> vtxBindings[video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
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

    core::vector<asset::MeshPackerBase::PackedMeshBuffer<asset::ICPUBuffer>> packedMeshBuffer(mbRangesTex.size());
    core::vector<DrawIndexedIndirectInput> mdiCallParams(mbRangesTex.size());
    core::vector<core::smart_refctd_ptr<IGPUBuffer>> gpuIndirectDrawBuffer(mbRangesTex.size());
    for(size_t i = 0u; i < mbRangesTex.size(); i++)
    {
        asset::SVertexInputParams& inputParams = mesh_raw->getMeshBuffers().begin()[0]->getPipeline()->getVertexInputParams();
        asset::MeshPackerBase::AllocationParams allocationParams;

        auto& mbRange = mbRangesTex[i].mbRanges;
        const ptrdiff_t meshBuffersInRangeCnt = std::distance(mbRange.begin(), mbRange.end());

        allocationParams.indexBuffSupportedCnt = 20000000u;
        allocationParams.indexBufferMinAllocSize = 5000u;
        allocationParams.vertexBuffSupportedCnt = 20000000u;
        allocationParams.vertexBufferMinAllocSize = 5000u;
        allocationParams.MDIDataBuffSupportedCnt = 20000u;
        allocationParams.MDIDataBuffMinAllocSize = 1u; //so structs are adjacent in memory
        allocationParams.perInstanceVertexBuffSupportedCnt = 0u;
        allocationParams.perInstanceVertexBufferMinAllocSize = 0u;

            //pack mesh buffers
        asset::CCPUMeshPacker<CustomIndirectCommand> packer(inputParams, allocationParams, std::numeric_limits<uint16_t>::max() / 3u, std::numeric_limits<uint16_t>::max() / 3u);
        
        MeshPackerBase::ReservedAllocationMeshBuffers resData;
        asset::MeshPackerBase::PackedMeshBufferData pmbData;
        
        resData = packer.alloc(mbRange.begin(), mbRange.end());
        assert(resData.isValid());

        packer.instantiateDataStorage();

        pmbData = packer.commit(mbRange.begin(), mbRange.end(), resData);
        assert(pmbData.isValid());

        packedMeshBuffer[i] = packer.getPackedMeshBuffer();
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
        mdiCallParams[i].indexBuff = driver->createFilledDeviceLocalGPUBufferOnDedMem(packedMeshBuffer[i].indexBuffer.buffer->getSize(), packedMeshBuffer[i].indexBuffer.buffer->getPointer());

        auto& cpuVtxBuff = packedMeshBuffer[i].vertexBufferBindings[0].buffer;

        gpuIndirectDrawBuffer[i] = driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(CustomIndirectCommand) * pmbData.mdiParameterCount, packedMeshBuffer[i].MDIDataBuffer->getPointer());
        mdiCallParams[i].indirectDrawBuff = core::smart_refctd_ptr(gpuIndirectDrawBuffer[i]);

        auto gpuVtxBuff = driver->createFilledDeviceLocalGPUBufferOnDedMem(cpuVtxBuff->getSize(), cpuVtxBuff->getPointer());

        for (uint32_t j = 0u; j < video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT; j++)
            mdiCallParams[i].vtxBindings[j] = { packedMeshBuffer[i].vertexBufferBindings[j].offset, gpuVtxBuff };

        mdiCallParams[i].stride = sizeof(CustomIndirectCommand);
        mdiCallParams[i].maxCount = pmbData.mdiParameterCount;
    }

        //create pipeline
    auto vertexShaderBundle = am->getAsset("../mesh.vert", lp);
    auto fragShaderBundle = am->getAsset("../mesh.frag", lp);
    ICPUSpecializedShader* shaders[2] =
    {
        IAsset::castDown<ICPUSpecializedShader>(vertexShaderBundle.getContents().begin()->get()),
        IAsset::castDown<ICPUSpecializedShader>(fragShaderBundle.getContents().begin()->get())
    };

    core::smart_refctd_ptr<IGPUPipelineLayout> gpuPipelineLayout;
    core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> gpuPipeline;
    core::smart_refctd_ptr<IGPUDescriptorSetLayout> ds0layout;
    core::smart_refctd_ptr<IGPUDescriptorSetLayout> ds1layout;

    ISampler::SParams sp;
    sp.TextureWrapU = ISampler::E_TEXTURE_CLAMP::ETC_REPEAT;
    sp.TextureWrapV = ISampler::E_TEXTURE_CLAMP::ETC_REPEAT;
    sp.MinFilter = ISampler::E_TEXTURE_FILTER::ETF_LINEAR;
    sp.MaxFilter = ISampler::E_TEXTURE_FILTER::ETF_LINEAR;
    auto sampler = driver->createGPUSampler(sp);
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

            ds0layout = driver->createGPUDescriptorSetLayout(b, b + 1);
        }
        {
            video::IGPUDescriptorSetLayout::SBinding b[1];
            b[0].binding = 0u;
            b[0].count = 1u;
            b[0].type = EDT_STORAGE_BUFFER;
            b[0].stageFlags = ISpecializedShader::ESS_FRAGMENT;

            ds1layout = driver->createGPUDescriptorSetLayout(b, b + 1);
        }

        auto gpuShaders = driver->getGPUObjectsFromAssets(shaders, shaders + 2);
        IGPUSpecializedShader* shaders[2] = { gpuShaders->operator[](0).get(), gpuShaders->operator[](1).get() };

        gpuPipelineLayout = driver->createGPUPipelineLayout(range, range + 1, core::smart_refctd_ptr(ds0layout), core::smart_refctd_ptr(ds1layout));
        gpuPipeline = driver->createGPURenderpassIndependentPipeline(nullptr, core::smart_refctd_ptr(gpuPipelineLayout), shaders, shaders + 2u, packedMeshBuffer[0].vertexInputParams, asset::SBlendParams(), asset::SPrimitiveAssemblyParams(), SRasterizationParams());
    }
    using GpuDescriptoSetPair = std::pair<core::smart_refctd_ptr<IGPUDescriptorSet>, core::smart_refctd_ptr<IGPUDescriptorSet>>;
    core::vector<GpuDescriptoSetPair> desc(mbRangesTex.size());
    for(uint32_t i = 0u; i < mbRangesTex.size(); i++)
    {
        auto& texDesc = desc[i].first;
        auto& ssboDesc = desc[i].second;

        auto& range = mbRangesTex[i].mbRanges;
        auto& textures = mbRangesTex[i].textures;
        const uint32_t texCnt = textures.size();

        texDesc = driver->createGPUDescriptorSet(core::smart_refctd_ptr(ds0layout));
        ssboDesc = driver->createGPUDescriptorSet(core::smart_refctd_ptr(ds1layout));
        video::IGPUDescriptorSet::SDescriptorInfo info[16u];
        video::IGPUDescriptorSet::SWriteDescriptorSet w[16u];

        for(auto& texBind : textures)
        {
            auto texture = texBind.first;
            const uint32_t bind = texBind.second;
            auto gpuTexture = driver->getGPUObjectsFromAssets(&texture, &texture + 1)->front();

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
        driver->updateDescriptorSets(texCnt, w, 0u, nullptr);

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

            driver->updateDescriptorSets(1u, &_w, 0u, nullptr);
        }
    }

    //! we want to move around the scene and view it from different angles
    scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeFPS(0, 100.0f, 0.5f);

    camera->setPosition(core::vector3df(-4, 0, 0));
    camera->setTarget(core::vector3df(0, 0, 0));
    camera->setNearValue(1.f);
    camera->setFarValue(5000.0f);

    smgr->setActiveCamera(camera);

    uint64_t lastFPSTime = 0;
    while (device->run() && receiver.keepOpen())
    {
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
    }

    return 0;
}