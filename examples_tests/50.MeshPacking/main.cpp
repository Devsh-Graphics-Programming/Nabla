// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <irrlicht.h>

//! I advise to check out this file, its a basic input handler
#include "../common/QToQuitEventReceiver.h"
#include "irr/asset/CCPUMeshPacker.h"

using namespace irr;
using namespace core;
using namespace asset;
using namespace video;

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

void packMeshBuffers(video::IVideoDriver* driver, core::vector<ICPUMeshBuffer*>& meshBuffers, SVertexInputParams& vipOutput, DrawIndexedIndirectInput& output)
{
    MeshPackerBase::PackedMeshBuffer<ICPUBuffer> packedMeshBuffer;
    //core::smart_refctd_ptr<IGPUBuffer> gpuIndirectDrawBuffer;

    MeshPackerBase::AllocationParams allocParams;
    allocParams.indexBuffSupportedCnt = 20000000u;
    allocParams.indexBufferMinAllocSize = 5000u;
    allocParams.vertexBuffSupportedCnt = 20000000u;
    allocParams.vertexBufferMinAllocSize = 5000u;
    allocParams.MDIDataBuffSupportedCnt = 20000u;
    allocParams.MDIDataBuffMinAllocSize = 1u; //so structs are adjacent in memory
    allocParams.perInstanceVertexBuffSupportedCnt = 0u;
    allocParams.perInstanceVertexBufferMinAllocSize = 0u;

    CCPUMeshPacker mp(meshBuffers[0]->getPipeline()->getVertexInputParams(), allocParams, std::numeric_limits<uint16_t>::max() / 3u, std::numeric_limits<uint16_t>::max() / 3u);

    //TODO: test for multiple alloc
    //TODO: test mp.getPackerCreationParamsFromMeshBufferRange()
    MeshPackerBase::ReservedAllocationMeshBuffers ramb = mp.alloc(meshBuffers.begin(), meshBuffers.end());
    assert(ramb.isValid());

    mp.instantiateDataStorage();

    MeshPackerBase::PackedMeshBufferData pmbd =  mp.commit(meshBuffers.begin(), meshBuffers.end(), ramb);
    assert(pmbd.isValid());

    MeshPackerBase::PackedMeshBuffer pmb = mp.getPackedMeshBuffer();
    assert(pmb.isValid());

    auto& cpuVtxBuff = pmb.vertexBufferBindings[0].buffer;
    auto gpuVtxBuff = driver->createFilledDeviceLocalGPUBufferOnDedMem(cpuVtxBuff->getSize(), cpuVtxBuff->getPointer());

    for (uint32_t i = 0u; i < video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT; i++)
        output.vtxBindings[i] = { pmb.vertexBufferBindings[i].offset, gpuVtxBuff };
    output.indexBuff = driver->createFilledDeviceLocalGPUBufferOnDedMem(pmb.indexBuffer.buffer->getSize(), pmb.indexBuffer.buffer->getPointer());
    output.indirectDrawBuff = driver->createFilledDeviceLocalGPUBufferOnDedMem(pmb.MDIDataBuffer->getSize(), pmb.MDIDataBuffer->getPointer());
    output.maxCount = pmbd.mdiParameterCount;

    vipOutput = pmb.vertexInputParams;
}

void setPipeline(IVideoDriver* driver, IAssetManager* am, SVertexInputParams& vtxInputParams,
    core::smart_refctd_ptr<IGPURenderpassIndependentPipeline>& gpuPipeline)
{
    IAssetLoader::SAssetLoadParams lp;

    auto vertexShaderBundle = am->getAsset("../shader.vert", lp);
    auto fragShaderBundle = am->getAsset("../shader.frag", lp);
    ICPUSpecializedShader* shaders[2] =
    {
        IAsset::castDown<ICPUSpecializedShader>(vertexShaderBundle.getContents().begin()->get()),
        IAsset::castDown<ICPUSpecializedShader>(fragShaderBundle.getContents().begin()->get())
    };

    {
        auto gpuShaders = driver->getGPUObjectsFromAssets(shaders, shaders + 2);
        IGPUSpecializedShader* shaders[2] = { gpuShaders->operator[](0).get(), gpuShaders->operator[](1).get() };

        asset::SPushConstantRange pcRange = { asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD) };
        auto pipelineLayout = driver->createGPUPipelineLayout(&pcRange, &pcRange + 1);

        gpuPipeline = driver->createGPURenderpassIndependentPipeline(
            nullptr, core::smart_refctd_ptr(pipelineLayout),
            shaders, shaders + 2u,
            vtxInputParams,
            asset::SBlendParams(), asset::SPrimitiveAssemblyParams(), SRasterizationParams());
    }
}

int main()
{
    // create device with full flexibility over creation parameters
    // you can add more parameters if desired, check irr::SIrrlichtCreationParameters
    irr::SIrrlichtCreationParameters params;
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
    qnc->loadNormalQuantCacheFromFile<asset::CQuantNormalCache::E_CACHE_TYPE::ECT_2_10_10_10>(fs, "../../tmp/normalCache101010.sse", true);

    // register the zip
    device->getFileSystem()->addFileArchive("../../media/sponza.zip");

    asset::IAssetLoader::SAssetLoadParams lp;
    auto meshes_bundle = am->getAsset("sponza.obj", lp);
    assert(!meshes_bundle.isEmpty());
    auto mesh = meshes_bundle.getContents().begin()[0];
    auto mesh_raw = static_cast<asset::ICPUMesh*>(mesh.get());

    //saving cache to file
    qnc->saveCacheToFile(asset::CQuantNormalCache::E_CACHE_TYPE::ECT_2_10_10_10, fs, "../../tmp/normalCache101010.sse");

    core::vector<ICPUMeshBuffer*> meshBuffers;
    for (uint32_t i = 0u; i < mesh_raw->getMeshBufferCount(); i++)
        meshBuffers.push_back(mesh_raw->getMeshBuffer(i));

    DrawIndexedIndirectInput mdiCallParams;
    SVertexInputParams vtxInputParams;
    packMeshBuffers(driver, meshBuffers, vtxInputParams, mdiCallParams);

    core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> gpuPipeline;
    setPipeline(driver, am, vtxInputParams, gpuPipeline);

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

        driver->pushConstants(gpuPipeline->getLayout(), asset::ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD), camera->getConcatenatedMatrix().pointer());
        driver->drawIndexedIndirect(mdiCallParams.vtxBindings, mdiCallParams.mode, mdiCallParams.indexType, mdiCallParams.indexBuff.get(), mdiCallParams.indirectDrawBuff.get(), mdiCallParams.offset, mdiCallParams.maxCount, mdiCallParams.stride);

        driver->endScene();
    }
}