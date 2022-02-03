// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <irrlicht.h>

//! I advise to check out this file, its a basic input handler
#include "../common/QToQuitEventReceiver.h"
#include "irr/ext/FullScreenTriangle/FullScreenTriangle.h"

//#include "irr/ext/ScreenShot/ScreenShot.h"

#include "irr/ext/DebugDraw/CDraw3DLine.h"
#include "irr/asset/IMeshManipulator.h"

using namespace irr;
using namespace core;
using namespace asset;
using namespace video;

void setPipeline(IVideoDriver* driver, IAssetManager* am, ICPUMeshBuffer* mb, 
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
            mb->getPipeline()->getVertexInputParams(),
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

    //load mesh
    asset::IAssetLoader::SAssetLoadParams lp;
    auto meshes_bundle = am->getAsset("../../media/cow.obj", lp);
    assert(!meshes_bundle.isEmpty());
    auto mesh = meshes_bundle.getContents().begin()[0];
    auto mesh_raw = dynamic_cast<asset::ICPUMesh*>(mesh.get());
    auto cpuMB = mesh_raw->getMeshBuffer(0u);

    {
        uint32_t posAttrIdx = cpuMB->getPositionAttributeIx();
        uint8_t* posPtr = cpuMB->getAttribPointer(posAttrIdx);
        const uint32_t stride = cpuMB->getPipeline()->getVertexInputParams().bindings[0].stride;
        core::matrix3x4SIMD rs;
        rs.setRotation(core::quaternion(0.0f, -3.1415f / 4.0f, -3.1415f / 4.0f));
        core::matrix3x4SIMD s;
        //s.setScale(core::vectorSIMDf(10.0f, 1.0f, 1.0f));
        rs = core::matrix3x4SIMD::concatenateBFollowedByA(rs, s);

        for (size_t i = 0ull; i < cpuMB->calcVertexCount(); i++)
        {
            core::vectorSIMDf pos = cpuMB->getPosition(i);
            rs.pseudoMulWith4x1(pos);
            memcpy(posPtr, pos.pointer, sizeof(float) * 3);
            posPtr += stride;
        }
    }

    asset::IMeshManipulator::OBB obb = asset::IMeshManipulator::calcOBB_DiTO26(cpuMB);

    core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> pipeline;
    setPipeline(driver, am, cpuMB, pipeline);

    core::smart_refctd_ptr<ext::DebugDraw::CDraw3DLine> lineRenderer = ext::DebugDraw::CDraw3DLine::create(driver);
    core::vector<std::pair<ext::DebugDraw::S3DLineVertex, ext::DebugDraw::S3DLineVertex>> linesData;
    lineRenderer->enqueueBox(linesData, core::aabbox3df(), 0, 1, 0, 1, obb.asMat3x4);

    scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeFPS(0, 100.0f, 0.5f);

    camera->setPosition(core::vector3df(-4, 0, 0));
    camera->setTarget(core::vector3df(0, 0, 0));
    camera->setNearValue(1.f);
    camera->setFarValue(5000.0f);

    smgr->setActiveCamera(camera);
    
    auto gpuMB = driver->getGPUObjectsFromAssets(&cpuMB, &cpuMB + 1)->front();

    uint64_t lastFPSTime = 0;
    while (device->run() && receiver.keepOpen())
    {
        driver->beginScene(true, true, video::SColor(255, 0, 0, 0));

        //! This animates (moves) the camera and sets the transforms
        camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count() / 100);
        camera->render();

        driver->bindGraphicsPipeline(pipeline.get());
        driver->pushConstants(pipeline->getLayout(), ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD), camera->getConcatenatedMatrix().pointer());
        driver->drawMeshBuffer(gpuMB.get());

        lineRenderer->draw(camera->getConcatenatedMatrix(), linesData);

        driver->endScene();
    }

    return 0;
}