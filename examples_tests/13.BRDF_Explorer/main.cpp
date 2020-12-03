// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_

#include <irrlicht.h>
#include "BRDFExplorerApp.h"
#include "CBRDFBuiltinIncludeLoader.h"

using namespace nbl;

int main()
{
    // create device with full flexibility over creation parameters
    // you can add more parameters if desired, check nbl::SIrrlichtCreationParameters
    nbl::SIrrlichtCreationParameters params;
    params.Bits = 24; //may have to set to 32bit for some platforms
    params.ZBufferBits = 24; //we'd like 32bit here
    params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
    params.WindowSize = core::dimension2d<uint32_t>(1280, 720);
    params.Fullscreen = false;
    params.Vsync = true; //! If supported by target platform
    params.Doublebuffer = true;
    params.Stencilbuffer = false; //! This will not even be a choice soon
    params.AuxGLContexts = 16;
    IrrlichtDevice* device = createDeviceEx(params);
    device->setWindowCaption(L"BRDF Explorer");
    device->getCursorControl()->setVisible(false);

    if (device == 0)
        return 1; // could not create selected driver.

    video::IVideoDriver* driver = device->getVideoDriver();
    scene::ISceneManager* smgr = device->getSceneManager();

    scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeModifiedMaya(nullptr, -400.f, 20.f, 200.f, -1, 28.f, 1.f);
    camera->setNearValue(0.01f);
    camera->setFarValue(1000.0f);
    smgr->setActiveCamera(camera);

    {
    auto brdfBuiltinLoader = new CBRDFBuiltinIncludeLoader();
    device->getIncludeHandler()->addBuiltinIncludeLoader(brdfBuiltinLoader);
    brdfBuiltinLoader->drop();
    }

    auto* brdfExplorerApp = new BRDFExplorerApp(device, camera);

    uint64_t lastFPSTime = 0;

    while(device->run())
    if (device->isWindowActive())
    {
        driver->beginScene(true, true, video::SColor(255,0,0,0) );

        // needed for camera to move
        smgr->drawAll();

        brdfExplorerApp->update();
        brdfExplorerApp->renderMesh();
        brdfExplorerApp->renderGUI();
        driver->endScene();
    }
    device->drop();

    return 0;
}
