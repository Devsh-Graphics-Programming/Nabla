#define _IRR_STATIC_LIB_

#include <irrlicht.h>
#include "BRDFExplorerApp.h"
#include "CBRDFBuiltinIncludeLoader.h"

using namespace irr;

int main()
{
    // create device with full flexibility over creation parameters
    // you can add more parameters if desired, check irr::SIrrlichtCreationParameters
    irr::SIrrlichtCreationParameters params;
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

    scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeMaya(nullptr, -750.f, 200.f, 200.f, -1, 10.f);
    camera->setPosition(core::vector3df(-4.f, 0.f, 0.f));
    camera->setTarget(core::vector3df(0.f, 0.f, 0.f));
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
        driver->beginScene(true, true, video::SColor(255,255,0,0) );

        // needed for camera to move
        smgr->drawAll();

        brdfExplorerApp->renderMesh();
        brdfExplorerApp->renderGUI();
        driver->endScene();
    }
    // create a screenshot using example 09's mechanism
    device->drop();

    return 0;
}
