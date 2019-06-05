#define _IRR_STATIC_LIB_

#include <irrlicht.h>
#include "BRDFExplorerApp.h"

// TODO: document example
/**
This example just shows a screen which clears to red,
nothing fancy, just to show that Irrlicht links fine
**/
using namespace irr;

/*
The start of the main function starts like in most other example. We ask the
user for the desired renderer and start it up.
*/
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

    auto* brdfExplorerApp = new BRDFExplorerApp(device);

    uint64_t lastFPSTime = 0;

    while(device->run())
    if (device->isWindowActive())
    {
        driver->beginScene(true, false, video::SColor(255,255,0,0) ); //this gets 11k FPS
        brdfExplorerApp->renderGUI();
        driver->endScene();
    }
    // create a screenshot using example 09's mechanism
    device->drop();

    return 0;
}
