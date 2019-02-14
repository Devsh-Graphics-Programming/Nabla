#define _IRR_STATIC_LIB_
#include <irrlicht.h>

#include "../../ext/DebugDraw/CDraw3DLine.h"
#include "../source/Irrlicht/COpenGLExtensionHandler.h"

using namespace irr;
using namespace core;

int main()
{
    irr::SIrrlichtCreationParameters params;
    params.Bits = 24;
    params.ZBufferBits = 24;
    params.DriverType = video::EDT_OPENGL;
    params.WindowSize = core::dimension2d<uint32_t>(1280, 720);
    params.Fullscreen = false;
    params.Vsync = true;
    params.Doublebuffer = true;
    params.Stencilbuffer = false;
    params.AuxGLContexts = 16;
    IrrlichtDevice* device = createDeviceEx(params);

    if (device == 0)
        return 1;

    video::IVideoDriver* driver = device->getVideoDriver();
    scene::ISceneManager* smgr = device->getSceneManager();
    auto draw3DLine = ext::DebugDraw::CDraw3DLine::create(driver);

    auto camera = smgr->addCameraSceneNodeFPS(0,100.0f,0.001f);

    camera->setPosition(core::vector3df(0,0,-10));
    camera->setTarget(core::vector3df(0,0,0));
    camera->setNearValue(0.01f);
    camera->setFarValue(100.0f);

    smgr->setActiveCamera(camera);

    uint64_t lastFPSTime = 0;

    core::vector<std::pair<ext::DebugDraw::S3DLineVertex, ext::DebugDraw::S3DLineVertex>> lines;

    for (int i = 0; i < 100; ++i)
    {
        lines.push_back({
        {
            { 0.f, 0.f, 0.f },     // start origin
            { 1.f, 0.f, 0.f, 1.f } // start color
        }, {
            { i % 2 ? float(i) : float(-i), 50.f, 10.f}, // end origin
            { 1.f, 0.f, 0.f, 1.f }         // end color
        }
        });
    }

    while(device->run())
    if (device->isWindowActive())
    {
        driver->beginScene(true, true, video::SColor(255,255,255,255));

        smgr->drawAll();

        draw3DLine->draw(
            0.f, 0.f, 0.f,   // start
            0.f, 100.f, 0.f, // end
            1.f, 0, 0, 1.f   // color
        );

        draw3DLine->draw(lines); // multiple lines

        driver->endScene();

        // display frames per second in window title
        uint64_t time = device->getTimer()->getRealTime();
        if (time-lastFPSTime > 1000)
        {
            std::wostringstream str(L"Draw3DLine Ext - Irrlicht Engine [");
            str.seekp(0,std::ios_base::end);
            str << driver->getName() << "] FPS:" << driver->getFPS();

            device->setWindowCaption(str.str());
            lastFPSTime = time;
        }
    }

    draw3DLine->drop();

    //create a screenshot
    video::IImage* screenshot = driver->createImage(asset::EF_B8G8R8A8_UNORM,params.WindowSize);
    glReadPixels(0,0, params.WindowSize.Width,params.WindowSize.Height, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, screenshot->getData());
    {
        // images are horizontally flipped, so we have to fix that here.
        uint8_t* pixels = (uint8_t*)screenshot->getData();

        const int32_t pitch=screenshot->getPitch();
        uint8_t* p2 = pixels + (params.WindowSize.Height - 1) * pitch;
        uint8_t* tmpBuffer = new uint8_t[pitch];
        for (uint32_t i=0; i < params.WindowSize.Height; i += 2)
        {
            memcpy(tmpBuffer, pixels, pitch);
            memcpy(pixels, p2, pitch);
            memcpy(p2, tmpBuffer, pitch);
            pixels += pitch;
            p2 -= pitch;
        }
        delete [] tmpBuffer;
    }
    asset::CImageData* img = new asset::CImageData(screenshot);
    asset::IAssetWriter::SAssetWriteParams wparams(img);
    device->getAssetManager().writeAsset("screenshot.png", wparams);
    img->drop();
    screenshot->drop();

    device->drop();

    return 0;
}
