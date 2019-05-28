#define _IRR_STATIC_LIB_

/* GCC compile Flags
-flto
-fuse-linker-plugin
-fno-omit-frame-pointer //for debug
-msse3
-mfpmath=sse
-ggdb3 //for debug
*/
/* Linker Flags
-lIrrlicht
-lXrandr
-lGL
-lX11
-lpthread
-ldl

-fuse-ld=gold
-flto
-fuse-linker-plugin
-msse3
*/
#include <irrlicht.h>
#include "../source/Irrlicht/COpenGLExtensionHandler.h"

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

	if (device == 0)
		return 1; // could not create selected driver.


	video::IVideoDriver* driver = device->getVideoDriver();
	scene::ISceneManager* smgr = device->getSceneManager();


	uint64_t lastFPSTime = 0;

	while(device->run())
	if (device->isWindowActive())
	{
		driver->beginScene(true, false, video::SColor(255,255,0,0) ); //this gets 11k FPS


		driver->endScene();

		// display frames per second in window title
		uint64_t time = device->getTimer()->getRealTime();
		if (time-lastFPSTime > 1000)
		{
		    std::wostringstream str(L"Hello World - Irrlicht Engine [");
		    str.seekp(0,std::ios_base::end);
			str << driver->getName() << "] FPS:" << driver->getFPS();

			device->setWindowCaption(str.str());
			lastFPSTime = time;
		}
	}
	/*
    //create a screenshot
	video::IImage* screenshot = driver->createImage(asset::EF_B8G8R8A8_UNORM,params.WindowSize);
    glReadPixels(0,0, params.WindowSize.Width,params.WindowSize.Height, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, screenshot->getData());
	glFinish(); // to make sure the data is done reading
    asset::CImageData* img = new asset::CImageData();
    asset::IAssetWriter::SAssetWriteParams wparams(img);
    device->getAssetManager().writeAsset("screenshot.png", wparams);
    img->drop();
	screenshot->drop();
	*/
	device->drop();

	return 0;
}
