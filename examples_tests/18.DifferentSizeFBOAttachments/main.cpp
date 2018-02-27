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
#define _IRR_STATIC_LIB_
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

	uint32_t dim[3] = {1024,1024,1};
	video::ITexture* colorAttach = driver->addTexture(video::ITexture::ETT_2D,dim,1,"no_name_color",video::ECF_A16B16G16R16F);
	uint32_t dimHalf[3] = {512,512,1};
	video::ITexture* stencilAttach = driver->addTexture(video::ITexture::ETT_2D,dimHalf,1,"no_name_stencil",video::ECF_STENCIL8);
	//! can even try with a renderbuffer
	//video::IRenderBuffer* stencilAttach = driver->addRenderBuffer(*reinterpret_cast<core::dimension2du*>(dimHalf),video::ECF_STENCIL8);

	video::IFrameBuffer* fbo = driver->addFrameBuffer();
	fbo->attach(video::EFAP_COLOR_ATTACHMENT0,colorAttach);
	fbo->attach(video::EFAP_STENCIL_ATTACHMENT,stencilAttach);

	uint64_t lastFPSTime = 0;

	while(device->run())
	if (device->isWindowActive())
	{
		driver->beginScene(true, false, video::SColor(255,255,0,0) ); //this gets 11k FPS

		bool success = driver->setRenderTarget(fbo);

		//! You can enable this branch for a visual test if you see white then we have success
        if (false && success)
        {
            float color[4] = {1.f,1.f,1.f,1.f};
            driver->clearColorBuffer(video::EFAP_COLOR_ATTACHMENT0,color);
            driver->blitRenderTargets(fbo,0,false,false);
        }

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

	device->drop();

	return 0;
}
