
#define _IRR_STATIC_LIB_
#include <irrlicht.h>

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
	IrrlichtDevice* device = createDeviceEx(params);

	if (device == 0)
		return 1; // could not create selected driver.


	video::IVideoDriver* driver = device->getVideoDriver();

	uint32_t dim[3] = {params.WindowSize.Width,params.WindowSize.Height,1};
	video::ITexture* colorAttach = driver->addTexture(video::ITexture::ETT_2D,dim,1,"no_name_color",irr::video::ECF_R8G8B8A8);
	video::ITexture* depthAttach = driver->addTexture(video::ITexture::ETT_2D,dim,1,"no_name_depth",irr::video::ECF_DEPTH32F);

	video::IFrameBuffer* fbo = driver->addFrameBuffer();
	fbo->attach(video::EFAP_COLOR_ATTACHMENT0,colorAttach);
	fbo->attach(video::EFAP_DEPTH_ATTACHMENT,depthAttach);
    driver->setRenderTarget(fbo);


	uint64_t lastFPSTime = 0;

	while(device->run())
	{
	    // this should clear backbuffer to red
		driver->beginScene(true, false, video::SColor(255,255,0,0) );

		//these should clear bound FBO
		driver->clearZBuffer();
        float color[4] = {1.f,1.f,1.f,1.f};
        driver->clearColorBuffer(video::EFAP_COLOR_ATTACHMENT0,color);

        for (auto i=0; i<700; i++)
        {
            //green
            {
                float color[4] = {0.f,1.f,0.f,0.f};
                driver->clearScreen(irr::video::ESB_BACK_LEFT,color);
            }
            //blue
            {
                float color[4] = {0.f,0.f,1.f,1.f};
                driver->clearScreen(irr::video::ESB_BACK_LEFT,color);
            }
            //black
            {
                float color[4] = {0.f,0.f,0.f,1.f};
                driver->clearScreen(irr::video::ESB_BACK_LEFT,color);
            }
        }

        driver->blitRenderTargets(fbo,0,false,false);

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
