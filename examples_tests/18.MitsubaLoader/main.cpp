#define _IRR_STATIC_LIB_
#include <irrlicht.h>

#include "../3rdparty/portable-file-dialogs/portable-file-dialogs.h"
#include "..\..\ext\MitsubaLoader\CMitsubaLoader.h"

using namespace irr;
using namespace core;

bool quit = false;
class MyEventReceiver : public IEventReceiver
{
public:

	MyEventReceiver()
	{
	}

	bool OnEvent(const SEvent& event)
	{
		if (event.EventType == irr::EET_KEY_INPUT_EVENT && !event.KeyInput.PressedDown)
		{
			switch (event.KeyInput.Key)
			{
			case irr::KEY_ESCAPE:
			case irr::KEY_KEY_Q:
				quit = true;
				return true;
			default:
				break;
			}
		}

		return false;
	}

private:
};

int main()
{
	srand(time(0));
	// create device with full flexibility over creation parameters
	// you can add more parameters if desired, check irr::SIrrlichtCreationParameters
	irr::SIrrlichtCreationParameters params;
	params.Bits = 24; //may have to set to 32bit for some platforms
	params.ZBufferBits = 24; //we'd like 32bit here
	params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
	params.WindowSize = dimension2d<uint32_t>(800, 600);
	params.Fullscreen = false;
	params.Vsync = true; //! If supported by target platform
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon
	IrrlichtDevice* device = createDeviceEx(params);
	
	if (device == 0)
		return 1; // could not create selected driver.

	video::IVideoDriver* driver = device->getVideoDriver();
	io::IFileSystem* fs = device->getFileSystem();
	asset::IAssetManager& am = device->getAssetManager();

	//temporary solution ofc
	ext::MitsubaLoader::CMitsubaLoader* loader = new ext::MitsubaLoader::CMitsubaLoader(device);

#define MITSUBA_LOADER_TESTS

#ifdef MITSUBA_LOADER_TESTS
	std::string filePath = "C:\\IrrlichtBAW\\IrrlichtBAW\\ext\\MitsubaLoader\\testScene.xml";
#else
	pfd::message("Choose file to load", "Choose mitsuba XML file to load. \nIf you cancel or choosen file fails to load bathroom will be loaded.", pfd::choice::ok);
	pfd::open_file file("Choose XML file", "", { "XML files (.xml)", "*.xml" });
	std::string filePath = file.result().empty() ? "C:\\IrrlichtBAW\\/IrrlichtBAW\\examples_tests\\media\\mitsuba\\bathroom\\scene.xml" : file.result()[0];
#endif

	io::IReadFile* xmlFile = fs->createAndOpenFile(filePath.c_str());
	loader->loadAsset(xmlFile, {});


	loader->drop();
	device->drop();
	return 0;
}
