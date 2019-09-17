#define _IRR_STATIC_LIB_
#include <irrlicht.h>

#include "../3rdparty/portable-file-dialogs/portable-file-dialogs.h"
#include "../../ext/MitsubaLoader/CMitsubaLoader.h"

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

class SimpleCallBack : public video::IShaderConstantSetCallBack
{
	int32_t mvpUniformLocation;
	int32_t cameraDirUniformLocation;
	int32_t texUniformLocation[4];
	video::E_SHADER_CONSTANT_TYPE mvpUniformType;
	video::E_SHADER_CONSTANT_TYPE cameraDirUniformType;
	video::E_SHADER_CONSTANT_TYPE texUniformType[4];
public:
	SimpleCallBack() : cameraDirUniformLocation(-1), cameraDirUniformType(video::ESCT_FLOAT_VEC3) {}

	virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE& materialType, const core::vector<video::SConstantLocationNamePair>& constants)
	{
		for (size_t i = 0; i < constants.size(); i++)
		{
			if (constants[i].name == "MVP")
			{
				mvpUniformLocation = constants[i].location;
				mvpUniformType = constants[i].type;
			}
			else if (constants[i].name == "cameraPos")
			{
				cameraDirUniformLocation = constants[i].location;
				cameraDirUniformType = constants[i].type;
			}
		}
	}

	virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t userData)
	{
		core::vectorSIMDf modelSpaceCamPos;
		modelSpaceCamPos.set(services->getVideoDriver()->getTransform(video::E4X3TS_WORLD_VIEW_INVERSE).getTranslation());
		services->setShaderConstant(&modelSpaceCamPos, cameraDirUniformLocation, cameraDirUniformType, 1);
		services->setShaderConstant(services->getVideoDriver()->getTransform(video::EPTS_PROJ_VIEW_WORLD).pointer(), mvpUniformLocation, mvpUniformType, 1);
	}

	virtual void OnUnsetMaterial() {}
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
	
	video::IVideoDriver* driver = device->getVideoDriver();

	scene::ISceneManager* smgr = device->getSceneManager();
	driver->setTextureCreationFlag(video::ETCF_ALWAYS_32_BIT, true);
	scene::ICameraSceneNode* camera =
		smgr->addCameraSceneNodeFPS(0, 100.0f, 0.01f);
	camera->setPosition(core::vector3df(0, 0, 3));
	camera->setTarget(core::vector3df(0, 0, 0));
	camera->setNearValue(0.01f);
	camera->setFarValue(10000.0f);
	smgr->setActiveCamera(camera);
	device->getCursorControl()->setVisible(false);
	MyEventReceiver receiver;
	device->setEventReceiver(&receiver);

	SimpleCallBack* cb = new SimpleCallBack();
	video::E_MATERIAL_TYPE newMaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../mesh.vert",
		"", "", "", //! No Geometry or Tessellation Shaders
		"../mesh.frag",
		3, video::EMT_SOLID, //! 3 vertices per primitive (this is tessellation shader relevant only
		cb, //! Our Shader Callback
		0); //! No custom user data
	cb->drop();

	if (device == 0)
		return 1; // could not create selected driver.

	io::IFileSystem* fs = device->getFileSystem();
	asset::IAssetManager* am = device->getAssetManager();

	am.addAssetLoader(core::make_smart_refctd_ptr<irr::ext::MitsubaLoader::CMitsubaLoader>(device));

#define MITSUBA_LOADER_TESTS

#ifdef MITSUBA_LOADER_TESTS
	std::string filePath = "../../media/mitsuba/staircase2/scene.xml";
#else
	pfd::message("Choose file to load", "Choose mitsuba XML file to load or ZIP containing an XML. \nIf you cancel or choosen file fails to load bathroom will be loaded.", pfd::choice::ok);
	pfd::open_file file("Choose XML file", "", { "XML files (.xml)", "*.xml" });
	std::string filePath = file.result().empty() ? "C:\\IrrlichtBAW\\/IrrlichtBAW\\examples_tests\\media\\mitsuba\\bathroom\\sce===
		netest.xml" : file.result()[0];
#endif
	asset::SAssetBundle meshes = am->getAsset(filePath, {});

	for (int i = 0; i < meshes.getSize(); i++)
	{
		asset::ICPUMesh* cpumesh = static_cast<asset::ICPUMesh*>((meshes.getContents().first + i)->get());
		video::IGPUMesh* gpumesh = driver->getGPUObjectsFromAssets(&cpumesh, (&cpumesh) + 1)[0];
		smgr->addMeshSceneNode(gpumesh)->setMaterialType(newMaterialType);
		gpumesh->drop();
	}

	while (!quit && device->run())
	{
		driver->beginScene(true, true, video::SColor(255, 0, 0, 255));

		//! This animates (moves) the camera and sets the transforms
		//! Also draws the meshbuffer
		smgr->drawAll();

		driver->endScene();
	}

	device->drop();
	return 0;
}
