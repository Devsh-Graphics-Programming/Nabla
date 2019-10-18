#define _IRR_STATIC_LIB_
#include <irrlicht.h>

#include "../ext/ScreenShot/ScreenShot.h"
#include "../common/QToQuitEventReceiver.h"

using namespace irr;
using namespace core;


class SimpleCallBack : public video::IShaderConstantSetCallBack
{
	int32_t mvpUniformLocation;
	video::E_SHADER_CONSTANT_TYPE mvpUniformType;
public:
	SimpleCallBack() {}

	virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE& materialType, const core::vector<video::SConstantLocationNamePair>& constants)
	{
		for (size_t i = 0; i < constants.size(); i++)
		{
			if (constants[i].name == "MVP")
			{
				mvpUniformLocation = constants[i].location;
				mvpUniformType = constants[i].type;
			}
		}
	}

	virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t userData)
	{
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
	params.WindowSize = dimension2d<uint32_t>(1280, 720);
	params.Fullscreen = false;
	params.Vsync = false;
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon
	IrrlichtDevice* device = createDeviceEx(params);

	if (device == 0)
		return 1; // could not create selected driver.


	device->getCursorControl()->setVisible(false);

	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);


	video::IVideoDriver* driver = device->getVideoDriver();

	SimpleCallBack* cb = new SimpleCallBack();
	video::E_MATERIAL_TYPE newMaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../mesh.vert",
		"", "", "", //! No Geometry or Tessellation Shaders
		"../mesh.frag",
		3, video::EMT_SOLID, //! 3 vertices per primitive (this is tessellation shader relevant only
		cb, //! Our Shader Callback
		0); //! No custom user data
	cb->drop();



	scene::ISceneManager* smgr = device->getSceneManager();
	driver->setTextureCreationFlag(video::ETCF_ALWAYS_32_BIT, true);
	scene::ICameraSceneNode* camera =
		smgr->addCameraSceneNodeFPS(0, 100.0f, 1.f);
	camera->setPosition(core::vector3df(-4, 0, 0));
	camera->setTarget(core::vector3df(0, 0, 0));
	camera->setNearValue(1.f);
	camera->setFarValue(10000.0f);
	smgr->setActiveCamera(camera);

	io::IFileSystem* fs = device->getFileSystem();
	auto am = device->getAssetManager();

	//! Material setting lambda
	auto setMaterialTypeOnAllMeshBuffers = [](auto* node, auto type)
	{
		auto* mesh = node->getMesh();
		for (auto i = 0u; i < mesh->getMeshBufferCount(); i++)
			mesh->getMeshBuffer(i)->getMaterial().MaterialType = type;
	};

	//! Load big-ass sponza model
	// really want to get it working with a "../../media/sponza.zip?sponza.obj" path handling
	fs->addFileArchive("../../media/sponza.zip", false, false);
	asset::IAssetLoader::SAssetLoadParams lparams;
	auto cpumesh = core::smart_refctd_ptr_static_cast<asset::ICPUMesh>(*am->getAsset("sponza.obj", lparams).getContents().first);
	if (cpumesh)
		setMaterialTypeOnAllMeshBuffers(smgr->addMeshSceneNode(std::move(driver->getGPUObjectsFromAssets(&cpumesh.get(), (&cpumesh.get()) + 1)->operator[](0))),newMaterialType);


	uint64_t lastFPSTime = 0;

	while (device->run() && receiver.keepOpen())
		//if (device->isWindowActive())
	{
		driver->beginScene(true, true, video::SColor(255, 0, 0, 255));

		//! This animates (moves) the camera and sets the transforms
		//! Also draws the meshbuffer
		smgr->drawAll();

		driver->endScene();

		// display frames per second in window title
		uint64_t time = device->getTimer()->getRealTime();
		if (time - lastFPSTime > 1000)
		{
			std::wostringstream sstr;
			sstr << L"Builtin Nodes Demo - Irrlicht Engine FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(sstr.str().c_str());
			lastFPSTime = time;
		}
	}


	//create a screenshot
	{
		core::rect<uint32_t> sourceRect(0, 0, params.WindowSize.Width, params.WindowSize.Height);
		ext::ScreenShot::dirtyCPUStallingScreenshot(device, "screenshot.png", sourceRect, asset::EF_R8G8B8_SRGB);
	}


	device->drop();

	return 0;
}
