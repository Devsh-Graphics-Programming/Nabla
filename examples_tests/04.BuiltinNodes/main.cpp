#define _IRR_STATIC_LIB_
#include <irrlicht.h>
#include "../common/QToQuitEventReceiver.h"
#include "../../ext/ScreenShot/ScreenShot.h"

using namespace irr;
using namespace core;

class SimpleCallBack : public video::IShaderConstantSetCallBack
{
    int32_t mvpUniformLocation;
    video::E_SHADER_CONSTANT_TYPE mvpUniformType;
public:
    SimpleCallBack() : mvpUniformLocation(-1), mvpUniformType(video::ESCT_FLOAT_VEC3) {}

    virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE& materialType, const core::vector<video::SConstantLocationNamePair>& constants)
    {
        //! Normally we'd iterate through the array and check our actual constant names before mapping them to locations but oh well
        mvpUniformLocation = constants[0].location;
        mvpUniformType = constants[0].type;
    }

    virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t userData)
    {
        services->setShaderConstant(services->getVideoDriver()->getTransform(video::EPTS_PROJ_VIEW_WORLD).pointer(),mvpUniformLocation,mvpUniformType,1);
    }

    virtual void OnUnsetMaterial() {}
};


int main()
{
	// create device with full flexibility over creation parameters
	// you can add more parameters if desired, check irr::SIrrlichtCreationParameters
	irr::SIrrlichtCreationParameters params;
	params.Bits = 24; //may have to set to 32bit for some platforms
	params.ZBufferBits = 24; //we'd like 32bit here
	params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
	params.WindowSize = dimension2d<uint32_t>(1280, 720);
	params.Fullscreen = false;
	params.Vsync = true; //! If supported by target platform
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon
	IrrlichtDevice* device = createDeviceEx(params);

	if (device == 0)
		return 1; // could not create selected driver.


	video::IVideoDriver* driver = device->getVideoDriver();
    SimpleCallBack* callBack = new SimpleCallBack();

    //! First need to make a material other than default to be able to draw with custom shader
    video::SGPUMaterial material;
    material.BackfaceCulling = false; //! Triangles will be visible from both sides
    material.MaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../mesh.vert",
                                                        "","","", //! No Geometry or Tessellation Shaders
                                                        "../mesh.frag",
                                                        3,video::EMT_SOLID, //! 3 vertices per primitive (this is tessellation shader relevant only
                                                        callBack, //! No Shader Callback (we dont have any constants/uniforms to pass to the shader)
                                                        0); //! No custom user data
    callBack->drop();


	scene::ISceneManager* smgr = device->getSceneManager();
	driver->setTextureCreationFlag(video::ETCF_ALWAYS_32_BIT, true);
	scene::ICameraSceneNode* camera =
		smgr->addCameraSceneNodeFPS(0,100.0f,0.01f);
	camera->setPosition(core::vector3df(-4,0,0));
	camera->setTarget(core::vector3df(0,0,0));
	camera->setNearValue(0.01f);
	camera->setFarValue(100.0f);
    smgr->setActiveCamera(camera);
	device->getCursorControl()->setVisible(false);
	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);

	//! Test Creation Of Builtin
	scene::IMeshSceneNode* cube = dynamic_cast<scene::IMeshSceneNode*>(smgr->addCubeSceneNode(1.f, 0, -1));
	cube->setRotation(core::vector3df(45, 20, 15));

	scene::IMeshSceneNode* sphere = dynamic_cast<scene::IMeshSceneNode*>(smgr->addSphereSceneNode(2, 128));
	sphere->getMaterial(0).MaterialType = material.MaterialType;
	sphere->setPosition(core::vector3df(4, 0, 0));

	scene::ISceneNode* billboard = smgr->addBillboardSceneNode(0, core::dimension2df(1.f, 1.f), core::vector3df(-4, 0, 0));

	auto assetMgr = device->getAssetManager();
	//! load textures and set them
	{
		asset::IAssetLoader::SAssetLoadParams lparams;
		asset::ICPUTexture* cputextures[]{
			static_cast<asset::ICPUTexture*>(assetMgr->getAsset("../../media/irrlicht2_dn.jpg", lparams).getContents().first->get()),
			static_cast<asset::ICPUTexture*>(assetMgr->getAsset("../../media/skydome.jpg", lparams).getContents().first->get()),
			static_cast<asset::ICPUTexture*>(assetMgr->getAsset("../../media/yellowflowers.dds", lparams).getContents().first->get()) //loads all mipmap levels
		};
		auto gputextures = driver->getGPUObjectsFromAssets(cputextures, cputextures + 3);

		cube->getMaterial(0).setTexture(0, std::move(gputextures->operator[](0u)));
		sphere->getMaterial(0).setTexture(0, std::move(gputextures->operator[](1u)));
		billboard->getMaterial(0).setTexture(0, std::move(gputextures->operator[](2u)));
	}

	uint64_t lastFPSTime = 0;

	while(device->run() && receiver.keepOpen())
	//if (device->isWindowActive())
	{
		driver->beginScene(true, true, video::SColor(255,0,0,255) );

        //! This animates (moves) the camera and sets the transforms
        //! Also draws the meshbuffer
        smgr->drawAll();

		driver->endScene();

		// display frames per second in window title
		uint64_t time = device->getTimer()->getRealTime();
		if (time-lastFPSTime > 1000)
		{
			std::wostringstream str;
			str << L"Builtin Nodes Demo - Irrlicht Engine FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(str.str().c_str());
			lastFPSTime = time;
		}
	}

    //create a screenshot
	{
		core::rect<uint32_t> sourceRect(0, 0, params.WindowSize.Width, params.WindowSize.Height);
		ext::ScreenShot::dirtyCPUStallingScreenshot(device,"screenshot.png",sourceRect,asset::EF_R8G8B8_SRGB);
	}

	device->drop();

	return 0;
}
