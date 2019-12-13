#define _IRR_STATIC_LIB_
#include <irrlicht.h>

#include "../common/QToQuitEventReceiver.h"

// TODO: remove dependency
#include "../src/irr/asset/CBAWMeshWriter.h"

using namespace irr;
using namespace core;


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
        for (size_t i=0; i<constants.size(); i++)
        {
            if (constants[i].name=="MVP")
            {
                mvpUniformLocation = constants[i].location;
                mvpUniformType = constants[i].type;
            }
            else if (constants[i].name=="cameraPos")
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
        services->setShaderConstant(&modelSpaceCamPos,cameraDirUniformLocation,cameraDirUniformType,1);
        services->setShaderConstant(services->getVideoDriver()->getTransform(video::EPTS_PROJ_VIEW_WORLD).pointer(),mvpUniformLocation,mvpUniformType,1);
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
	params.Vsync = true; //! If supported by target platform
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
                                                        "","","", //! No Geometry or Tessellation Shaders
                                                        "../mesh.frag",
                                                        3,video::EMT_SOLID, //! 3 vertices per primitive (this is tessellation shader relevant only
                                                        cb, //! Our Shader Callback
                                                        0); //! No custom user data
    cb->drop();



	scene::ISceneManager* smgr = device->getSceneManager();
	driver->setTextureCreationFlag(video::ETCF_ALWAYS_32_BIT, true);
	scene::ICameraSceneNode* camera =
		smgr->addCameraSceneNodeFPS(0,100.0f,0.01f);
	camera->setPosition(core::vector3df(-4,0,0));
	camera->setTarget(core::vector3df(0,0,0));
	camera->setNearValue(0.01f);
	camera->setFarValue(100.0f);
    smgr->setActiveCamera(camera);

	io::IFileSystem* fs = device->getFileSystem();

	// from Criss:
	// here i'm testing baw mesh writer and loader
	// (import from .stl/.obj, then export to .baw, then import from .baw :D)
	// Seems to work for those two simple meshes, but need more testing!

    auto am = device->getAssetManager();

	//! Test Loading of Obj
    asset::IAssetLoader::SAssetLoadParams lparams;
    auto cpumesh = core::smart_refctd_ptr_static_cast<asset::ICPUMesh>(*am->getAsset("../../media/extrusionLogo_TEST_fixed.stl", lparams).getContents().first);
	// export mesh
    asset::CBAWMeshWriter::WriteProperties bawprops;
    asset::IAssetWriter::SAssetWriteParams wparams(cpumesh.get(), asset::EWF_COMPRESSED, 0.f, 0, nullptr, &bawprops);
	am->writeAsset("extrusionLogo_TEST_fixed.baw", wparams);
	// end export

	// import .baw mesh (test)
    cpumesh = core::smart_refctd_ptr_static_cast<asset::ICPUMesh>(*am->getAsset("extrusionLogo_TEST_fixed.baw", lparams).getContents().first);
	// end import

	//!
	auto setMaterialTypeOnAllMaterials = [](auto* node, auto newMaterialType)
	{
		auto* mesh = node->getMesh();
		for (auto i = 0u; i < mesh->getMeshBufferCount(); i++)
		{
			auto& material = mesh->getMeshBuffer(i)->getMaterial();
			material.MaterialType = newMaterialType;
		}
	};

    if (cpumesh)
		setMaterialTypeOnAllMaterials(smgr->addMeshSceneNode(std::move(driver->getGPUObjectsFromAssets(&cpumesh.get(), (&cpumesh.get())+1)->operator[](0))),newMaterialType);

    cpumesh = core::smart_refctd_ptr_static_cast<asset::ICPUMesh>(*am->getAsset("../../media/cow.obj", lparams).getContents().first);
	// export mesh
    wparams.rootAsset = cpumesh.get();
	am->writeAsset("cow.baw", wparams);
	// end export

	// import .baw mesh (test)
	cpumesh = core::smart_refctd_ptr_static_cast<asset::ICPUMesh>(*am->getAsset("cow.baw", lparams).getContents().first);
	// end import

    if (cpumesh)
		setMaterialTypeOnAllMaterials(smgr->addMeshSceneNode(std::move(driver->getGPUObjectsFromAssets(&cpumesh.get(), (&cpumesh.get())+1)->operator[](0)),0,-1,core::vector3df(3.f,1.f,0.f)),newMaterialType);


	uint64_t lastFPSTime = 0;

	while(device->run() && receiver.keepOpen() )
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
			std::wostringstream sstr;
			sstr << L"Builtin Nodes Demo - Irrlicht Engine FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(sstr.str().c_str());
			lastFPSTime = time;
		}
	}

    
	device->drop();

	return 0;
}
