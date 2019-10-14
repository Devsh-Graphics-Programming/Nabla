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
	// create device with full flexibility over creation parameters
	// you can add more parameters if desired, check irr::SIrrlichtCreationParameters
	irr::SIrrlichtCreationParameters params;
	params.Bits = 24; //may have to set to 32bit for some platforms
	params.ZBufferBits = 24; //we'd like 32bit here
	params.DriverType = video::EDT_NULL;
	params.Fullscreen = false;
	params.Vsync = true; //! If supported by target platform
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon

	//
	asset::SAssetBundle meshes;
	core::smart_refctd_ptr<ext::MitsubaLoader::CGlobalMitsubaMetadata> globalMeta;
	{
		IrrlichtDevice* device = createDeviceEx(params);


		io::IFileSystem* fs = device->getFileSystem();
		asset::IAssetManager* am = device->getAssetManager();

		am->addAssetLoader(core::make_smart_refctd_ptr<irr::ext::MitsubaLoader::CMitsubaLoader>(am));

		std::string filePath = "../../media/mitsuba/staircase2.zip";
	#define MITSUBA_LOADER_TESTS
	#ifndef MITSUBA_LOADER_TESTS
		pfd::message("Choose file to load", "Choose mitsuba XML file to load or ZIP containing an XML. \nIf you cancel or choosen file fails to load bathroom will be loaded.", pfd::choice::ok);
		pfd::open_file file("Choose XML or ZIP file", "../../media/mitsuba", { "XML files (.xml)", "*.xml", "ZIP files (.zip)", "*.zip" });
		if (!file.result().empty())
			filePath = file.result()[0];
	#endif
		if (core::hasFileExtension(io::path(filePath.c_str()), "zip", "ZIP"))
		{
			io::IFileArchive* arch = nullptr;
			device->getFileSystem()->addFileArchive(filePath.c_str(),false,false,io::EFAT_ZIP,"",&arch);
			if (!arch)
				device->getFileSystem()->addFileArchive("../../media/mitsuba/staircase2.zip", false, false, io::EFAT_ZIP, "", &arch);
			if (!arch)
				return 2;

			auto flist = arch->getFileList();
			if (!flist)
				return 3;
			auto files = flist->getFiles();

			for (auto it=files.begin(); it!=files.end(); )
			{
				if (core::hasFileExtension(it->FullName, "xml", "XML"))
					it++;
				else
					it = files.erase(it);
			}
			if (files.size() == 0u)
				return 4;

			std::cout << "Choose File (0-" << files.size() - 1ull << "):" << std::endl;
			for (auto i = 0u; i < files.size(); i++)
				std::cout << i << ": " << files[i].FullName.c_str() << std::endl;
			uint32_t chosen = 0;
	#ifndef MITSUBA_LOADER_TESTS
			std::cin >> chosen;
	#endif
			if (chosen >= files.size())
				chosen = 0u;

			filePath = files[chosen].FullName.c_str();
		}

		meshes = am->getAsset(filePath, {});

		device->drop();


		auto firstmesh = *meshes.getContents().first;
		if (!firstmesh)
			return 2;

		auto meta = firstmesh->getMetadata();
		if (!meta)
			return 3;
		assert(core::strcmpi(meta->getLoaderName(),ext::MitsubaLoader::IMitsubaMetadata::LoaderName) == 0);
		globalMeta = static_cast<ext::MitsubaLoader::IMeshMetadata*>(meta)->globalMetadata;
	}


	// recreate wth resolution
	params.WindowSize = dimension2d<uint32_t>(1280, 720);
	// set resolution
	if (globalMeta->sensors.size())
	{
		const auto& film = globalMeta->sensors.front().film;
		params.WindowSize.Width = film.width;
		params.WindowSize.Width = film.height;
	}
	params.DriverType = video::EDT_OPENGL;
	IrrlichtDevice* device = createDeviceEx(params);

	if (device == 0)
		return 1; // could not create selected driver.


	scene::ISceneManager* smgr = device->getSceneManager();
	MyEventReceiver receiver;
	device->setEventReceiver(&receiver);


	video::IVideoDriver* driver = device->getVideoDriver();
	driver->setTextureCreationFlag(video::ETCF_ALWAYS_32_BIT, true);

	SimpleCallBack* cb = new SimpleCallBack();
	video::E_MATERIAL_TYPE newMaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../mesh.vert",
		"", "", "", //! No Geometry or Tessellation Shaders
		"../mesh.frag",
		3, video::EMT_SOLID, //! 3 vertices per primitive (this is tessellation shader relevant only
		cb, //! Our Shader Callback
		0); //! No custom user data
	cb->drop();

	{
		auto gpumeshes = driver->getGPUObjectsFromAssets<asset::ICPUMesh>(meshes.getContents().first, meshes.getContents().second);
		auto cpuit = meshes.getContents().first;
		for (auto gpuit = gpumeshes->begin(); gpuit!=gpumeshes->end(); gpuit++,cpuit++)
		{
			auto* meta = (*cpuit)->getMetadata();
			assert(meta && core::strcmpi(meta->getLoaderName(),ext::MitsubaLoader::IMitsubaMetadata::LoaderName) == 0);
			const auto* meshmeta = static_cast<const ext::MitsubaLoader::IMeshMetadata*>(meta);

			const auto& gpumesh = *gpuit;
			for (auto i=0u; i<gpumesh->getMeshBufferCount(); i++)
				gpumesh->getMeshBuffer(i)->getMaterial().MaterialType = newMaterialType;


			auto node = smgr->addMeshSceneNode(core::smart_refctd_ptr(gpumesh));
			node->setRelativeTransformationMatrix(meshmeta->getInstances()[0].getAsRetardedIrrlichtMatrix());
		}
	}

	// camera and viewport
	scene::ICameraSceneNode* camera = nullptr;
	core::recti viewport(core::position2di(0,0), core::position2di(params.WindowSize.Width,params.WindowSize.Height));

	auto isOkSensorType = [](const ext::MitsubaLoader::CElementSensor& sensor) -> bool {
		return sensor.type==ext::MitsubaLoader::CElementSensor::Type::PERSPECTIVE || sensor.type==ext::MitsubaLoader::CElementSensor::Type::THINLENS;
	};
	if (globalMeta->sensors.size() && isOkSensorType(globalMeta->sensors.front()))
	{
		const auto& sensor = globalMeta->sensors.front();
		const auto& film = sensor.film;
		viewport = core::recti(core::position2di(film.cropOffsetX,film.cropOffsetY), core::position2di(film.cropWidth,film.cropHeight));

		camera = smgr->addCameraSceneNodeFPS();
		camera->setRelativeTransformationMatrix(sensor.transform.matrix.extractSub3x4().getAsRetardedIrrlichtMatrix());

		const ext::MitsubaLoader::CElementSensor::PerspectivePinhole* persp = nullptr;
		switch (sensor.type)
		{
			case ext::MitsubaLoader::CElementSensor::Type::PERSPECTIVE:
				persp = &sensor.perspective;
				break;
			case ext::MitsubaLoader::CElementSensor::Type::THINLENS:
				persp = &sensor.thinlens;
				break;
			default:
				assert(false);
				break;
		}
		assert(persp->fovAxis == ext::MitsubaLoader::CElementSensor::PerspectivePinhole::FOVAxis::Y);
		camera->setProjectionMatrix(core::matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(persp->fov,float(viewport.getHeight())/float(viewport.getWidth()),persp->nearClip,persp->farClip));
	}
	else
		camera = smgr->addCameraSceneNodeFPS(0, 100.0f, 0.01f);
	smgr->setActiveCamera(camera);

	device->getCursorControl()->setVisible(false);
	while (!quit && device->run())
	{
		driver->beginScene(true, true, video::SColor(255, 0, 0, 255));
		driver->setViewPort(viewport);

		//! This animates (moves) the camera and sets the transforms
		//! Also draws the meshbuffer
		smgr->drawAll();

		driver->endScene();
	}

	device->drop();
	return 0;
}
