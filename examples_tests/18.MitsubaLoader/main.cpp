#define _IRR_STATIC_LIB_
#include <irrlicht.h>

#include "../../ext/ScreenShot/ScreenShot.h"

#include "../common/QToQuitEventReceiver.h"

#include "../3rdparty/portable-file-dialogs/portable-file-dialogs.h"
#include "../../ext/MitsubaLoader/CMitsubaLoader.h"

using namespace irr;
using namespace core;

class SimpleCallBack : public video::IShaderConstantSetCallBack
{
		int32_t mvpUniformLocation;
		int32_t colorUniformLocation;
		int32_t nastyUniformLocation;
		video::E_SHADER_CONSTANT_TYPE mvpUniformType;
		video::E_SHADER_CONSTANT_TYPE colorUniformType;
		video::E_SHADER_CONSTANT_TYPE nastyUniformType;
	public:
		virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE& materialType, const core::vector<video::SConstantLocationNamePair>& constants)
		{
			for (size_t i=0; i<constants.size(); i++)
			{
				if (constants[i].name == "MVP")
				{
					mvpUniformLocation = constants[i].location;
					mvpUniformType = constants[i].type;
				}
				else if (constants[i].name == "color")
				{
					colorUniformLocation = constants[i].location;
					colorUniformType = constants[i].type;
				}
				else if (constants[i].name == "nasty")
				{
					nastyUniformLocation = constants[i].location;
					nastyUniformType = constants[i].type;
				}
			}
		}

		virtual void OnSetMaterial(video::IMaterialRendererServices* services, const video::SGPUMaterial& material, const video::SGPUMaterial& lastMaterial)
		{
			services->setShaderConstant(&material.AmbientColor, colorUniformLocation, colorUniformType, 1);
			services->setShaderConstant(&material.MaterialTypeParam, nastyUniformLocation, nastyUniformType, 1);
		}

		virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t userData)
		{
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
	params.Vsync = false;
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

		std::string filePath = "../../media/mitsuba/bedroom.zip";
	#define MITSUBA_LOADER_TESTS
	#ifndef MITSUBA_LOADER_TESTS
		pfd::message("Choose file to load", "Choose mitsuba XML file to load or ZIP containing an XML. \nIf you cancel or choosen file fails to load staircase will be loaded.", pfd::choice::ok);
		pfd::open_file file("Choose XML or ZIP file", "../../media/mitsuba", { "XML files (.xml)", "*.xml", "ZIP files (.zip)", "*.zip" });
		if (!file.result().empty())
			filePath = file.result()[0];
	#endif
		if (core::hasFileExtension(io::path(filePath.c_str()), "zip", "ZIP"))
		{
			io::IFileArchive* arch = nullptr;
			device->getFileSystem()->addFileArchive(filePath.c_str(),io::EFAT_ZIP,"",&arch);
			if (!arch)
				device->getFileSystem()->addFileArchive("../../media/mitsuba/staircase2.zip", io::EFAT_ZIP, "", &arch);
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

		//! read cache results -- speeds up mesh generation
		{
			io::IReadFile* cacheFile = device->getFileSystem()->createAndOpenFile("./normalCache101010.sse");
			if (cacheFile)
			{
				asset::normalCacheFor2_10_10_10Quant.resize(cacheFile->getSize() / sizeof(asset::QuantizationCacheEntry2_10_10_10));
				cacheFile->read(asset::normalCacheFor2_10_10_10Quant.data(), cacheFile->getSize());
				cacheFile->drop();

				//make sure its still ok
				std::sort(asset::normalCacheFor2_10_10_10Quant.begin(), asset::normalCacheFor2_10_10_10Quant.end());
			}
		}
		//! load the mitsuba scene
		meshes = am->getAsset(filePath, {});
		//! cache results -- speeds up mesh generation on second run
		{
			io::IWriteFile* cacheFile = device->getFileSystem()->createAndWriteFile("./normalCache101010.sse");
			cacheFile->write(asset::normalCacheFor2_10_10_10Quant.data(), asset::normalCacheFor2_10_10_10Quant.size() * sizeof(asset::QuantizationCacheEntry2_10_10_10));
			cacheFile->drop();
		}

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
		params.WindowSize.Height = film.height;
	}
	params.DriverType = video::EDT_OPENGL;
	IrrlichtDevice* device = createDeviceEx(params);

	if (device == 0)
		return 1; // could not create selected driver.


	scene::ISceneManager* smgr = device->getSceneManager();
	QToQuitEventReceiver receiver;
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

		camera = smgr->addCameraSceneNodeFPS(nullptr,100.f,0.01f);
		// need to extract individual components
		{
			auto relativeTransform = sensor.transform.matrix.extractSub3x4();
			auto pos = relativeTransform.getTranslation();
			camera->setPosition(pos.getAsVector3df());

			core::vectorSIMDf up;
			auto target = pos;
			for (auto i=0; i<3; i++)
			{
				up[i] = relativeTransform.rows[i].y;
				target[i] += relativeTransform.rows[i].z;
			}
			camera->setTarget(target.getAsVector3df());
			camera->setUpVector(up.getAsVector3df());
		}

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
		float realFoVDegrees;
		auto width = viewport.getWidth();
		auto height = viewport.getHeight();
		float aspectRatio = float(width) / float(height);
		auto convertFromXFoV = [=](float fov) -> float
		{
			float aspectX = tan(core::radians(fov)*0.5f);
			return core::degrees(atan(aspectX/aspectRatio)*2.f);
		};
		switch (persp->fovAxis)
		{
			case ext::MitsubaLoader::CElementSensor::PerspectivePinhole::FOVAxis::X:
				realFoVDegrees = convertFromXFoV(persp->fov);
				break;
			case ext::MitsubaLoader::CElementSensor::PerspectivePinhole::FOVAxis::Y:
				realFoVDegrees = persp->fov;
				break;
			case ext::MitsubaLoader::CElementSensor::PerspectivePinhole::FOVAxis::DIAGONAL:
				{
					float aspectDiag = tan(core::radians(persp->fov)*0.5f);
					float aspectY = aspectDiag/core::sqrt(1.f+aspectRatio*aspectRatio);
					realFoVDegrees = core::degrees(atan(aspectY)*2.f);
				}
				break;
			case ext::MitsubaLoader::CElementSensor::PerspectivePinhole::FOVAxis::SMALLER:
				if (width < height)
					realFoVDegrees = convertFromXFoV(persp->fov);
				else
					realFoVDegrees = persp->fov;
				break;
			case ext::MitsubaLoader::CElementSensor::PerspectivePinhole::FOVAxis::LARGER:
				if (width < height)
					realFoVDegrees = persp->fov;
				else
					realFoVDegrees = convertFromXFoV(persp->fov);
				break;
			default:
				realFoVDegrees = NAN;
				assert(false);
				break;
		}
		auto projMat = core::matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(realFoVDegrees),aspectRatio,persp->nearClip,persp->farClip);
		camera->setProjectionMatrix(projMat);
	}
	else
		camera = smgr->addCameraSceneNodeFPS(0, 100.0f, 0.01f);
	camera->setLeftHanded(false);
	smgr->setActiveCamera(camera);
	device->getCursorControl()->setVisible(false);

	uint64_t lastFPSTime = 0;
	float lastFastestMeshFrameNr = -1.f;

	while (device->run() && receiver.keepOpen())
	{
		driver->beginScene(true, true, video::SColor(255, 0, 0, 255));
		driver->setViewPort(viewport);

		//! This animates (moves) the camera and sets the transforms
		//! Also draws the meshbuffer
		smgr->drawAll();

		driver->endScene();

		// display frames per second in window title
		uint64_t time = device->getTimer()->getRealTime();
		if (time - lastFPSTime > 1000)
		{
			std::wostringstream str;
			str << L"Mitsuba Loader Demo - Irrlicht Engine [" << driver->getName() << "] FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(str.str());
			lastFPSTime = time;
		}
	}

	// create a screenshot
	{
		core::rect<uint32_t> sourceRect(0, 0, params.WindowSize.Width, params.WindowSize.Height);
		ext::ScreenShot::dirtyCPUStallingScreenshot(device, "screenshot.png", sourceRect, asset::EF_R8G8B8_SRGB);
	}

	device->drop();
	return 0;
}
