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
		video::E_MATERIAL_TYPE currentMat;

		int32_t mvpUniformLocation[video::EMT_COUNT+2];
		int32_t nUniformLocation[video::EMT_COUNT+2];
		int32_t colorUniformLocation[video::EMT_COUNT+2];
		int32_t nastyUniformLocation[video::EMT_COUNT+2];
		video::E_SHADER_CONSTANT_TYPE mvpUniformType[video::EMT_COUNT+2];
		video::E_SHADER_CONSTANT_TYPE nUniformType[video::EMT_COUNT+2];
		video::E_SHADER_CONSTANT_TYPE colorUniformType[video::EMT_COUNT+2];
		video::E_SHADER_CONSTANT_TYPE nastyUniformType[video::EMT_COUNT+2];
	public:
		SimpleCallBack() : currentMat(video::EMT_SOLID)
		{
			std::fill(mvpUniformLocation, reinterpret_cast<int32_t*>(mvpUniformType), -1);
		}

		virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE& materialType, const core::vector<video::SConstantLocationNamePair>& constants)
		{
			for (size_t i=0; i<constants.size(); i++)
			{
				if (constants[i].name == "MVP")
				{
					mvpUniformLocation[materialType] = constants[i].location;
					mvpUniformType[materialType] = constants[i].type;
				}
				else if (constants[i].name == "NormalMatrix")
				{
					nUniformLocation[materialType] = constants[i].location;
					nUniformType[materialType] = constants[i].type;
				}
				else if (constants[i].name == "color")
				{
					colorUniformLocation[materialType] = constants[i].location;
					colorUniformType[materialType] = constants[i].type;
				}
				else if (constants[i].name == "nasty")
				{
					nastyUniformLocation[materialType] = constants[i].location;
					nastyUniformType[materialType] = constants[i].type;
				}
			}
		}

		virtual void OnSetMaterial(video::IMaterialRendererServices* services, const video::SGPUMaterial& material, const video::SGPUMaterial& lastMaterial)
		{
			currentMat = material.MaterialType;
			services->setShaderConstant(&material.AmbientColor, colorUniformLocation[currentMat], colorUniformType[currentMat], 1);
			services->setShaderConstant(&material.MaterialTypeParam, nastyUniformLocation[currentMat], nastyUniformType[currentMat], 1);
		}

		virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t userData)
		{
			if (nUniformLocation[currentMat]>=0)
			{
				float tmp[9];
				services->getVideoDriver()->getTransform(video::E4X3TS_VIEW).getSub3x3InverseTransposePacked(tmp);
				services->setShaderConstant(tmp, nUniformLocation[currentMat], nUniformType[currentMat], 1);
			}
			if (mvpUniformLocation[currentMat]>=0)
				services->setShaderConstant(services->getVideoDriver()->getTransform(video::EPTS_PROJ_VIEW_WORLD).pointer(), mvpUniformLocation[currentMat], mvpUniformType[currentMat], 1);
		}

		virtual void OnUnsetMaterial() {}
};

int main()
{
	// create device with full flexibility over creation parameters
	// you can add more parameters if desired, check irr::SIrrlichtCreationParameters
	irr::SIrrlichtCreationParameters params;
	params.Bits = 24; //may have to set to 32bit for some platforms
	params.ZBufferBits = 24;
	params.DriverType = video::EDT_OPENGL;
	params.Fullscreen = false;
	params.Vsync = false;
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon
	params.WindowSize = dimension2d<uint32_t>(1600, 900);
	IrrlichtDevice* device = createDeviceEx(params);
	if (device == 0)
		return 1; // could not create selected driver.

	//
	asset::SAssetBundle meshes;
	core::smart_refctd_ptr<ext::MitsubaLoader::CGlobalMitsubaMetadata> globalMeta;
	{
		io::IFileSystem* fs = device->getFileSystem();
		asset::IAssetManager* am = device->getAssetManager();

		am->addAssetLoader(core::make_smart_refctd_ptr<irr::ext::MitsubaLoader::CSerializedLoader>(am));
		am->addAssetLoader(core::make_smart_refctd_ptr<irr::ext::MitsubaLoader::CMitsubaLoader>(am));

		std::string filePath = "../../media/mitsuba/daily_pt.xml";
	//#define MITSUBA_LOADER_TESTS
	#ifndef MITSUBA_LOADER_TESTS
		pfd::message("Choose file to load", "Choose mitsuba XML file to load or ZIP containing an XML. \nIf you cancel or choosen file fails to load, simple scene will be loaded.", pfd::choice::ok);
		pfd::open_file file("Choose XML or ZIP file", "../../media/mitsuba", { "ZIP files (.zip)", "*.zip", "XML files (.xml)", "*.xml"});
		if (!file.result().empty())
			filePath = file.result()[0];
	#endif
		if (core::hasFileExtension(io::path(filePath.c_str()), "zip", "ZIP"))
		{
			io::IFileArchive* arch = nullptr;
			device->getFileSystem()->addFileArchive(filePath.c_str(),io::EFAT_ZIP,"",&arch);
			if (arch)
			{
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
		}

		//! read cache results -- speeds up mesh generation
		{
			io::IReadFile* cacheFile = device->getFileSystem()->createAndOpenFile("../../tmp/normalCache101010.sse");
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
			io::IWriteFile* cacheFile = device->getFileSystem()->createAndWriteFile("../../tmp/normalCache101010.sse");
			if (cacheFile)
			{
				cacheFile->write(asset::normalCacheFor2_10_10_10Quant.data(), asset::normalCacheFor2_10_10_10Quant.size() * sizeof(asset::QuantizationCacheEntry2_10_10_10));
				cacheFile->drop();
			}
		}
		
		auto contents = meshes.getContents();
		if (contents.first>=contents.second)
			return 2;

		auto firstmesh = *contents.first;
		if (!firstmesh)
			return 3;

		auto meta = firstmesh->getMetadata();
		if (!meta)
			return 4;
		assert(core::strcmpi(meta->getLoaderName(),ext::MitsubaLoader::IMitsubaMetadata::LoaderName) == 0);
		globalMeta = static_cast<ext::MitsubaLoader::IMeshMetadata*>(meta)->globalMetadata;
	}

/*
	// set resolution
	if (globalMeta->sensors.size())
	{
		const auto& film = globalMeta->sensors.front().film;
		params.WindowSize.Width = film.width;
		params.WindowSize.Height = film.height;
	}
*/

	scene::ISceneManager* smgr = device->getSceneManager();
	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);

	video::IVideoDriver* driver = device->getVideoDriver();
	driver->setTextureCreationFlag(video::ETCF_ALWAYS_32_BIT, true);

	SimpleCallBack* cb = new SimpleCallBack();
	video::E_MATERIAL_TYPE nonInstanced = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../mesh.vert",
		"", "", "", //! No Geometry or Tessellation Shaders
		"../mesh.frag",
		3, video::EMT_SOLID, //! 3 vertices per primitive (this is tessellation shader relevant only
		cb, //! Our Shader Callback
		0); //! No custom user data
	cb->drop();

	//set up scene
	core::aabbox3df sceneBound;
	{
		core::vector<core::smart_refctd_ptr<scene::ISceneNode>> nodes;
		{
			auto contents = meshes.getContents();
			auto gpumeshes = driver->getGPUObjectsFromAssets<asset::ICPUMesh>(contents.first, contents.second);
			auto cpuit = contents.first;
			for (auto gpuit = gpumeshes->begin(); gpuit!=gpumeshes->end(); gpuit++,cpuit++)
			{
				auto* meta = (*cpuit)->getMetadata();
				assert(meta && core::strcmpi(meta->getLoaderName(),ext::MitsubaLoader::IMitsubaMetadata::LoaderName) == 0);
				const auto* meshmeta = static_cast<const ext::MitsubaLoader::IMeshMetadata*>(meta);
				const auto& instances = meshmeta->getInstances();

				const auto& gpumesh = *gpuit;
				for (auto i=0u; i<gpumesh->getMeshBufferCount(); i++)
					gpumesh->getMeshBuffer(i)->getMaterial().MaterialType = nonInstanced;

				for (auto instance : instances)
				{
					auto node = core::smart_refctd_ptr<scene::ISceneNode>(smgr->addMeshSceneNode(core::smart_refctd_ptr(gpumesh)));
					node->setRelativeTransformationMatrix(instance.getAsRetardedIrrlichtMatrix());
					sceneBound.addInternalBox(node->getTransformedBoundingBox());
					nodes.push_back(std::move(node));
				}
			}
		}

		// camera and viewport
		auto FBO_dimensions = params.WindowSize;

		auto extent = sceneBound.getExtent();
		auto camera = smgr->addCameraSceneNodeFPS(nullptr, 100.f, core::min(extent.X, extent.Y, extent.Z) * 0.0002f);
		//auto camera = smgr->addCameraSceneNode(nullptr);

		auto isOkSensorType = [](const ext::MitsubaLoader::CElementSensor& sensor) -> bool {
			return sensor.type == ext::MitsubaLoader::CElementSensor::Type::PERSPECTIVE || sensor.type == ext::MitsubaLoader::CElementSensor::Type::THINLENS;
		};
		if (globalMeta->sensors.size() && isOkSensorType(globalMeta->sensors.front()))
		{
			const auto& sensor = globalMeta->sensors.front();
			const auto& film = sensor.film;
			FBO_dimensions.set(film.cropWidth,film.cropHeight);

			// need to extract individual components
			bool leftHandedCamera = false;
			{
				auto relativeTransform = sensor.transform.matrix.extractSub3x4();
				if (relativeTransform.getPseudoDeterminant().x < 0.f)
					leftHandedCamera = true;

				auto pos = relativeTransform.getTranslation();
				camera->setPosition(pos.getAsVector3df());

				auto tpose = core::transpose(sensor.transform.matrix);
				auto up = tpose.rows[1];
				core::vectorSIMDf view = tpose.rows[2];
				auto target = view+pos;

				camera->setTarget(target.getAsVector3df());
				if (core::dot(core::normalize(core::cross(camera->getUpVector(),view)),core::cross(up,view)).x<0.99f)
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
			auto width = FBO_dimensions.Width;
			auto height = FBO_dimensions.Height;
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
			if (leftHandedCamera)
				camera->setProjectionMatrix(core::matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(realFoVDegrees), aspectRatio, persp->nearClip, persp->farClip));
			else
				camera->setProjectionMatrix(core::matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(realFoVDegrees), aspectRatio, persp->nearClip, persp->farClip));
		}
		else
		{
			camera->setNearValue(20.f);
			camera->setFarValue(5000.f);
		}
		smgr->setActiveCamera(camera);
		device->getCursorControl()->setVisible(false);

		auto framebuffer = driver->addFrameBuffer();
		framebuffer->attach(video::EFAP_DEPTH_ATTACHMENT, driver->createGPUTexture(video::ITexture::ETT_2D,&FBO_dimensions.Width,1,asset::EF_D32_SFLOAT).get());
		framebuffer->attach(video::EFAP_COLOR_ATTACHMENT0, driver->createGPUTexture(video::ITexture::ETT_2D,&FBO_dimensions.Width,1,asset::EF_A2B10G10R10_UNORM_PACK32).get());
		framebuffer->attach(video::EFAP_COLOR_ATTACHMENT1, driver->createGPUTexture(video::ITexture::ETT_2D,&FBO_dimensions.Width,1,asset::EF_R16G16_SNORM).get());

		uint64_t lastFPSTime = 0;
		float lastFastestMeshFrameNr = -1.f;

		//auto draw3DLine = ext::DebugDraw::CDraw3DLine::create(driver);
		while (device->run() && receiver.keepOpen())
		{
			driver->beginScene(false, false);


			driver->setRenderTarget(framebuffer);
			{ // clear
				driver->clearZBuffer();
				float zero[4] = { 0.f,0.f,0.f,0.f };
				driver->clearColorBuffer(video::EFAP_COLOR_ATTACHMENT0,zero);
				driver->clearColorBuffer(video::EFAP_COLOR_ATTACHMENT1,zero);
			}

			//! This animates (moves) the camera and sets the transforms
			//! Also draws the meshbuffer
			smgr->drawAll();
			/*
			for (auto node : nodes)
			{
				core::vector<std::pair<ext::DebugDraw::S3DLineVertex, ext::DebugDraw::S3DLineVertex>> lines;
				draw3DLine->enqueueBox(lines,node->getBoundingBox(),1.f,0.f,0.f,1.f, node->getAbsoluteTransformation());
				draw3DLine->draw(lines);
			}
			*/

			driver->blitRenderTargets(framebuffer,nullptr,false,false,{},{},true);


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

		driver->removeFrameBuffer(framebuffer);
	}

	// create a screenshot
	{
		core::rect<uint32_t> sourceRect(0, 0, params.WindowSize.Width, params.WindowSize.Height);
		ext::ScreenShot::dirtyCPUStallingScreenshot(device, "screenshot.png", sourceRect, asset::EF_R8G8B8_SRGB);
	}

	device->drop();
	return 0;
}
