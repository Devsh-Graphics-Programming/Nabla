// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include <chrono>
#include <filesystem>

#include "../common/QToQuitEventReceiver.h"

#include "../3rdparty/portable-file-dialogs/portable-file-dialogs.h"
#include "nbl/ext/MitsubaLoader/CMitsubaLoader.h"
#include "CommandLineHandler.hpp"

#include "CSceneNodeAnimatorCameraModifiedMaya.h"
#include "Renderer.h"

using namespace nbl;
using namespace core;

class RaytracerExampleEventReceiver : public nbl::IEventReceiver
{
	public:
		RaytracerExampleEventReceiver() 
			: running(true)
			, skipKeyPressed(false)
			, resetViewKeyPressed(false)
			, nextKeyPressed(false)
			, previousKeyPressed(false)
		{
		}

		bool OnEvent(const nbl::SEvent& event)
		{
			if (event.EventType == nbl::EET_KEY_INPUT_EVENT && !event.KeyInput.PressedDown)
			{
				switch (event.KeyInput.Key)
				{
					case ResetKey:
						resetViewKeyPressed = true;
						break;
					case NextKey:
						nextKeyPressed = true;
						break;
					case PreviousKey:
						previousKeyPressed = true;
						break;
					case SkipKey: // switch wire frame mode
						skipKeyPressed = true;
						break;
					case QuitKey: // switch wire frame mode
						running = false;
						return true;
					default:
						break;
				}
			}

			return false;
		}
		
		inline bool keepOpen() const { return running; }

		inline bool isSkipKeyPressed() const { return skipKeyPressed; }
		
		inline bool isResetViewPressed() const { return resetViewKeyPressed; }
		
		inline bool isNextPressed() const { return nextKeyPressed; }

		inline bool isPreviousPressed() const { return previousKeyPressed; }

		inline void resetKeys()
		{
			skipKeyPressed = false;
			resetViewKeyPressed = false;
			nextKeyPressed = false;
			previousKeyPressed = false;
		}

	private:
		static constexpr nbl::EKEY_CODE QuitKey = nbl::KEY_KEY_Q;
		static constexpr nbl::EKEY_CODE SkipKey = nbl::KEY_END;
		static constexpr nbl::EKEY_CODE ResetKey = nbl::KEY_HOME;
		static constexpr nbl::EKEY_CODE NextKey = nbl::KEY_PRIOR; // PAGE_UP
		static constexpr nbl::EKEY_CODE PreviousKey = nbl::KEY_NEXT; // PAGE_DOWN

		bool running = false;
		bool skipKeyPressed = false;
		bool resetViewKeyPressed = false;
		bool nextKeyPressed = false;
		bool previousKeyPressed = false;

};

int main(int argc, char** argv)
{
	std::vector<std::string> arguments;
	if (argc>1)
	{
		for (auto i = 1ul; i < argc; ++i)
			arguments.emplace_back(argv[i]);
	}

#ifdef TEST_ARGS
	arguments = std::vector<std::string> { 
		"-SCENE",
		"../../media/mitsuba/staircase2.zip",
		"scene.xml",
		"-TERMINATE",
		"-SCREENSHOT_OUTPUT_FOLDER",
		"\"C:\\Nabla-Screen-Shots\""
	};
#endif
	
	CommandLineHandler cmdHandler = CommandLineHandler(arguments);
	
	auto sceneDir = cmdHandler.getSceneDirectory();
	std::string filePath = (sceneDir.size() >= 1) ? sceneDir[0] : ""; // zip or xml
	std::string extraPath = (sceneDir.size() >= 2) ? sceneDir[1] : "";; // xml in zip
	bool shouldTerminateAfterRenders = cmdHandler.getTerminate(); // skip interaction with window and take screenshots only
	bool takeScreenShots = true;
	std::string mainFileName; // std::filesystem::path(filePath).filename().string();

	// create device with full flexibility over creation parameters
	// you can add more parameters if desired, check nbl::SIrrlichtCreationParameters
	nbl::SIrrlichtCreationParameters params;
	params.Bits = 24; //may have to set to 32bit for some platforms
	params.ZBufferBits = 24;
	params.DriverType = video::EDT_OPENGL;
	params.Fullscreen = false;
	params.Vsync = false;
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon
	params.WindowSize = dimension2d<uint32_t>(1920, 1080);
	auto device = createDeviceEx(params);
	if (!device)
		return 1; // could not create selected driver.
	
	//
	asset::SAssetBundle meshes;
	core::smart_refctd_ptr<const ext::MitsubaLoader::CMitsubaMetadata> globalMeta;
	{
		io::IFileSystem* fs = device->getFileSystem();
		asset::IAssetManager* am = device->getAssetManager();
		
		auto serializedLoader = core::make_smart_refctd_ptr<nbl::ext::MitsubaLoader::CSerializedLoader>(am);
		auto mitsubaLoader = core::make_smart_refctd_ptr<nbl::ext::MitsubaLoader::CMitsubaLoader>(am,fs);
		serializedLoader->initialize();
		mitsubaLoader->initialize();
		am->addAssetLoader(std::move(serializedLoader));
		am->addAssetLoader(std::move(mitsubaLoader));

		if(filePath.empty())
		{
			pfd::message("Choose file to load", "Choose mitsuba XML file to load or ZIP containing an XML. \nIf you cancel or choosen file fails to load, simple scene will be loaded.", pfd::choice::ok);
			pfd::open_file file("Choose XML or ZIP file", "../../media/mitsuba", { "ZIP files (.zip)", "*.zip", "XML files (.xml)", "*.xml"});
			if (!file.result().empty())
				filePath = file.result()[0];
		}

		if(filePath.empty())
			filePath = "../../media/mitsuba/staircase2.zip";
		
		mainFileName = std::filesystem::path(filePath).filename().string();
		mainFileName = mainFileName.substr(0u, mainFileName.find_first_of('.')); 
		
		std::cout << "\nSelected File = " << filePath << "\n" << std::endl;

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

				if(extraPath.empty())
				{
					uint32_t chosen = 0;

					// Don't ask for choosing file when there is only 1 available
					if(files.size() > 1)
					{
						std::cout << "Choose File (0-" << files.size() - 1ull << "):" << std::endl;
						for (auto i = 0u; i < files.size(); i++)
							std::cout << i << ": " << files[i].FullName.c_str() << std::endl;

						std::cin >> chosen;

						if (chosen >= files.size())
							chosen = 0u;
					}
					else if(files.size() >= 0)
					{
						std::cout << "The only available XML in zip Selected." << std::endl;
					}

					filePath = files[chosen].FullName.c_str();
					std::cout << "Selected XML File: "<< files[chosen].Name.c_str() << std::endl;
				}
				else
				{
					bool found = false;
					for (auto it=files.begin(); it!=files.end(); it++)
					{
						if(extraPath == std::string(it->Name.c_str()))
						{
							found = true;
							filePath = it->FullName.c_str();
							break;
						}
					}

					if(!found) {
						std::cout << "Cannot find requested file (" << extraPath.c_str() << ") in zip (" << filePath << ")" << std::endl;
						return 4;
					}
				}
			}
		}
		
		asset::CQuantNormalCache* qnc = am->getMeshManipulator()->getQuantNormalCache();

		//! read cache results -- speeds up mesh generation
		qnc->loadCacheFromFile<asset::EF_A2B10G10R10_SNORM_PACK32>(fs, "../../tmp/normalCache101010.sse");
		//! load the mitsuba scene
		meshes = am->getAsset(filePath, {});
		//! cache results -- speeds up mesh generation on second run
		qnc->saveCacheToFile<asset::EF_A2B10G10R10_SNORM_PACK32>(fs, "../../tmp/normalCache101010.sse");
		
		auto contents = meshes.getContents();
		if (!contents.size()) {
			std::cout << "[ERROR] Failed loading asset in " << filePath << ".";
			return 2;
		}

		globalMeta = core::smart_refctd_ptr<const ext::MitsubaLoader::CMitsubaMetadata>(meshes.getMetadata()->selfCast<const ext::MitsubaLoader::CMitsubaMetadata>());
		if (!globalMeta) {
			std::cout << "[ERROR] Couldn't get global Meta";
			return 3;
		}
	}
	
	constexpr float DefaultRotateSpeed = 300.0f;
	constexpr float DefaultZoomSpeed = 1.0f;
	constexpr float DefaultMoveSpeed = 100.0f;
	constexpr float DefaultSceneDiagonal = 50.0f; // reference for default zoom and move speed;

	struct SensorData
	{
		int32_t width = 0u;
		int32_t height = 0u;
		bool rightHandedCamera = true;
		uint32_t samplesNeeded = 0u;
		float moveSpeed = core::nan<float>();
		float stepZoomSpeed = core::nan<float>();
		float rotateSpeed = core::nan<float>();
		scene::ICameraSceneNode * staticCamera;
		scene::ICameraSceneNode * interactiveCamera;
		std::filesystem::path outputFilePath;
		ext::MitsubaLoader::CElementFilm::FileFormat fileFormat;

		scene::CSceneNodeAnimatorCameraModifiedMaya* getInteractiveCameraAnimator()
		{
			return reinterpret_cast<scene::CSceneNodeAnimatorCameraModifiedMaya*>(interactiveCamera->getAnimators()[0]);
		}

		void resetInteractiveCamera()
		{
			core::vectorSIMDf cameraTarget = staticCamera->getTarget();
			core::vector3df cameraTargetVec3f(cameraTarget.x, cameraTarget.y, cameraTarget.z); // I have to do this because of inconsistencies in using vectorSIMDf and vector3df in code most places.

			interactiveCamera->setPosition(staticCamera->getPosition());
			interactiveCamera->setTarget(cameraTargetVec3f);
			interactiveCamera->setUpVector(staticCamera->getUpVector());
			interactiveCamera->setLeftHanded(staticCamera->getLeftHanded());
			interactiveCamera->setProjectionMatrix(staticCamera->getProjectionMatrix());
		
			core::vectorSIMDf cameraPos; cameraPos.set(staticCamera->getPosition());
			auto modifiedMayaAnim = getInteractiveCameraAnimator();
			modifiedMayaAnim->setZoomAndRotationBasedOnTargetAndPosition(cameraPos, cameraTarget);
		}
	};

	auto smgr = device->getSceneManager();
	
	// When outputFilePath isn't set in Film Element in Mitsuba, use this to find the extension string.
	auto getFileExtensionFromFormat= [](ext::MitsubaLoader::CElementFilm::FileFormat format) -> std::string
	{
		std::string ret = "";
		using FileFormat = ext::MitsubaLoader::CElementFilm::FileFormat;
		switch (format) {
		case FileFormat::PNG:
			ret = ".png";
			break;
		case FileFormat::OPENEXR:
			ret = ".exr";
			break;
		case FileFormat::JPEG:
			ret = ".jpg";
			break;
		default: // TODO?
			break;
		}
		return ret;
	};

	auto isFileExtensionCompatibleWithFormat = [](std::string extension, ext::MitsubaLoader::CElementFilm::FileFormat format) -> bool
	{
		if(extension.empty())
			return false;

		if(extension[0] == '.')
			extension = extension.substr(1, extension.size());

		// TODO: get the supported extensions from loaders(?)
		using FileFormat = ext::MitsubaLoader::CElementFilm::FileFormat;
		switch (format) {
		case FileFormat::PNG:
			return extension == "png";
		case FileFormat::OPENEXR:
			return extension == "exr";
		case FileFormat::JPEG:
			return extension == "jpg" || extension == "jpeg" || extension == "jpe" || extension == "jif" || extension == "jfif" || extension == "jfi";
		default:
			return false;
		}
	};

	auto isOkSensorType = [](const ext::MitsubaLoader::CElementSensor& sensor) -> bool {
		return sensor.type == ext::MitsubaLoader::CElementSensor::Type::PERSPECTIVE || sensor.type == ext::MitsubaLoader::CElementSensor::Type::THINLENS;
	};

	std::vector<SensorData> sensors = std::vector<SensorData>(globalMeta->m_global.m_sensors.size());

	std::cout << "Total number of Sensors = " << sensors.size() << std::endl;

	if(sensors.empty())
	{
		std::cout << "[ERROR] No Sensors found." << std::endl;
		assert(false);
		return 5; // return code?
	}

	auto extractSensorData = [&](SensorData & outSensorData, const ext::MitsubaLoader::CElementSensor& sensor) -> bool
	{
		const auto& film = sensor.film;

		if(!isOkSensorType(sensor))
		{
			std::cout << "\tSensor Type is not valid" << std::endl;
			return false;
		}

		outSensorData.samplesNeeded = sensor.sampler.sampleCount;
		outSensorData.staticCamera = smgr->addCameraSceneNode(nullptr); 
		auto & staticCamera = outSensorData.staticCamera;
		
		std::cout << "\t SamplesPerPixelNeeded = " << outSensorData.samplesNeeded << std::endl;

		// need to extract individual components
		{
			auto relativeTransform = sensor.transform.matrix.extractSub3x4();
			if (relativeTransform.getPseudoDeterminant().x < 0.f)
				outSensorData.rightHandedCamera = false;
			else
				outSensorData.rightHandedCamera = true;
			
			std::cout << "\t IsRightHanded=" << ((outSensorData.rightHandedCamera) ? "TRUE" : "FALSE") << std::endl;

			auto pos = relativeTransform.getTranslation();
			staticCamera->setPosition(pos.getAsVector3df());
			
			std::cout << "\t Camera Position = <" << pos.x << "," << pos.y << "," << pos.z << ">" << std::endl;

			auto tpose = core::transpose(sensor.transform.matrix);

			auto up = tpose.rows[1];
			core::vectorSIMDf view = tpose.rows[2];
			auto target = view+pos;
			staticCamera->setTarget(target.getAsVector3df());

			std::cout << "\t Camera Target = <" << target.x << "," << target.y << "," << target.z << ">" << std::endl;

			if (core::dot(core::normalize(core::cross(staticCamera->getUpVector(),view)),core::cross(up,view)).x<0.99f)
				staticCamera->setUpVector(up);
		}
		
		const ext::MitsubaLoader::CElementSensor::PerspectivePinhole* persp = nullptr;
		switch (sensor.type)
		{
			case ext::MitsubaLoader::CElementSensor::Type::PERSPECTIVE:
				persp = &sensor.perspective;
				std::cout << "\t Type = PERSPECTIVE" << std::endl;
				break;
			case ext::MitsubaLoader::CElementSensor::Type::THINLENS:
				persp = &sensor.thinlens;
				std::cout << "\t Type = THINLENS" << std::endl;
				break;
			default:
				assert(false);
				break;
		}

		outSensorData.rotateSpeed = persp->rotateSpeed;
		outSensorData.stepZoomSpeed = persp->zoomSpeed;
		outSensorData.moveSpeed = persp->moveSpeed;

		if(core::isnan<float>(outSensorData.rotateSpeed))
		{
			outSensorData.rotateSpeed = DefaultRotateSpeed;
			std::cout << "\t Camera Rotate Speed = " << outSensorData.rotateSpeed << " = [Default Value]" << std::endl;
		}
		else
			std::cout << "\t Camera Rotate Speed = " << outSensorData.rotateSpeed << std::endl;

		if(core::isnan<float>(outSensorData.stepZoomSpeed))
			std::cout << "\t Camera Step Zoom Speed [Linear] = " << "[Value will be deduced from Scene Bounds] " << std::endl;
		else
			std::cout << "\t Camera Step Zoom Speed [Linear] = " << outSensorData.stepZoomSpeed << std::endl;
		
		if(core::isnan<float>(outSensorData.moveSpeed))
			std::cout << "\t Camera Move Speed = " << "[Value will be deduced from Scene Bounds] " << std::endl;
		else
			std::cout << "\t Camera Move Speed = " << outSensorData.moveSpeed << std::endl;

		
		float defaultZoomSpeedMultiplier = std::pow(DefaultSceneDiagonal, DefaultZoomSpeed / DefaultSceneDiagonal);
		outSensorData.interactiveCamera = smgr->addCameraSceneNodeModifiedMaya(nullptr, -1.0f * outSensorData.rotateSpeed, 50.0f, outSensorData.moveSpeed, -1, 2.0f, defaultZoomSpeedMultiplier, false, true);

		outSensorData.outputFilePath = std::filesystem::path(film.outputFilePath);
		outSensorData.fileFormat = film.fileFormat;
		if(!isFileExtensionCompatibleWithFormat(outSensorData.outputFilePath.extension().string(), outSensorData.fileFormat))
		{
			std::cout << "[ERROR] film.outputFilePath's extension is not compatible with film.fileFormat" << std::endl;
		}

		float realFoVDegrees;
		auto width = film.cropWidth;
		auto height = film.cropHeight;
		outSensorData.width = width;
		outSensorData.height = height;
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

		// TODO: apply the crop offset
		assert(film.cropOffsetX==0 && film.cropOffsetY==0);
		float nearClip = persp->nearClip;
		if(persp->farClip  > persp->nearClip * 10'000.0f)
			std::cout << "[WARN] Depth Range is too big: nearClip = " << persp->nearClip << ", farClip = " << persp->farClip << std::endl;
		if (outSensorData.rightHandedCamera)
			staticCamera->setProjectionMatrix(core::matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(realFoVDegrees), aspectRatio, nearClip, persp->farClip));
		else
			staticCamera->setProjectionMatrix(core::matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(realFoVDegrees), aspectRatio, nearClip, persp->farClip));

		outSensorData.resetInteractiveCamera();
		return true;
	};

	for(uint32_t s = 0u; s < sensors.size(); ++s)
	{
		std::cout << "Sensors[" << s << "] = " << std::endl;
		const auto& sensor = globalMeta->m_global.m_sensors[s];
		auto & outSensorData = sensors[s];
		extractSensorData(outSensorData, sensor);
	}

#if INJECT_TEST_SENSOR
	std::cout << "New Injected Sensors[0] = " << std::endl;
	SensorData newSensor = {};
	extractSensorData(newSensor, globalMeta->m_global.m_sensors[0]);
	newSensor.staticCamera->setPosition(core::vector3df(0.0f,2.0f,0.0f));
	newSensor.staticCamera->setTarget(core::vector3df(-0.900177f, 2.0f, -0.435524f));
	core::vectorSIMDf UpVector(0.0f, 1.0f, 0.0f);
	newSensor.staticCamera->setUpVector(UpVector);
	newSensor.staticCamera->render(); // It's not actually "render" :| It's basically recomputeViewMatrix ;
	
	newSensor.resetInteractiveCamera();
	sensors.push_back(newSensor);

	sensors[0].samplesNeeded = 4u;
	sensors[1].samplesNeeded = 4u;
#endif

	auto driver = device->getVideoDriver();

	core::smart_refctd_ptr<Renderer> renderer = core::make_smart_refctd_ptr<Renderer>(driver,device->getAssetManager(),smgr);
	constexpr uint32_t MaxSamples = MAX_ACCUMULATED_SAMPLES;
	auto sampleSequence = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(uint32_t)*MaxSamples*Renderer::MaxDimensions);
	{
		bool generateNewSamples = true;

		io::IReadFile* cacheFile = device->getFileSystem()->createAndOpenFile("../../tmp/rtSamples.bin");
		if (cacheFile)
		{
			if (cacheFile->getSize()>=sampleSequence->getSize()) // light validation
			{
				cacheFile->read(sampleSequence->getPointer(),sampleSequence->getSize());
				generateNewSamples = false;
			}
			cacheFile->drop();
		}

		if (generateNewSamples)
		{
			/** TODO: move into the renderer and redo the sampling (compress into R21G21B21_UINT)
			Locality Level 0: the 3 dimensions consumed for a BxDF or NEE sample
			Locality Level 1: the k = 3 (1 + NEE) samples which will be consumed in the same invocation
			Locality Level 2-COMP: the N = k dispatchSPP Resolution samples consumed by a raygen dispatch (another TODO: would be order CS and everything in a morton curve)
			Locality Level 2-RTX: the N = k Depth samples consumed as we recurse deeper
			Locality Level 3: the D = k dispatchSPP Resolution Depth samples consumed as we accumuate more samples
			**/
			constexpr uint32_t Channels = 3u;
			static_assert(Renderer::MaxDimensions%Channels==0u,"We cannot have this!");
			core::OwenSampler sampler(Renderer::MaxDimensions,0xdeadbeefu);

			uint32_t (&out)[][Channels] = *reinterpret_cast<uint32_t(*)[][Channels]>(sampleSequence->getPointer());
			for (auto realdim=0u; realdim<Renderer::MaxDimensions/Channels; realdim++)
			for (auto c=0u; c<Channels; c++)
			for (uint32_t i=0; i<MaxSamples; i++)
				out[realdim*MaxSamples+i][c] = sampler.sample(realdim*Channels+c,i);

			io::IWriteFile* cacheFile = device->getFileSystem()->createAndWriteFile("../../tmp/rtSamples.bin");
			if (cacheFile)
			{
				cacheFile->write(sampleSequence->getPointer(),sampleSequence->getSize());
				cacheFile->drop();
			}
		}
	}

	renderer->initSceneResources(meshes);
	meshes = {}; // free memory
	
	RaytracerExampleEventReceiver receiver;
	device->setEventReceiver(&receiver);

	// Deduce Move and Zoom Speeds if it is nan
	auto sceneBoundsExtent = renderer->getSceneBound().getExtent();
	auto sceneDiagonal = sceneBoundsExtent.getLength(); 

	for(uint32_t s = 0u; s < sensors.size(); ++s)
	{
		auto& sensorData = sensors[s];
		
		float linearStepZoomSpeed = sensorData.stepZoomSpeed;
		if(core::isnan<float>(sensorData.stepZoomSpeed))
		{
			linearStepZoomSpeed = sceneDiagonal * (DefaultZoomSpeed / DefaultSceneDiagonal);
		}

		// Set Zoom Multiplier
		{
			float logarithmicZoomSpeed = std::pow(sceneDiagonal, linearStepZoomSpeed / sceneDiagonal);
			sensorData.stepZoomSpeed =  logarithmicZoomSpeed;
			sensorData.getInteractiveCameraAnimator()->setStepZoomMultiplier(logarithmicZoomSpeed);
			printf("[INFO] Sensor[%d] Camera Step Zoom Speed deduced from scene bounds = %f [Linear], %f [Logarithmic] \n", s, linearStepZoomSpeed, logarithmicZoomSpeed);
		}

		if(core::isnan<float>(sensorData.moveSpeed))
		{
			float newMoveSpeed = DefaultMoveSpeed * (sceneDiagonal / DefaultSceneDiagonal);
			sensorData.moveSpeed = newMoveSpeed;
			sensorData.getInteractiveCameraAnimator()->setMoveSpeed(newMoveSpeed);
			printf("[INFO] Sensor[%d] Camera Move Speed deduced from scene bounds = %f\n", s, newMoveSpeed);
		}
		
		assert(!core::isnan<float>(sensorData.getInteractiveCameraAnimator()->getRotateSpeed()));
		assert(!core::isnan<float>(sensorData.getInteractiveCameraAnimator()->getStepZoomSpeed()));
		assert(!core::isnan<float>(sensorData.getInteractiveCameraAnimator()->getMoveSpeed()));
	}


	// Render To file
	int32_t prevWidth = 0;
	int32_t prevHeight = 0;
	for(uint32_t s = 0u; s < sensors.size(); ++s)
	{
		if(!receiver.keepOpen())
			break;

		const auto& sensorData = sensors[s];
		
		printf("[INFO] Rendering %s - Sensor(%d) to file.\n", filePath.c_str(), s);

		bool needsReinit = (prevWidth != sensorData.width) || (prevHeight != sensorData.height); // >= or !=
		prevWidth = sensorData.width;
		prevHeight = sensorData.height;
		
		renderer->resetSampleAndFrameCounters(); // so that renderer->getTotalSamplesPerPixelComputed is 0 at the very beginning
		if(needsReinit)
		{
			renderer->deinitScreenSizedResources();
			renderer->initScreenSizedResources(sensorData.width, sensorData.height, std::move(sampleSequence));
		}
		
		smgr->setActiveCamera(sensorData.staticCamera);

		const uint32_t samplesPerPixelPerDispatch = renderer->getSamplesPerPixelPerDispatch();
		const uint32_t maxNeededIterations = (sensorData.samplesNeeded + samplesPerPixelPerDispatch - 1) / samplesPerPixelPerDispatch;
		
		uint32_t itr = 0u;
		bool takenEnoughSamples = false;

		while(!takenEnoughSamples && (device->run() && !receiver.isSkipKeyPressed() && receiver.keepOpen()))
		{
			if(itr >= maxNeededIterations)
				std::cout << "[ERROR] Samples taken (" << renderer->getTotalSamplesPerPixelComputed() << ") must've exceeded samples needed for Sensor (" << sensorData.samplesNeeded << ") by now; something is wrong." << std::endl;

			driver->beginScene(false, false);
			renderer->render(device->getTimer());
			auto oldVP = driver->getViewPort();
			driver->blitRenderTargets(renderer->getColorBuffer(),nullptr,false,false,{},{},true);
			driver->setViewPort(oldVP);

			driver->endScene();
			
			if(renderer->getTotalSamplesPerPixelComputed() >= sensorData.samplesNeeded)
				takenEnoughSamples = true;
			
			itr++;
		}

		auto screenshotFilePath = sensorData.outputFilePath;
		if (screenshotFilePath.empty())
		{
			auto extensionStr = getFileExtensionFromFormat(sensorData.fileFormat);
			screenshotFilePath = std::filesystem::path("ScreenShot_" + mainFileName + "_Sensor_" + std::to_string(s) + extensionStr);
		}
		
		renderer->takeAndSaveScreenShot(screenshotFilePath);

		int progress = float(renderer->getTotalSamplesPerPixelComputed())/float(sensorData.samplesNeeded) * 100;
		printf("[INFO] Rendered Successfully - %d%% Progress = %u/%u SamplesPerPixel - FileName = %s. \n", progress, renderer->getTotalSamplesPerPixelComputed(), sensorData.samplesNeeded, screenshotFilePath.filename().string().c_str());

		receiver.resetKeys();
	}

	// Interactive
	if(!shouldTerminateAfterRenders && receiver.keepOpen())
	{
		int activeSensor = -1; // that outputs to current window when not in TERMIANTE mode.

		auto setActiveSensor = [&](int index) 
		{
			if(index >= 0 && index < sensors.size())
			{
				bool needsReinit = (activeSensor == -1) || (sensors[activeSensor].width != sensors[index].width) || (sensors[activeSensor].height != sensors[index].height); // should be >= or != ?
				activeSensor = index;

				renderer->resetSampleAndFrameCounters();
				if(needsReinit)
				{
					renderer->deinitScreenSizedResources();
					renderer->initScreenSizedResources(sensors[activeSensor].width, sensors[activeSensor].height, std::move(sampleSequence));
				}

				smgr->setActiveCamera(sensors[activeSensor].interactiveCamera);
				std::cout << "Active Sensor = " << activeSensor << std::endl;
			}
		};

		setActiveSensor(0);

		uint64_t lastFPSTime = 0;
		auto start = std::chrono::steady_clock::now();
		while (device->run() && receiver.keepOpen())
		{
			// Handle Inputs
			{
				if(receiver.isResetViewPressed())
				{
					sensors[activeSensor].resetInteractiveCamera();
					std::cout << "Interactive Camera Position and Target has been Reset." << std::endl;
				}
				if(receiver.isNextPressed())
				{
					setActiveSensor(activeSensor + 1);
				}
				if(receiver.isPreviousPressed())
				{
					setActiveSensor(activeSensor - 1);
				}
				receiver.resetKeys();
			}

			driver->beginScene(false, false);
			
			renderer->render(device->getTimer());

			auto oldVP = driver->getViewPort();
			driver->blitRenderTargets(renderer->getColorBuffer(),nullptr,false,false,{},{},true);
			driver->setViewPort(oldVP);

			driver->endScene();

			// display frames per second in window title
			uint64_t time = device->getTimer()->getRealTime();
			if (time - lastFPSTime > 1000)
			{
				std::wostringstream str;
				auto samples = renderer->getTotalSamplesComputed();
				auto rays = renderer->getTotalRaysCast();
				str << L"Raytraced Shadows Demo - Nabla Engine   MegaSamples: " << samples/1000000ull << "   MRay/s: "
					<< double(rays)/double(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count());

				device->setWindowCaption(str.str());
				lastFPSTime = time;
			}
		}
		
		auto extensionStr = getFileExtensionFromFormat(sensors[activeSensor].fileFormat);
		renderer->takeAndSaveScreenShot(std::filesystem::path("LastView_" + mainFileName + "_Sensor_" + std::to_string(activeSensor) + extensionStr));
		renderer->deinitScreenSizedResources();
	}

	renderer->deinitSceneResources();
	renderer = nullptr;

	return 0;
}