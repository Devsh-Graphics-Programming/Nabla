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
		RaytracerExampleEventReceiver() : running(true), renderingBeauty(true)
		{
			resetKeys();
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
					case ScreenshotKey:
						screenshotKeyPressed = true;
						break;
					case LogProgressKey:
						logProgressKeyPressed = true;
						break;
					case SkipKey:
						skipKeyPressed = true;
						break;
					case BeautyKey:
						renderingBeauty = !renderingBeauty;
						break;
					case QuitKey:
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

		inline bool isScreenshotKeyPressed() const { return screenshotKeyPressed; }

		inline bool isLogProgressKeyPressed() const { return logProgressKeyPressed; }

		inline bool isRenderingBeauty() const { return renderingBeauty; }

		inline void resetKeys()
		{
			skipKeyPressed = false;
			resetViewKeyPressed = false;
			nextKeyPressed = false;
			previousKeyPressed = false;
			screenshotKeyPressed = false;
			logProgressKeyPressed = false;
		}

	private:
		static constexpr nbl::EKEY_CODE QuitKey = nbl::KEY_KEY_Q;
		static constexpr nbl::EKEY_CODE SkipKey = nbl::KEY_END;
		static constexpr nbl::EKEY_CODE ResetKey = nbl::KEY_HOME;
		static constexpr nbl::EKEY_CODE NextKey = nbl::KEY_PRIOR; // PAGE_UP
		static constexpr nbl::EKEY_CODE PreviousKey = nbl::KEY_NEXT; // PAGE_DOWN
		static constexpr nbl::EKEY_CODE ScreenshotKey = nbl::KEY_KEY_P;
		static constexpr nbl::EKEY_CODE LogProgressKey = nbl::KEY_KEY_L;
		static constexpr nbl::EKEY_CODE BeautyKey = nbl::KEY_KEY_B;

		bool running;
		bool renderingBeauty;

		bool skipKeyPressed;
		bool resetViewKeyPressed;
		bool nextKeyPressed;
		bool previousKeyPressed;
		bool screenshotKeyPressed;
		bool logProgressKeyPressed;
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
	
	// will leak it because there's no cross platform input!
	std::thread cin_thread;

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
					uint32_t chosen = 0xffffffffu;

					// Don't ask for choosing file when there is only 1 available
					if(files.size() > 1)
					{
						std::cout << "Choose File (0-" << files.size() - 1ull << "):" << std::endl;
						for (auto i = 0u; i < files.size(); i++)
							std::cout << i << ": " << files[i].FullName.c_str() << std::endl;

						// std::cin with timeout
						{
							std::atomic<bool> started = false;
							cin_thread = std::thread([&chosen,&started]()
							{
								started = true;
								std::cin >> chosen;
							});
							const auto end = std::chrono::steady_clock::now()+std::chrono::seconds(10u);
							while (!started || chosen==0xffffffffu && std::chrono::steady_clock::now()<end) {}
						}
					}
					else if(files.size() >= 0)
					{
						std::cout << "The only available XML in zip Selected." << std::endl;
					}
					
					if (chosen >= files.size())
						chosen = 0u;

					filePath = files[chosen].FullName.c_str();
					std::cout << "Selected XML File: "<< files[chosen].Name.c_str() << std::endl;
					mainFileName += std::string("_") + std::filesystem::path(files[chosen].Name.c_str()).replace_extension().string();
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
							mainFileName += std::string("_") + std::filesystem::path(it->Name.c_str()).replace_extension().string();
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
		ext::MitsubaLoader::CElementSensor::Type type;
		ext::MitsubaLoader::CElementFilm::FileFormat fileFormat;
		Renderer::DenoiserArgs denoiserInfo = {};
		int32_t highQualityEdges = 0u;
		bool envmap = false;

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
	
	struct CubemapRender
	{
		uint32_t sensorIdx = 0u;
		uint32_t getSensorsBeginIdx() const { return sensorIdx; }
		uint32_t getSensorsEndIdx() const { return sensorIdx + 5; }
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

	std::cout << "Total number of Sensors = " << globalMeta->m_global.m_sensors.size() << std::endl;

	if(globalMeta->m_global.m_sensors.empty())
	{
		std::cout << "[ERROR] No Sensors found." << std::endl;
		assert(false);
		return 5; // return code?
	}
	
	const bool shouldHaveSensorIdxInFileName = globalMeta->m_global.m_sensors.size() > 1;
	std::vector<SensorData> sensors = std::vector<SensorData>();
	std::vector<CubemapRender> cubemapRenders = std::vector<CubemapRender>();

	auto extractAndAddToSensorData = [&](const ext::MitsubaLoader::CElementSensor& sensor, uint32_t idx) -> bool
	{
		SensorData mainSensorData = {};

		const auto& film = sensor.film;
		mainSensorData.denoiserInfo.bloomFilePath = std::filesystem::path(film.denoiserBloomFilePath);
		mainSensorData.denoiserInfo.bloomScale = film.denoiserBloomScale;
		mainSensorData.denoiserInfo.bloomIntensity = film.denoiserBloomIntensity;
		mainSensorData.denoiserInfo.tonemapperArgs = std::string(film.denoiserTonemapperArgs);
		mainSensorData.fileFormat = film.fileFormat;
		mainSensorData.highQualityEdges = film.highQualityEdges;
		mainSensorData.outputFilePath = std::filesystem::path(film.outputFilePath);
		if(!isFileExtensionCompatibleWithFormat(mainSensorData.outputFilePath.extension().string(), mainSensorData.fileFormat))
		{
			std::cout << "[ERROR] film.outputFilePath's extension is not compatible with film.fileFormat" << std::endl;
		}
		// handle missing output path
		if (mainSensorData.outputFilePath.empty())
		{
			auto extensionStr = getFileExtensionFromFormat(mainSensorData.fileFormat);
			if(shouldHaveSensorIdxInFileName)
				mainSensorData.outputFilePath = std::filesystem::path("Render_" + mainFileName + "_Sensor_" + std::to_string(idx) + extensionStr);
			else
				mainSensorData.outputFilePath = std::filesystem::path("Render_" + mainFileName + extensionStr);
		}

		mainSensorData.samplesNeeded = sensor.sampler.sampleCount;
		std::cout << "\t SamplesPerPixelNeeded = " << mainSensorData.samplesNeeded << std::endl;

		const ext::MitsubaLoader::CElementSensor::PerspectivePinhole* persp = nullptr;
		const ext::MitsubaLoader::CElementSensor::Orthographic* ortho = nullptr;
		const ext::MitsubaLoader::CElementSensor::CameraBase* cameraBase = nullptr;
		switch (sensor.type)
		{
			case ext::MitsubaLoader::CElementSensor::Type::PERSPECTIVE:
				persp = &sensor.perspective;
				cameraBase = persp;
				std::cout << "\t Type = PERSPECTIVE" << std::endl;
				break;
			case ext::MitsubaLoader::CElementSensor::Type::THINLENS:
				persp = &sensor.thinlens;
				cameraBase = persp;
				std::cout << "\t Type = THINLENS" << std::endl;
				break;
			case ext::MitsubaLoader::CElementSensor::Type::ORTHOGRAPHIC:
				ortho = &sensor.orthographic;
				cameraBase = ortho;
				std::cout << "\t Type = ORTHOGRAPHIC" << std::endl;
				break;
			case ext::MitsubaLoader::CElementSensor::Type::TELECENTRIC:
				ortho = &sensor.telecentric;
				cameraBase = ortho;
				std::cout << "\t Type = TELECENTRIC" << std::endl;
				break;
			case ext::MitsubaLoader::CElementSensor::Type::SPHERICAL:
				cameraBase = &sensor.spherical;
				std::cout << "\t Type = SPHERICAL" << std::endl;
				break;
			default:
				std::cout << "\tSensor Type is not valid" << std::endl;
				return false;
		}
		mainSensorData.type = sensor.type;
		mainSensorData.rotateSpeed = cameraBase->rotateSpeed;
		mainSensorData.stepZoomSpeed = cameraBase->zoomSpeed;
		mainSensorData.moveSpeed = cameraBase->moveSpeed;
		
		if(core::isnan<float>(mainSensorData.rotateSpeed))
		{
			mainSensorData.rotateSpeed = DefaultRotateSpeed;
			std::cout << "\t Camera Rotate Speed = " << mainSensorData.rotateSpeed << " = [Default Value]" << std::endl;
		}
		else
			std::cout << "\t Camera Rotate Speed = " << mainSensorData.rotateSpeed << std::endl;

		if(core::isnan<float>(mainSensorData.stepZoomSpeed))
			std::cout << "\t Camera Step Zoom Speed [Linear] = " << "[Value will be deduced from Scene Bounds] " << std::endl;
		else
			std::cout << "\t Camera Step Zoom Speed [Linear] = " << mainSensorData.stepZoomSpeed << std::endl;
		
		if(core::isnan<float>(mainSensorData.moveSpeed))
			std::cout << "\t Camera Move Speed = " << "[Value will be deduced from Scene Bounds] " << std::endl;
		else
			std::cout << "\t Camera Move Speed = " << mainSensorData.moveSpeed << std::endl;
		
		float defaultZoomSpeedMultiplier = std::pow(DefaultSceneDiagonal, DefaultZoomSpeed / DefaultSceneDiagonal);
		mainSensorData.interactiveCamera = smgr->addCameraSceneNodeModifiedMaya(nullptr, -1.0f * mainSensorData.rotateSpeed, 50.0f, mainSensorData.moveSpeed, -1, 2.0f, defaultZoomSpeedMultiplier, false, true);
		
		nbl::core::vectorSIMDf mainCamPos;
		nbl::core::vectorSIMDf mainCamUp;
		nbl::core::vectorSIMDf mainCamView;
		// need to extract individual components from matrix to camera
		{
			auto relativeTransform = sensor.transform.matrix.extractSub3x4();
			if (relativeTransform.getPseudoDeterminant().x < 0.f)
				mainSensorData.rightHandedCamera = false;
			else
				mainSensorData.rightHandedCamera = true;
			
			std::cout << "\t IsRightHanded=" << ((mainSensorData.rightHandedCamera) ? "TRUE" : "FALSE") << std::endl;

			mainCamPos = relativeTransform.getTranslation();
			
			std::cout << "\t Camera Position = <" << mainCamPos.x << "," << mainCamPos.y << "," << mainCamPos.z << ">" << std::endl;

			auto tpose = core::transpose(sensor.transform.matrix);
			mainCamUp = tpose.rows[1];
			mainCamView = tpose.rows[2];
		}
		
		float realFoVDegrees;
		auto width = film.cropWidth;
		auto height = film.cropHeight;
		mainSensorData.width = width;
		mainSensorData.height = height;
		float aspectRatio = float(width) / float(height);
		auto convertFromXFoV = [=](float fov) -> float
		{
			float aspectX = tan(core::radians(fov)*0.5f);
			return core::degrees(atan(aspectX/aspectRatio)*2.f);
		};

		// TODO: apply the crop offset
		assert(film.cropOffsetX==0 && film.cropOffsetY==0);
		
		float nearClip = cameraBase->nearClip;
		float farClip = cameraBase->farClip;
		if(farClip > nearClip * 10'000.0f)
			std::cout << "[WARN] Depth Range is too big: nearClip = " << nearClip << ", farClip = " << farClip << std::endl;

		if (mainSensorData.type == ext::MitsubaLoader::CElementSensor::Type::SPHERICAL)
		{
			nbl::core::vectorSIMDf camViews[6] =
			{
				nbl::core::vectorSIMDf(+1, 0, 0, 0), // +X
				nbl::core::vectorSIMDf(-1, 0, 0, 0), // -X
				nbl::core::vectorSIMDf(0, +1, 0, 0), // +Y
				nbl::core::vectorSIMDf(0, -1, 0, 0), // -Y
				nbl::core::vectorSIMDf(0, 0, +1, 0), // +Z
				nbl::core::vectorSIMDf(0, 0, -1, 0), // -Z
			};

			if(!mainSensorData.rightHandedCamera)
			{
				camViews[0] *= -1;
				camViews[1] *= -1;
			}
			
			const nbl::core::vectorSIMDf upVectors[6] =
			{
				nbl::core::vectorSIMDf(0, +1, 0, 0), // +Y
				nbl::core::vectorSIMDf(0, +1, 0, 0), // +Y
				nbl::core::vectorSIMDf(0, 0, +1, 0), // -Z
				nbl::core::vectorSIMDf(0, 0, -1, 0), // +Z
				nbl::core::vectorSIMDf(0, +1, 0, 0), // +Y
				nbl::core::vectorSIMDf(0, +1, 0, 0), // +Y
			};

			CubemapRender cubemapRender = {};
			cubemapRender.sensorIdx = sensors.size();
			cubemapRenders.push_back(cubemapRender);

			for(uint32_t i = 0; i < 6; ++i)
			{
				SensorData cubemapFaceSensorData = mainSensorData;
				cubemapFaceSensorData.envmap = true;

				if(mainSensorData.width != mainSensorData.height)
				{
					std::cout << "[ERROR] Cannot generate cubemap faces where film.width and film.height are not equal. (Aspect Ration must be 1)" << std::endl;
					assert(false);
				}
				const auto baseResolution = core::max(mainSensorData.width,mainSensorData.height);
				cubemapFaceSensorData.width = baseResolution + mainSensorData.highQualityEdges * 2;
				cubemapFaceSensorData.height = baseResolution + mainSensorData.highQualityEdges * 2;

				// FIXME: suffix added after extension
				cubemapFaceSensorData.outputFilePath.replace_extension();
				constexpr const char* suffixes[6] =
				{
					"_x+.exr",
					"_x-.exr",
					"_y+.exr",
					"_y-.exr",
					"_z+.exr",
					"_z-.exr",
				};
				cubemapFaceSensorData.outputFilePath += suffixes[i];

				cubemapFaceSensorData.staticCamera = smgr->addCameraSceneNode(nullptr); 
				auto& staticCamera = cubemapFaceSensorData.staticCamera;
				
				const auto& camView = camViews[i];
				const auto& upVector = upVectors[i];

				staticCamera->setPosition(mainCamPos.getAsVector3df());
				staticCamera->setTarget((mainCamPos + camView).getAsVector3df());
				staticCamera->setUpVector(upVector);

				// auto fov = core::radians(90.0f);
				auto fov = atanf(float(cubemapFaceSensorData.width) / float(mainSensorData.width)) * 2.0f;
				auto aspectRatio = 1.0f;
				
				if (mainSensorData.rightHandedCamera)
					staticCamera->setProjectionMatrix(core::matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(fov, 1.0f, nearClip, farClip));
				else
					staticCamera->setProjectionMatrix(core::matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(fov, 1.0f, nearClip, farClip));
				
				cubemapFaceSensorData.interactiveCamera = smgr->addCameraSceneNodeModifiedMaya(nullptr, -1.0f * mainSensorData.rotateSpeed, 50.0f, mainSensorData.moveSpeed, -1, 2.0f, defaultZoomSpeedMultiplier, false, true);
				cubemapFaceSensorData.resetInteractiveCamera();
				sensors.push_back(cubemapFaceSensorData);
			}
		}
		else
		{
			mainSensorData.staticCamera = smgr->addCameraSceneNode(nullptr); 
			auto& staticCamera = mainSensorData.staticCamera;

			staticCamera->setPosition(mainCamPos.getAsVector3df());
			
			{
				auto target = mainCamView+mainCamPos;
				std::cout << "\t Camera Target = <" << target.x << "," << target.y << "," << target.z << ">" << std::endl;
				staticCamera->setTarget(target.getAsVector3df());
			}

			if (core::dot(core::normalize(core::cross(staticCamera->getUpVector(),mainCamView)),core::cross(mainCamUp,mainCamView)).x<0.99f)
				staticCamera->setUpVector(mainCamUp);

			//
			if (ortho)
			{
				const auto scale = sensor.transform.matrix.extractSub3x4().getScale();
				const float volumeX = 2.f*scale.x;
				const float volumeY = (2.f/aspectRatio)*scale.y;
				if (mainSensorData.rightHandedCamera)
					staticCamera->setProjectionMatrix(core::matrix4SIMD::buildProjectionMatrixOrthoRH(volumeX, volumeY, nearClip, farClip));
				else
					staticCamera->setProjectionMatrix(core::matrix4SIMD::buildProjectionMatrixOrthoLH(volumeX, volumeY, nearClip, farClip));
			}
			else if (persp)
			{
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

				if (mainSensorData.rightHandedCamera)
					staticCamera->setProjectionMatrix(core::matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(realFoVDegrees), aspectRatio, nearClip, farClip));
				else
					staticCamera->setProjectionMatrix(core::matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(realFoVDegrees), aspectRatio, nearClip, farClip));
			}
			else
			{
				assert(false);
			}

			mainSensorData.resetInteractiveCamera();
			sensors.push_back(mainSensorData);
		}

		return true;
	};

	for(uint32_t s = 0u; s < globalMeta->m_global.m_sensors.size(); ++s)
	{
		std::cout << "Sensors[" << s << "] = " << std::endl;
		const auto& sensor = globalMeta->m_global.m_sensors[s];
		extractAndAddToSensorData(sensor, s);
	}

	auto driver = device->getVideoDriver();

	core::smart_refctd_ptr<Renderer> renderer = core::make_smart_refctd_ptr<Renderer>(driver,device->getAssetManager(),smgr);
	renderer->initSceneResources(meshes,"LowDiscrepancySequenceCache.bin");
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
		//assert(!core::isnan<float>(sensorData.getInteractiveCameraAnimator()->getStepZoomSpeed()));
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
			renderer->initScreenSizedResources(sensorData.width,sensorData.height);
		}
		
		smgr->setActiveCamera(sensorData.staticCamera);

		const uint32_t samplesPerPixelPerDispatch = renderer->getSamplesPerPixelPerDispatch();
		const uint32_t maxNeededIterations = (sensorData.samplesNeeded + samplesPerPixelPerDispatch - 1) / samplesPerPixelPerDispatch;
		
		uint32_t itr = 0u;
		bool takenEnoughSamples = false;
		bool renderFailed = false;
		while(!takenEnoughSamples && (device->run() && !receiver.isSkipKeyPressed() && receiver.keepOpen()))
		{
			if(itr >= maxNeededIterations)
				std::cout << "[ERROR] Samples taken (" << renderer->getTotalSamplesPerPixelComputed() << ") must've exceeded samples needed for Sensor (" << sensorData.samplesNeeded << ") by now; something is wrong." << std::endl;

			// Handle Inputs
			{
				if(receiver.isLogProgressKeyPressed())
				{
					int progress = float(renderer->getTotalSamplesPerPixelComputed())/float(sensorData.samplesNeeded) * 100;
					printf("[INFO] Rendering in progress - %d%% Progress = %u/%u SamplesPerPixel. \n", progress, renderer->getTotalSamplesPerPixelComputed(), sensorData.samplesNeeded);
				}
				receiver.resetKeys();
			}


			driver->beginScene(false, false);

			if(!renderer->render(device->getTimer(),!sensorData.envmap))
			{
				renderFailed = true;
				driver->endScene();
				break;
			}

			auto oldVP = driver->getViewPort();
			driver->blitRenderTargets(renderer->getColorBuffer(),nullptr,false,false,{},{},true);
			driver->setViewPort(oldVP);

			driver->endScene();
			
			if(renderer->getTotalSamplesPerPixelComputed() >= sensorData.samplesNeeded)
				takenEnoughSamples = true;
			
			itr++;
		}

		auto screenshotFilePath = sensorData.outputFilePath;
		
		if(renderFailed)
		{
			std::cout << "[ERROR] Render Failed." << std::endl;
		}
		else
		{
			bool shouldDenoise = sensorData.type != ext::MitsubaLoader::CElementSensor::Type::SPHERICAL;
			renderer->takeAndSaveScreenShot(screenshotFilePath, shouldDenoise, sensorData.denoiserInfo);
			int progress = float(renderer->getTotalSamplesPerPixelComputed())/float(sensorData.samplesNeeded) * 100;
			printf("[INFO] Rendered Successfully - %d%% Progress = %u/%u SamplesPerPixel - FileName = %s. \n", progress, renderer->getTotalSamplesPerPixelComputed(), sensorData.samplesNeeded, screenshotFilePath.filename().string().c_str());
		}

		receiver.resetKeys();
	}

	// Denoise Cubemaps that weren't denoised seperately
	for(uint32_t i = 0; i < cubemapRenders.size(); ++i)
	{
		uint32_t beginIdx = cubemapRenders[i].getSensorsBeginIdx();
		assert(beginIdx + 6 <= sensors.size());
		auto borderPixels = sensors[beginIdx].highQualityEdges;

		std::filesystem::path filePaths[6] = {};

		for(uint32_t f = beginIdx; f < beginIdx + 6; ++f)
		{
			const auto & sensor = sensors[f];
			filePaths[f] = sensor.outputFilePath;
		}

		std::string mergedFileName = "Merge_CubeMap_" + mainFileName;
		renderer->denoiseCubemapFaces(filePaths, mergedFileName, borderPixels, sensors[beginIdx].denoiserInfo);
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
					renderer->initScreenSizedResources(sensors[activeSensor].width,sensors[activeSensor].height);
				}

				smgr->setActiveCamera(sensors[activeSensor].interactiveCamera);
				std::cout << "Active Sensor = " << activeSensor << std::endl;
			}
		};

		setActiveSensor(0);

		uint64_t lastFPSTime = 0;
		auto start = std::chrono::steady_clock::now();
		bool renderFailed = false;
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
				if(receiver.isScreenshotKeyPressed())
				{
					const std::string screenShotFilesPrefix = "ScreenShot";
					const char seperator = '_';
					int maxFileNumber = -1;
					for (const auto & entry : std::filesystem::directory_iterator(std::filesystem::current_path()))
					{
						const auto entryPathStr = entry.path().filename().string();
						const auto firstSeperatorLoc = entryPathStr.find_first_of(seperator) ;
						const auto lastSeperatorLoc = entryPathStr.find_last_of(seperator);
						const auto firstDotLoc = entryPathStr.find_first_of('.');

						const auto firstSection = entryPathStr.substr(0u, firstSeperatorLoc);
						const bool isScreenShot = (firstSection == screenShotFilesPrefix);
						if(isScreenShot)
						{
							const auto middleSection = entryPathStr.substr(firstSeperatorLoc + 1, lastSeperatorLoc - (firstSeperatorLoc + 1));
							const auto numberString = entryPathStr.substr(lastSeperatorLoc + 1, firstDotLoc - (lastSeperatorLoc + 1));

							if(middleSection == mainFileName) 
							{
								const auto number = std::stoi(numberString);
								if(number > maxFileNumber)
								{
									maxFileNumber = number;
								}
							}
						}
					}
					std::string fileNameWoExt = screenShotFilesPrefix + seperator + mainFileName + seperator + std::to_string(maxFileNumber + 1);
					renderer->takeAndSaveScreenShot(std::filesystem::path(fileNameWoExt), true, sensors[activeSensor].denoiserInfo);
				}
				if(receiver.isLogProgressKeyPressed())
				{
					printf("[INFO] Rendering in progress - %d Total SamplesPerPixel Computed. \n", renderer->getTotalSamplesPerPixelComputed());
				}
				receiver.resetKeys();
			}

			driver->beginScene(false, false);
			if(!renderer->render(device->getTimer(),true,receiver.isRenderingBeauty()))
			{
				renderFailed = true;
				driver->endScene();
				break;
			}

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
				const double microsecondsElapsed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count();
				str << L"Raytraced Shadows Demo - Nabla Engine   MegaSamples: " << samples/1000000ull
					<< "   MSample/s: " << double(samples)/microsecondsElapsed
					<< "   MRay/s: " << double(rays)/microsecondsElapsed;

				device->setWindowCaption(str.str());
				lastFPSTime = time;
			}
		}
		
		if(renderFailed)
		{
			std::cout << "[ERROR] Render Failed." << std::endl;
		}
		else
		{
			auto extensionStr = getFileExtensionFromFormat(sensors[activeSensor].fileFormat);
			renderer->takeAndSaveScreenShot(std::filesystem::path("LastView_" + mainFileName + "_Sensor_" + std::to_string(activeSensor) + extensionStr), true, sensors[activeSensor].denoiserInfo);
		}

		renderer->deinitScreenSizedResources();
	}

	renderer->deinitSceneResources();
	renderer = nullptr;

	// will leak thread because there's no cross platform input!
	std::exit(0);
	return 0;
}