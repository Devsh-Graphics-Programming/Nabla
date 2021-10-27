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
	std::string outputScreenshotsFolderPath = cmdHandler.getOutputScreenshotsFolderPath();
	bool shouldTerminate = cmdHandler.getTerminate();

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
		if (!contents.size())
			return 2;

		globalMeta = core::smart_refctd_ptr<const ext::MitsubaLoader::CMitsubaMetadata>(meshes.getMetadata()->selfCast<const ext::MitsubaLoader::CMitsubaMetadata>());
		if (!globalMeta)
			return 3;
	}

	auto smgr = device->getSceneManager();

	// TODO: Move into renderer?
	bool rightHandedCamera = true;
	float moveSpeed = core::nan<float>();
	uint32_t sensorSamplesNeeded = 0u;

	auto camera = smgr->addCameraSceneNodeModifiedMaya(nullptr, -400.0f, 20.0f, 200.0f, -1, 2.0f, 1.0f, false, true);

	auto isOkSensorType = [](const ext::MitsubaLoader::CElementSensor& sensor) -> bool {
		return sensor.type == ext::MitsubaLoader::CElementSensor::Type::PERSPECTIVE || sensor.type == ext::MitsubaLoader::CElementSensor::Type::THINLENS;
	};
	if (globalMeta->m_global.m_sensors.size() && isOkSensorType(globalMeta->m_global.m_sensors.front()))
	{
		const auto& sensor = globalMeta->m_global.m_sensors.front();
		const auto& film = sensor.film;

		sensorSamplesNeeded = sensor.sampler.sampleCount;

		// need to extract individual components
		{
			auto relativeTransform = sensor.transform.matrix.extractSub3x4();
			if (relativeTransform.getPseudoDeterminant().x < 0.f)
				rightHandedCamera = false;

			auto pos = relativeTransform.getTranslation();
			camera->setPosition(pos.getAsVector3df());

			auto tpose = core::transpose(sensor.transform.matrix);
			auto up = tpose.rows[1];
			core::vectorSIMDf view = tpose.rows[2];
			auto target = view+pos;
			
			camera->setTarget(target.getAsVector3df());
			if (core::dot(core::normalize(core::cross(camera->getUpVector(),view)),core::cross(up,view)).x<0.99f)
				camera->setUpVector(up);
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
		auto width = film.cropWidth;
		auto height = film.cropHeight;
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
		float nearClip = core::max(persp->nearClip, persp->farClip * 0.0001);
		if (rightHandedCamera)
			camera->setProjectionMatrix(core::matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(realFoVDegrees), aspectRatio, nearClip, persp->farClip));
		else
			camera->setProjectionMatrix(core::matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(realFoVDegrees), aspectRatio, nearClip, persp->farClip));
		moveSpeed = persp->moveSpeed;
	}
	else
	{
		camera->setNearValue(20.f);
		camera->setFarValue(5000.f);
	}
	
	auto modifiedMayaAnim = reinterpret_cast<scene::CSceneNodeAnimatorCameraModifiedMaya*>(camera->getAnimators()[0]);
	core::vectorSIMDf cameraPos; cameraPos.set(camera->getPosition());
	core::vectorSIMDf cameraTarget; cameraTarget.set(camera->getTarget());
	modifiedMayaAnim->setZoomAndRotationBasedOnTargetAndPosition(cameraPos, cameraTarget);

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

	renderer->init(meshes, std::move(sampleSequence));
	meshes = {}; // free memory

	auto extent = renderer->getSceneBound().getExtent();
	smgr->setActiveCamera(camera);

	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);

	uint64_t lastFPSTime = 0;
	auto start = std::chrono::steady_clock::now();
	while (device->run() && receiver.keepOpen())
	{
		driver->beginScene(false, false);

		std::cout << "Camera Position Before Render: (" << camera->getPosition().X << "," << camera->getPosition().Y << "," << camera->getPosition().Z << ")" << std::endl;
		std::cout << "Target Before Render: (" << camera->getTarget().X << "," << camera->getTarget().Y << "," << camera->getTarget().Z << ")" << std::endl;
		renderer->render(device->getTimer());
		std::cout << "Camera Position After Render: (" << camera->getPosition().X << "," << camera->getPosition().Y << "," << camera->getPosition().Z << ")" << std::endl;
		std::cout << "Target After Render: (" << camera->getTarget().X << "," << camera->getTarget().Y << "," << camera->getTarget().Z << ")" << std::endl;

		auto oldVP = driver->getViewPort();
		driver->blitRenderTargets(renderer->getColorBuffer(),nullptr,false,false,{},{},true);
		driver->setViewPort(oldVP);

		driver->endScene();

		if(shouldTerminate && renderer->getTotalSamplesPerPixelComputed() >= sensorSamplesNeeded)
			break;

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
	renderer->takeAndSaveScreenShot("tonemapped", outputScreenshotsFolderPath);
	renderer->deinit();
	renderer = nullptr;

	return 0;
}