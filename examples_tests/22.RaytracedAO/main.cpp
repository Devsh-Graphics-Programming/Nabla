// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include <chrono>

#include "../common/QToQuitEventReceiver.h"

#include "../3rdparty/portable-file-dialogs/portable-file-dialogs.h"
#include "nbl/ext/MitsubaLoader/CMitsubaLoader.h"

#include "Renderer.h"


using namespace nbl;
using namespace core;

int main()
{
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

		//std::string filePath = "../../media/mitsuba/daily_pt.xml";
		std::string filePath = "../../media/mitsuba/staircase2.zip";
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
	auto camera = smgr->addCameraSceneNode(nullptr);
	auto isOkSensorType = [](const ext::MitsubaLoader::CElementSensor& sensor) -> bool {
		return sensor.type == ext::MitsubaLoader::CElementSensor::Type::PERSPECTIVE || sensor.type == ext::MitsubaLoader::CElementSensor::Type::THINLENS;
	};
	if (globalMeta->m_global.m_sensors.size() && isOkSensorType(globalMeta->m_global.m_sensors.front()))
	{
		const auto& sensor = globalMeta->m_global.m_sensors.front();
		const auto& film = sensor.film;

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
	}
	else
	{
		camera->setNearValue(20.f);
		camera->setFarValue(5000.f);
	}


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
	// want dynamic camera or not?
	if (true)
	{
		core::vector3df_SIMD ptu[] = {core::vectorSIMDf().set(camera->getPosition()),camera->getTarget(),camera->getUpVector()};
		auto proj = camera->getProjectionMatrix();

		camera = smgr->addCameraSceneNodeFPS(nullptr, 80.f, core::min(extent.X, extent.Y, extent.Z) * 0.0001f);
		camera->setPosition(ptu[0].getAsVector3df());
		camera->setTarget(ptu[1].getAsVector3df());
		camera->setUpVector(ptu[2]);
		camera->setProjectionMatrix(proj);

		device->getCursorControl()->setVisible(false);
	}

	smgr->setActiveCamera(camera);


	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);

	uint64_t lastFPSTime = 0;
	auto start = std::chrono::steady_clock::now();
	while (device->run() && receiver.keepOpen())
	{
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
			str << L"Raytraced Shadows Demo - Nabla Engine   MegaSamples: " << samples/1000000ull << "   MRay/s: "
				<< double(samples)/double(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count());

			device->setWindowCaption(str.str());
			lastFPSTime = time;
		}
	}
	renderer->deinit();
	renderer = nullptr;

	return 0;
}