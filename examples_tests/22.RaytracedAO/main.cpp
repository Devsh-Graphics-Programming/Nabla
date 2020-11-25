#define _IRR_STATIC_LIB_
#include <irrlicht.h>

#include <chrono>

#include "../common/QToQuitEventReceiver.h"

#include "../3rdparty/portable-file-dialogs/portable-file-dialogs.h"
#include "irr/ext/MitsubaLoader/CMitsubaLoader.h"

#include "./dirty_source/ExtraCrap.h"


using namespace irr;
using namespace core;

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
	params.WindowSize = dimension2d<uint32_t>(1920, 1080);
	auto device = createDeviceEx(params);
	if (!device)
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

		asset::CQuantNormalCache* qnc = am->getMeshManipulator()->getQuantNormalCache();

		//! read cache results -- speeds up mesh generation
		//qnc->loadNormalQuantCacheFromFile<asset::E_QUANT_NORM_CACHE_TYPE::Q_2_10_10_10>(fs, "../../tmp/normalCache101010.sse", true);
		//! load the mitsuba scene
		meshes = am->getAsset(filePath, {});
		//! cache results -- speeds up mesh generation on second run
		qnc->saveCacheToFile(asset::CQuantNormalCache::E_CACHE_TYPE::ECT_2_10_10_10, fs, "../../tmp/normalCache101010.sse");
		
		auto contents = meshes.getContents();
		if (!contents.size())
			return 2;

		auto firstmesh = *contents.begin();
		if (!firstmesh)
			return 3;

		auto meta = firstmesh->getMetadata();
		if (!meta)
			return 4;
		assert(core::strcmpi(meta->getLoaderName(),ext::MitsubaLoader::IMitsubaMetadata::LoaderName) == 0);
		globalMeta = static_cast<ext::MitsubaLoader::IMeshMetadata*>(meta)->globalMetadata;
	}


	auto smgr = device->getSceneManager();

	bool rightHandedCamera = true;
	auto camera = smgr->addCameraSceneNode(nullptr);
	auto isOkSensorType = [](const ext::MitsubaLoader::CElementSensor& sensor) -> bool {
		return sensor.type == ext::MitsubaLoader::CElementSensor::Type::PERSPECTIVE || sensor.type == ext::MitsubaLoader::CElementSensor::Type::THINLENS;
	};
	if (globalMeta->sensors.size() && isOkSensorType(globalMeta->sensors.front()))
	{
		const auto& sensor = globalMeta->sensors.front();
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

	auto& _1stmesh = meshes.getContents().begin()[0];
	auto* meta_ = static_cast<asset::ICPUMesh*>(_1stmesh.get())->getMeshBuffer(0)->getPipeline()->getMetadata();
	auto* meta = static_cast<ext::MitsubaLoader::CMitsubaPipelineMetadata*>(meta_);
	auto cpuds0 = meta->getDescriptorSet();
	auto gpuds0 = driver->getGPUObjectsFromAssets(&cpuds0, &cpuds0 + 1)->front();

	core::smart_refctd_ptr<Renderer> renderer = core::make_smart_refctd_ptr<Renderer>(driver, device->getAssetManager(), smgr, std::move(gpuds0));
	constexpr uint32_t MaxSamples = 1024u*1024u;
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
			core::OwenSampler sampler(Renderer::MaxDimensions,0xdeadbeefu);

			uint32_t (&out)[][2] = *reinterpret_cast<uint32_t(*)[][2]>(sampleSequence->getPointer());
			for (auto dim=0u; dim<Renderer::MaxDimensions; dim++)
			for (uint32_t i=0; i<MaxSamples; i++)
			{
				out[(dim>>1u)*MaxSamples+i][dim&0x1u] = sampler.sample(dim,i);
			}

			io::IWriteFile* cacheFile = device->getFileSystem()->createAndWriteFile("../../tmp/rtSamples.bin");
			if (cacheFile)
			{
				cacheFile->write(sampleSequence->getPointer(),sampleSequence->getSize());
				cacheFile->drop();
			}
		}
	}
	renderer->init(meshes, rightHandedCamera, std::move(sampleSequence));


	meshes = {}; // free memory
	auto extent = renderer->getSceneBound().getExtent();

	// want dynamic camera or not?
	if (true)
	{
		core::vector3df_SIMD ptu[] = {core::vectorSIMDf().set(camera->getPosition()),camera->getTarget(),camera->getUpVector()};
		auto proj = camera->getProjectionMatrix();

		camera = smgr->addCameraSceneNodeFPS(nullptr, 100.f, core::min(extent.X, extent.Y, extent.Z) * 0.0002f);
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

		renderer->render();

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
			str << L"Raytraced Shadows Demo - IrrlichtBAW Engine   MegaSamples: " << samples/1000000ull << "   MRay/s: "
				<< double(samples)/double(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count());

			device->setWindowCaption(str.str());
			lastFPSTime = time;
		}
	}
	renderer->deinit();
	renderer = nullptr;

	return 0;
}
