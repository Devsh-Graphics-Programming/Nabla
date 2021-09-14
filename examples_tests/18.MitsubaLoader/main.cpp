// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "nbl/ext/ScreenShot/ScreenShot.h"

#include "../common/Camera.hpp"
#include "../common/CommonAPI.h"

#include "../3rdparty/portable-file-dialogs/portable-file-dialogs.h"
#include "nbl/ext/MitsubaLoader/CMitsubaLoader.h"

#define USE_ENVMAP

using namespace nbl;
using namespace core;

#define MITSUBA_LOADER_TESTS

int main(int argc, char** argv)
{
	system::path CWD = system::path(argv[0]).parent_path().generic_string() + "/";
	constexpr uint32_t WIN_W = 1280;
	constexpr uint32_t WIN_H = 720;
	constexpr uint32_t SC_IMG_COUNT = 3u;
	constexpr uint32_t FRAMES_IN_FLIGHT = 5u;
	static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

	auto initOutput = CommonAPI::Init<WIN_W, WIN_H, SC_IMG_COUNT>(video::EAT_OPENGL, "MitsubaLoader", nbl::asset::EF_D32_SFLOAT);
	auto window = std::move(initOutput.window);
	auto gl = std::move(initOutput.apiConnection);
	auto surface = std::move(initOutput.surface);
	auto gpuPhysicalDevice = std::move(initOutput.physicalDevice);
	auto logicalDevice = std::move(initOutput.logicalDevice);
	auto queues = std::move(initOutput.queues);
	auto swapchain = std::move(initOutput.swapchain);
	auto renderpass = std::move(initOutput.renderpass);
	auto fbos = std::move(initOutput.fbo);
	auto commandPool = std::move(initOutput.commandPool);
	auto assetManager = std::move(initOutput.assetManager);
	auto logger = std::move(initOutput.logger);
	auto inputSystem = std::move(initOutput.inputSystem);
	auto system = std::move(initOutput.system);
	auto windowCallback = std::move(initOutput.windowCb);
	auto cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
	auto utilities = std::move(initOutput.utilities);

	//
	asset::SAssetBundle meshes;
	core::smart_refctd_ptr<const ext::MitsubaLoader::CMitsubaMetadata> globalMeta;
	{
		asset::CQuantNormalCache* qnc = assetManager->getMeshManipulator()->getQuantNormalCache();

		auto serializedLoader = core::make_smart_refctd_ptr<nbl::ext::MitsubaLoader::CSerializedLoader>(assetManager.get());
		auto mitsubaLoader = core::make_smart_refctd_ptr<nbl::ext::MitsubaLoader::CMitsubaLoader>(assetManager.get(), system.get());
		serializedLoader->initialize();
		mitsubaLoader->initialize();
		assetManager->addAssetLoader(std::move(serializedLoader));
		assetManager->addAssetLoader(std::move(mitsubaLoader));

		std::string filePath = "../../media/mitsuba/staircase2.zip";
		//#define MITSUBA_LOADER_TESTS
#ifndef MITSUBA_LOADER_TESTS
		pfd::message("Choose file to load", "Choose mitsuba XML file to load or ZIP containing an XML. \nIf you cancel or choosen file fails to load staircase will be loaded.", pfd::choice::ok);
		pfd::open_file file("Choose XML or ZIP file", (CWD/"../../media/mitsuba").string(), { "ZIP files (.zip)", "*.zip", "XML files (.xml)", "*.xml" });
		if (!file.result().empty())
			filePath = file.result()[0];
#endif
		if (core::hasFileExtension(filePath, "zip", "ZIP"))
		{
			const system::path archPath = CWD/filePath;
			core::smart_refctd_ptr<system::IFileArchive> arch = nullptr;
			arch = system->openFileArchive(archPath);

			if (!arch)
				arch = system->openFileArchive(CWD/ "../../media/mitsuba/staircase2.zip");
			if (!arch)
				return 2;

			system->mount(std::move(arch), "resources");

			auto flist = arch->getArchivedFiles();
			if (flist.empty())
				return 3;

			for (auto it = flist.begin(); it != flist.end(); )
			{
				if (core::hasFileExtension(it->fullName, "xml", "XML"))
					it++;
				else
					it = flist.erase(it);
			}
			if (flist.size() == 0u)
				return 4;

			std::cout << "Choose File (0-" << flist.size() - 1ull << "):" << std::endl;
			for (auto i = 0u; i < flist.size(); i++)
				std::cout << i << ": " << flist[i].fullName << std::endl;
			uint32_t chosen = 0;
#ifndef MITSUBA_LOADER_TESTS
			std::cin >> chosen;
#endif
			if (chosen >= flist.size())
				chosen = 0u;

			filePath = flist[chosen].fullName.string();
		}

		//! read cache results -- speeds up mesh generation
		qnc->loadCacheFromFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), "../../tmp/normalCache101010.sse");
		//! load the mitsuba scene
		asset::IAssetLoader::SAssetLoadParams loadParams;
		loadParams.workingDirectory = "resources";
		loadParams.logger = logger.get();
		meshes = assetManager->getAsset(filePath, loadParams);
		//! cache results -- speeds up mesh generation on second run
		qnc->saveCacheToFile<asset::EF_A2B10G10R10_SNORM_PACK32>(system.get(), "../../tmp/normalCache101010.sse");

		auto contents = meshes.getContents();
		if (contents.begin() >= contents.end())
			return 2;

		auto firstmesh = *contents.begin();
		if (!firstmesh)
			return 3;

		globalMeta = core::smart_refctd_ptr<const ext::MitsubaLoader::CMitsubaMetadata>(meshes.getMetadata()->selfCast<const ext::MitsubaLoader::CMitsubaMetadata>());
		if (!globalMeta)
			return 4;
	}


}