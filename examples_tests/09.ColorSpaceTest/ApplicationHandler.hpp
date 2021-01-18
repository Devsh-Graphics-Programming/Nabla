// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _APPLICATION_HANDLER_
#define _APPLICATION_HANDLER_

#include <nabla.h>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <chrono>

#define SWITCH_IMAGES_PER_X_MILISECONDS 500
constexpr std::string_view testingImagePathsFile = "../imagesTestList.txt";

class ApplicationHandler
{
	public:

		ApplicationHandler();

		void executeColorSpaceTest();
		bool getStatus() { return status; }

	private:

		using gpuPipeline = nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline>;

		bool initializeApplication();
		void fetchTestingImagePaths();
		void performImageTest(std::string path);
		void presentImageOnTheScreen(nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> gpuImageView, std::string currentHandledImageFileName, std::string currentHandledImageExtension);

		bool status;

		nbl::core::smart_refctd_ptr<nbl::IrrlichtDevice> device;
		nbl::video::IVideoDriver* driver;

		nbl::core::smart_refctd_ptr<nbl::video::IGPUMeshBuffer> currentGpuMeshBuffer;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSetLayout> gpuDescriptorSetLayout3;

		gpuPipeline currentGpuPipelineFor2D;
		gpuPipeline currentGpuPipelineFor2DArrays;
		gpuPipeline currentGpuPipelineForCubemaps;

		nbl::video::IFrameBuffer* screenShotFrameBuffer;
		nbl::core::vector<std::string> imagePaths;
};

#endif // _APPLICATION_HANDLER_