#ifndef _APPLICATION_HANDLER_
#define _APPLICATION_HANDLER_

#include <irrlicht.h>
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

		bool initializeApplication();
		void fetchTestingImagePaths();
		void performImageTest(std::string path);
		void presentImageOnTheScreen(irr::core::smart_refctd_ptr<irr::video::IGPUImageView> gpuImageView, std::string currentHandledImageFileName, std::string currentHandledImageExtension);

		bool status;

		irr::core::smart_refctd_ptr<irr::IrrlichtDevice> device;
		irr::video::IVideoDriver* driver;

		irr::core::smart_refctd_ptr<irr::video::IGPUDescriptorSetLayout> gpuDescriptorSetLayout3;
		irr::core::smart_refctd_ptr<irr::video::IGPURenderpassIndependentPipeline> currentGpuPipeline;
		irr::core::smart_refctd_ptr<irr::video::IGPUMeshBuffer> currentGpuMeshBuffer;

		irr::video::IFrameBuffer* screenShotFrameBuffer;
		irr::core::vector<std::string> imagePaths;
};

#endif // _APPLICATION_HANDLER_