#include "ApplicationHandler.hpp"

#include "../../ext/FullScreenTriangle/FullScreenTriangle.h"
#include "../../ext/ScreenShot/ScreenShot.h"

using namespace irr;
using namespace core;
using namespace asset;
using namespace video;

ApplicationHandler::ApplicationHandler()
{
	status = initializeApplication();
	fetchTestingImagePaths();
}

void ApplicationHandler::executeColorSpaceTest()
{
	for (const auto& pathToAnImage : imagePaths)
		performImageTest(pathToAnImage);
}

void ApplicationHandler::fetchTestingImagePaths()
{
	std::ifstream list(testingImagePathsFile.data());
	if (list.is_open())
	{
		std::string line;
		for (; std::getline(list, line); )
		{
			if (line != "" && line[0] != ';')
				imagePaths.push_back(line);
		}
	}
}

void ApplicationHandler::presentImageOnTheScreen(irr::core::smart_refctd_ptr<irr::video::IGPUImageView> gpuImageView, std::string currentHandledImageFileName, std::string currentHandledImageExtension)
{
	auto samplerDescriptorSet3 = driver->createGPUDescriptorSet(core::smart_refctd_ptr(gpuDescriptorSetLayout3));

	IGPUDescriptorSet::SDescriptorInfo info;
	{
		info.desc = std::move(gpuImageView);
		ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_LINEAR, ISampler::ETF_LINEAR, ISampler::ESMM_LINEAR, 0u, false, ECO_ALWAYS };
		info.image.sampler = driver->createGPUSampler(samplerParams);
		info.image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;
	}

	IGPUDescriptorSet::SWriteDescriptorSet write;
	write.dstSet = samplerDescriptorSet3.get();
	write.binding = 0u;
	write.arrayElement = 0u;
	write.count = 1u;
	write.descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
	write.info = &info;

	driver->updateDescriptorSets(1u, &write, 0u, nullptr);

	std::wostringstream characterStream;
	characterStream << L"Color Space Test Demo - Irrlicht Engine [" << driver->getName() << "] - CURRENT IMAGE: " << currentHandledImageFileName.c_str() << " - EXTENSION: " << currentHandledImageExtension.c_str();
	device->setWindowCaption(characterStream.str().c_str());

	auto startPoint = std::chrono::high_resolution_clock::now();

	while (device->run())
	{
		auto aPoint = std::chrono::high_resolution_clock::now();
		if (std::chrono::duration_cast<std::chrono::milliseconds>(aPoint - startPoint).count() > SWITCH_IMAGES_PER_X_MILISECONDS)
			break;

		driver->beginScene(true, true);

		driver->bindGraphicsPipeline(currentGpuPipeline.get());
		driver->bindDescriptorSets(EPBP_GRAPHICS, currentGpuPipeline->getLayout(), 3u, 1u, &samplerDescriptorSet3.get(), nullptr);
		driver->drawMeshBuffer(currentGpuMeshBuffer.get());

		driver->blitRenderTargets(nullptr, screenShotFrameBuffer, false, false);
		driver->endScene();
	}

	ext::ScreenShot::createScreenShoot(device, screenShotFrameBuffer->getAttachment(video::EFAP_COLOR_ATTACHMENT0), "screenShot_" + currentHandledImageFileName + ".png");
}

void ApplicationHandler::performImageTest(std::string path)
{
	os::Printer::log("Reading", path);

	auto assetManager = device->getAssetManager();

	smart_refctd_ptr<ICPUImageView> cpuImageView;

	auto cpuTexture = assetManager->getAsset(path, {});
	auto cpuTextureContents = cpuTexture.getContents();
	
	if (cpuTextureContents.first == cpuTextureContents.second)
	{
		os::Printer::log("CANNOT PERFORM THE TEST - SKIPPING. LOADING WENT WRONG", ELL_WARNING);
		return;
	}

	io::path filename, extension, finalFileNameWithExtension;
	core::splitFilename(path.c_str(), nullptr, &filename, &extension);
	finalFileNameWithExtension = filename + ".";
	finalFileNameWithExtension += extension;

	// Can't do that because write is still broken
	bool writeable = (extension != "dds");
	smart_refctd_ptr<ICPUImageView> copyImageView;

	auto asset = *cpuTextureContents.first;
	core::smart_refctd_ptr<video::IGPUImageView> gpuImageView;
	switch (asset->getAssetType())
	{
		case IAsset::ET_IMAGE:
		{
			ICPUImageView::SCreationParams viewParams;
			viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
			viewParams.image = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(asset);
			viewParams.format = viewParams.image->getCreationParameters().format;
			viewParams.viewType = IImageView<ICPUImage>::ET_2D;
			viewParams.subresourceRange.baseArrayLayer = 0u;
			viewParams.subresourceRange.layerCount = 1u;
			viewParams.subresourceRange.baseMipLevel = 0u;
			viewParams.subresourceRange.levelCount = 1u;

			cpuImageView = ICPUImageView::create(std::move(viewParams));
		} break;

		case IAsset::ET_IMAGE_VIEW:
		{
			cpuImageView = core::smart_refctd_ptr_static_cast<asset::ICPUImageView>(asset);
		} break;

		default:
		{
			os::Printer::log("EXPECTED IMAGE ASSET TYPE!", ELL_ERROR);
			break;
		}
	}

	copyImageView = core::smart_refctd_ptr_static_cast<ICPUImageView>(cpuImageView->clone());
	gpuImageView = driver->getGPUObjectsFromAssets(&cpuImageView.get(), &cpuImageView.get() + 1u)->front();

	if (gpuImageView)
	{
		auto gpuViewParams = gpuImageView->getCreationParameters();
		gpuViewParams.image = driver->createDeviceLocalGPUImageOnDedMem(video::IGPUImage::SCreationParams(gpuViewParams.image->getCreationParameters()));
		gpuViewParams.viewType = IGPUImageView::ET_2D;

		presentImageOnTheScreen(gpuImageView, std::string(filename.c_str()), std::string(extension.c_str()));
	}

	if (writeable)
	{
		auto tryToWrite = [&](asset::IAsset* asset)
		{
			asset::IAssetWriter::SAssetWriteParams wparams(asset);
			return assetManager->writeAsset((io::path("imageAsset_") + finalFileNameWithExtension).c_str(), wparams);
		};

		if(!tryToWrite(copyImageView->getCreationParameters().image.get()))
			if(!tryToWrite(copyImageView.get()))
				os::Printer::log("An unexcepted error occoured while trying to write the asset!", irr::ELL_WARNING);
	}

	assetManager->removeCachedGPUObject(asset.get(), gpuImageView);
	assetManager->removeAssetFromCache(cpuTexture);
}


bool ApplicationHandler::initializeApplication()
{
	irr::SIrrlichtCreationParameters params;
	params.Bits = 24;
	params.ZBufferBits = 24; 
	params.DriverType = video::EDT_OPENGL; 
	params.WindowSize = dimension2d<uint32_t>(1600, 900);
	params.Fullscreen = false;

	device = createDeviceEx(params);
	if (!device)
		return false;

	driver = device->getVideoDriver();
	screenShotFrameBuffer = ext::ScreenShot::createDefaultFBOForScreenshoting(device);
	auto fullScreenTriangle = ext::FullScreenTriangle::createFullScreenTriangle(device->getAssetManager(), device->getVideoDriver());

	IGPUDescriptorSetLayout::SBinding binding{ 0u, EDT_COMBINED_IMAGE_SAMPLER, 1u, IGPUSpecializedShader::ESS_FRAGMENT, nullptr };
	gpuDescriptorSetLayout3 = driver->createGPUDescriptorSetLayout(&binding, &binding + 1u);
	{
		auto gpuPipelineLayout = driver->createGPUPipelineLayout(nullptr, nullptr, nullptr, nullptr, nullptr, core::smart_refctd_ptr(gpuDescriptorSetLayout3));

		IAssetLoader::SAssetLoadParams lp;
		auto fs_bundle = device->getAssetManager()->getAsset("../present.frag", lp);
		auto fs_contents = fs_bundle.getContents();
		if (fs_contents.first == fs_contents.second)
			return false;
		ICPUSpecializedShader* fs = static_cast<ICPUSpecializedShader*>(fs_contents.first->get());

		auto fragShader = driver->getGPUObjectsFromAssets(&fs, &fs + 1)->front();
		if (!fragShader)
			return false;

		IGPUSpecializedShader* shaders[2] = { std::get<0>(fullScreenTriangle).get(),fragShader.get() };
		SBlendParams blendParams;
		blendParams.logicOpEnable = false;
		blendParams.logicOp = ELO_NO_OP;
		for (size_t i = 0ull; i < SBlendParams::MAX_COLOR_ATTACHMENT_COUNT; i++)
			blendParams.blendParams[i].attachmentEnabled = (i == 0ull);
		SRasterizationParams rasterParams;
		rasterParams.faceCullingMode = EFCM_NONE;
		rasterParams.depthCompareOp = ECO_ALWAYS;
		rasterParams.minSampleShading = 1.f;
		rasterParams.depthWriteEnable = false;
		rasterParams.depthTestEnable = false;

		currentGpuPipeline = driver->createGPURenderpassIndependentPipeline(nullptr, std::move(gpuPipelineLayout), shaders, shaders + sizeof(shaders) / sizeof(IGPUSpecializedShader*),
			std::get<SVertexInputParams>(fullScreenTriangle), blendParams,
			std::get<SPrimitiveAssemblyParams>(fullScreenTriangle), rasterParams);
	}

	{
		SBufferBinding<IGPUBuffer> idxBinding{ 0ull, nullptr };
		currentGpuMeshBuffer = core::make_smart_refctd_ptr<IGPUMeshBuffer>(nullptr, nullptr, nullptr, std::move(idxBinding));
		currentGpuMeshBuffer->setIndexCount(3u);
		currentGpuMeshBuffer->setInstanceCount(1u);
	}

	return true;
}