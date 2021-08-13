// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include <nabla.h>

#include "../common/QToQuitEventReceiver.h"

#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"
#include "nbl/ext/ScreenShot/ScreenShot.h"

using namespace nbl;
using namespace core;
using namespace asset;
using namespace video;

smart_refctd_ptr<IGPUImageView> createHDRImageView(core::smart_refctd_ptr<IrrlichtDevice> device, asset::E_FORMAT colorFormat)
{
	auto driver = device->getVideoDriver();

	smart_refctd_ptr<IGPUImageView> gpuImageViewColorBuffer;
	{
		IGPUImage::SCreationParams imgInfo;
		imgInfo.format = colorFormat;
		imgInfo.type = IGPUImage::ET_2D;
		imgInfo.extent.width = driver->getScreenSize().Width;
		imgInfo.extent.height = driver->getScreenSize().Height;
		imgInfo.extent.depth = 1u;
		imgInfo.mipLevels = 1u;
		imgInfo.arrayLayers = 1u;
		imgInfo.samples = asset::ICPUImage::ESCF_1_BIT;
		imgInfo.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);

		auto image = driver->createGPUImageOnDedMem(std::move(imgInfo),driver->getDeviceLocalGPUMemoryReqs());

		IGPUImageView::SCreationParams imgViewInfo;
		imgViewInfo.image = std::move(image);
		imgViewInfo.format = colorFormat;
		imgViewInfo.viewType = IGPUImageView::ET_2D;
		imgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
		imgViewInfo.subresourceRange.baseArrayLayer = 0u;
		imgViewInfo.subresourceRange.baseMipLevel = 0u;
		imgViewInfo.subresourceRange.layerCount = 1u;
		imgViewInfo.subresourceRange.levelCount = 1u;

		gpuImageViewColorBuffer = driver->createGPUImageView(std::move(imgViewInfo));
	}

	return gpuImageViewColorBuffer;
}

struct ShaderParameters
{
	const uint32_t MaxDepthLog2 = 4; //5
	const uint32_t MaxSamplesLog2 = 10; //18
} kShaderParameters;

enum E_LIGHT_GEOMETRY
{
	ELG_SPHERE,
	ELG_TRIANGLE,
	ELG_RECTANGLE
};

struct DispatchInfo_t
{
	uint32_t workGroupCount[3];
};

_NBL_STATIC_INLINE_CONSTEXPR uint32_t DEFAULT_WORK_GROUP_SIZE = 16u;

DispatchInfo_t getDispatchInfo(uint32_t imgWidth, uint32_t imgHeight) {
	DispatchInfo_t ret = {};
	ret.workGroupCount[0] = (uint32_t)core::ceil<float>((float)imgWidth / (float)DEFAULT_WORK_GROUP_SIZE);
	ret.workGroupCount[1] = (uint32_t)core::ceil<float>((float)imgHeight / (float)DEFAULT_WORK_GROUP_SIZE);
	ret.workGroupCount[2] = 1;
	return ret;
}

int main()
{
	nbl::SIrrlichtCreationParameters params;
	params.Bits = 32;
	params.ZBufferBits = 24;
	params.DriverType = video::EDT_OPENGL;
	params.WindowSize = dimension2d<uint32_t>(1920, 1080);
	params.Fullscreen = false;
	params.Doublebuffer = true;
	params.Vsync = true;
	params.Stencilbuffer = false;

	auto device = createDeviceEx(params);
	if (!device)
		return false;

	device->getCursorControl()->setVisible(false);
	auto driver = device->getVideoDriver();

	auto assetManager = device->getAssetManager();
	auto sceneManager = device->getSceneManager();
	auto geometryCreator = device->getAssetManager()->getGeometryCreator();

	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);

	scene::ICameraSceneNode* camera = sceneManager->addCameraSceneNodeFPS(0, 100.0f, 0.001f);
	camera->setLeftHanded(false);

	camera->setPosition(core::vector3df(0, 2, 3));
	camera->setTarget(core::vector3df(0, 0, 0));
	camera->setNearValue(0.03125f);
	camera->setFarValue(200.0f);
	camera->setFOV(core::radians(60.f));

	sceneManager->setActiveCamera(camera);

	IGPUDescriptorSetLayout::SBinding descriptorSet0Bindings[] = {
		{ 0u, EDT_STORAGE_IMAGE, 1u, IGPUSpecializedShader::ESS_COMPUTE, nullptr },
	};
	IGPUDescriptorSetLayout::SBinding uboBinding {0, asset::EDT_UNIFORM_BUFFER, 1u, IGPUSpecializedShader::ESS_FRAGMENT, nullptr};
	IGPUDescriptorSetLayout::SBinding descriptorSet3Bindings[] = {
		{ 0u, EDT_COMBINED_IMAGE_SAMPLER, 1u, IGPUSpecializedShader::ESS_COMPUTE, nullptr },
		{ 1u, EDT_UNIFORM_TEXEL_BUFFER, 1u, IGPUSpecializedShader::ESS_COMPUTE, nullptr },
		{ 2u, EDT_COMBINED_IMAGE_SAMPLER, 1u, IGPUSpecializedShader::ESS_COMPUTE, nullptr }
	};
	
	auto gpuDescriptorSetLayout0 = driver->createGPUDescriptorSetLayout(descriptorSet0Bindings, descriptorSet0Bindings + 1u);
	auto gpuDescriptorSetLayout1 = driver->createGPUDescriptorSetLayout(&uboBinding, &uboBinding + 1u);
	auto gpuDescriptorSetLayout3 = driver->createGPUDescriptorSetLayout(descriptorSet3Bindings, descriptorSet3Bindings+3u);

	auto createGpuResources = [&](std::string pathToShader) -> core::smart_refctd_ptr<video::IGPUComputePipeline>
	{
		auto cpuComputeSpecializedShader = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(assetManager->getAsset(pathToShader, {}).getContents().begin()[0]);


		ISpecializedShader::SInfo info = cpuComputeSpecializedShader->getSpecializationInfo();
		info.m_backingBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(sizeof(ShaderParameters));
		memcpy(info.m_backingBuffer->getPointer(),&kShaderParameters,sizeof(ShaderParameters));
		info.m_entries = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ISpecializedShader::SInfo::SMapEntry>>(2u);
		for (uint32_t i=0; i<2; i++)
			info.m_entries->operator[](i) = {i,i*sizeof(uint32_t),sizeof(uint32_t)};


		cpuComputeSpecializedShader->setSpecializationInfo(std::move(info));

		auto gpuComputeSpecializedShader = driver->getGPUObjectsFromAssets(&cpuComputeSpecializedShader.get(), &cpuComputeSpecializedShader.get() + 1)->front();

		auto gpuPipelineLayout = driver->createGPUPipelineLayout(nullptr, nullptr, core::smart_refctd_ptr(gpuDescriptorSetLayout0), core::smart_refctd_ptr(gpuDescriptorSetLayout1), nullptr, core::smart_refctd_ptr(gpuDescriptorSetLayout3));

		auto gpuPipeline = driver->createGPUComputePipeline(nullptr, std::move(gpuPipelineLayout), std::move(gpuComputeSpecializedShader));

		return gpuPipeline;
	};

	E_LIGHT_GEOMETRY lightGeom = ELG_SPHERE;
	constexpr const char* shaderPaths[] = {"../litBySphere.comp","../litByTriangle.comp","../litByRectangle.comp"};
	auto gpuComputePipeline = createGpuResources(shaderPaths[lightGeom]);
	
	DispatchInfo_t dispatchInfo = getDispatchInfo(driver->getScreenSize().Width, driver->getScreenSize().Height);

	auto createGPUImageView = [&](std::string pathToOpenEXRHDRIImage)
	{
		auto pathToTexture = pathToOpenEXRHDRIImage;
		IAssetLoader::SAssetLoadParams lp(0ull, nullptr, IAssetLoader::ECF_DONT_CACHE_REFERENCES);
		auto cpuTexture = assetManager->getAsset(pathToTexture, lp);
		auto cpuTextureContents = cpuTexture.getContents();

		auto asset = *cpuTextureContents.begin();

		ICPUImageView::SCreationParams viewParams;
		viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
		viewParams.image = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(asset);
		viewParams.format = viewParams.image->getCreationParameters().format;
		viewParams.viewType = IImageView<ICPUImage>::ET_2D;
		viewParams.subresourceRange.baseArrayLayer = 0u;
		viewParams.subresourceRange.layerCount = 1u;
		viewParams.subresourceRange.baseMipLevel = 0u;
		viewParams.subresourceRange.levelCount = 1u;

		auto cpuImageView = ICPUImageView::create(std::move(viewParams));
		auto gpuImageView = driver->getGPUObjectsFromAssets(&cpuImageView.get(), &cpuImageView.get() + 1u)->front();

		return gpuImageView;
	};

	auto gpuEnvmapImageView = createGPUImageView("../../media/envmap/envmap_0.exr");

	smart_refctd_ptr<IGPUBufferView> gpuSequenceBufferView;
	{
		const uint32_t MaxDimensions = 3u<<kShaderParameters.MaxDepthLog2;
		const uint32_t MaxSamples = 1u<<kShaderParameters.MaxSamplesLog2;

		auto sampleSequence = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(uint32_t)*MaxDimensions*MaxSamples);
		
		core::OwenSampler sampler(MaxDimensions, 0xdeadbeefu);
		//core::SobolSampler sampler(MaxDimensions);

		auto out = reinterpret_cast<uint32_t*>(sampleSequence->getPointer());
		for (auto dim=0u; dim<MaxDimensions; dim++)
		for (uint32_t i=0; i<MaxSamples; i++)
		{
			out[i*MaxDimensions+dim] = sampler.sample(dim,i);
		}
		auto gpuSequenceBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(sampleSequence->getSize(), sampleSequence->getPointer());
		gpuSequenceBufferView = driver->createGPUBufferView(gpuSequenceBuffer.get(), asset::EF_R32G32B32_UINT);
	}

	smart_refctd_ptr<IGPUImageView> gpuScrambleImageView;
	{
		IGPUImage::SCreationParams imgParams;
		imgParams.flags = static_cast<IImage::E_CREATE_FLAGS>(0u);
		imgParams.type = IImage::ET_2D;
		imgParams.format = EF_R32G32_UINT;
		imgParams.extent = {params.WindowSize.Width,params.WindowSize.Height,1u};
		imgParams.mipLevels = 1u;
		imgParams.arrayLayers = 1u;
		imgParams.samples = IImage::ESCF_1_BIT;

		IGPUImage::SBufferCopy region;
		region.imageExtent = imgParams.extent;
		region.imageSubresource.layerCount = 1u;

		constexpr auto ScrambleStateChannels = 2u;
		const auto renderPixelCount = imgParams.extent.width*imgParams.extent.height;
		core::vector<uint32_t> random(renderPixelCount*ScrambleStateChannels);
		{
			core::RandomSampler rng(0xbadc0ffeu);
			for (auto& pixel : random)
				pixel = rng.nextSample();
		}
		auto buffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(random.size()*sizeof(uint32_t),random.data());

		IGPUImageView::SCreationParams viewParams;
		viewParams.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
		viewParams.image = driver->createFilledDeviceLocalGPUImageOnDedMem(std::move(imgParams),buffer.get(),1u,&region);
		viewParams.viewType = IGPUImageView::ET_2D;
		viewParams.format = EF_R32G32_UINT;
		viewParams.subresourceRange.levelCount = 1u;
		viewParams.subresourceRange.layerCount = 1u;
		gpuScrambleImageView = driver->createGPUImageView(std::move(viewParams));
	}
	
	// Create Out Image TODO
	auto outImgView = createHDRImageView(device, asset::EF_R16G16B16A16_SFLOAT);

	auto uboDescriptorSet0 = driver->createGPUDescriptorSet(core::smart_refctd_ptr(gpuDescriptorSetLayout0));
	{
		video::IGPUDescriptorSet::SWriteDescriptorSet writeDescriptorSet;
		writeDescriptorSet.dstSet = uboDescriptorSet0.get();
		writeDescriptorSet.binding = 0;
		writeDescriptorSet.count = 1u;
		writeDescriptorSet.arrayElement = 0u;
		writeDescriptorSet.descriptorType = asset::EDT_STORAGE_IMAGE;
		video::IGPUDescriptorSet::SDescriptorInfo info;
		{
			info.desc = outImgView;
			info.image.sampler = nullptr;
			info.image.imageLayout = static_cast<asset::E_IMAGE_LAYOUT>(0u);;
		}
		writeDescriptorSet.info = &info;
		driver->updateDescriptorSets(1u, &writeDescriptorSet, 0u, nullptr);
	}

	auto gpuubo = driver->createDeviceLocalGPUBufferOnDedMem(sizeof(SBasicViewParameters));
	auto uboDescriptorSet1 = driver->createGPUDescriptorSet(core::smart_refctd_ptr(gpuDescriptorSetLayout1));
	{
		video::IGPUDescriptorSet::SWriteDescriptorSet uboWriteDescriptorSet;
		uboWriteDescriptorSet.dstSet = uboDescriptorSet1.get();
		uboWriteDescriptorSet.binding = 0;
		uboWriteDescriptorSet.count = 1u;
		uboWriteDescriptorSet.arrayElement = 0u;
		uboWriteDescriptorSet.descriptorType = asset::EDT_UNIFORM_BUFFER;
		video::IGPUDescriptorSet::SDescriptorInfo info;
		{
			info.desc = gpuubo;
			info.buffer.offset = 0ull;
			info.buffer.size = sizeof(SBasicViewParameters);
		}
		uboWriteDescriptorSet.info = &info;
		driver->updateDescriptorSets(1u, &uboWriteDescriptorSet, 0u, nullptr);
	}

	auto descriptorSet3 = driver->createGPUDescriptorSet(core::smart_refctd_ptr(gpuDescriptorSetLayout3));
	{
		constexpr auto kDescriptorCount = 3;
		IGPUDescriptorSet::SWriteDescriptorSet samplerWriteDescriptorSet[kDescriptorCount];
		IGPUDescriptorSet::SDescriptorInfo samplerDescriptorInfo[kDescriptorCount];
		for (auto i=0; i<kDescriptorCount; i++)
		{
			samplerWriteDescriptorSet[i].dstSet = descriptorSet3.get();
			samplerWriteDescriptorSet[i].binding = i;
			samplerWriteDescriptorSet[i].arrayElement = 0u;
			samplerWriteDescriptorSet[i].count = 1u;
			samplerWriteDescriptorSet[i].info = samplerDescriptorInfo+i;
		}
		samplerWriteDescriptorSet[0].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
		samplerWriteDescriptorSet[1].descriptorType = EDT_UNIFORM_TEXEL_BUFFER;
		samplerWriteDescriptorSet[2].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;

		samplerDescriptorInfo[0].desc = gpuEnvmapImageView;
		{
			ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK, ISampler::ETF_LINEAR, ISampler::ETF_LINEAR, ISampler::ESMM_LINEAR, 0u, false, ECO_ALWAYS };
			samplerDescriptorInfo[0].image.sampler = driver->createGPUSampler(samplerParams);
			samplerDescriptorInfo[0].image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;
		}
		samplerDescriptorInfo[1].desc = gpuSequenceBufferView;
		samplerDescriptorInfo[2].desc = gpuScrambleImageView;
		{
			ISampler::SParams samplerParams = { ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_INT_OPAQUE_BLACK, ISampler::ETF_NEAREST, ISampler::ETF_NEAREST, ISampler::ESMM_NEAREST, 0u, false, ECO_ALWAYS };
			samplerDescriptorInfo[2].image.sampler = driver->createGPUSampler(samplerParams);
			samplerDescriptorInfo[2].image.imageLayout = EIL_SHADER_READ_ONLY_OPTIMAL;
		}
		driver->updateDescriptorSets(kDescriptorCount, samplerWriteDescriptorSet, 0u, nullptr);
	}

	auto blitFBO = driver->addFrameBuffer();
	blitFBO->attach(video::EFAP_COLOR_ATTACHMENT0, std::move(outImgView));

	float colorClearValues[] = { 1.f, 1.f, 1.f, 1.f };

	uint64_t lastFPSTime = 0;
	while (device->run() && receiver.keepOpen())
	if (device->isWindowFocused())
	{
		driver->beginScene(false, false);

		camera->OnAnimate(std::chrono::duration_cast<std::chrono::milliseconds>(device->getTimer()->getTime()).count());
		camera->render();

		const auto viewMatrix = camera->getViewMatrix();
		const auto viewProjectionMatrix = camera->getConcatenatedMatrix();

		// cube envmap handle
		{
			auto mv = viewMatrix;
			auto mvp = viewProjectionMatrix;
			core::matrix3x4SIMD normalMat;
			mv.getSub3x3InverseTranspose(normalMat);

			SBasicViewParameters uboData;
			memcpy(uboData.MV, mv.pointer(), sizeof(mv));
			memcpy(uboData.MVP, mvp.pointer(), sizeof(mvp));
			memcpy(uboData.NormalMat, normalMat.pointer(), sizeof(normalMat));
			driver->updateBufferRangeViaStagingBuffer(gpuubo.get(), 0ull, sizeof(uboData), &uboData);

			driver->bindComputePipeline(gpuComputePipeline.get());
			driver->bindDescriptorSets(EPBP_COMPUTE, gpuComputePipeline->getLayout(), 0u, 1u, &uboDescriptorSet0.get(), nullptr);
			driver->bindDescriptorSets(EPBP_COMPUTE, gpuComputePipeline->getLayout(), 1u, 1u, &uboDescriptorSet1.get(), nullptr);
			driver->bindDescriptorSets(EPBP_COMPUTE, gpuComputePipeline->getLayout(), 3u, 1u, &descriptorSet3.get(), nullptr);
			driver->dispatch(dispatchInfo.workGroupCount[0], dispatchInfo.workGroupCount[1], dispatchInfo.workGroupCount[2]);
		}
		// TODO: tone mapping and stuff

		driver->setRenderTarget(nullptr, false);
		driver->blitRenderTargets(blitFBO, nullptr, false, false);

		driver->endScene();

		uint64_t time = device->getTimer()->getRealTime();
		if (time - lastFPSTime > 1000)
		{
			std::wostringstream str;
			str << L"Envmap Example - Irrlicht Engine [" << driver->getName() << "] FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(str.str().c_str());
			lastFPSTime = time;
		}
	}

	// TODO: screenshot
	ext::ScreenShot::createScreenShot(device, blitFBO->getAttachment(video::EFAP_COLOR_ATTACHMENT0), "screenshot.exr");
}
