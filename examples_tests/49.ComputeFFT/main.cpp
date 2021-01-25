// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>
#include <iostream>
#include <cstdio>


#include "nbl/ext/ToneMapper/CToneMapper.h"
#include "nbl/ext/FFT/FFT.h"
#include "../common/QToQuitEventReceiver.h"

using namespace nbl;
using namespace nbl::core;
using namespace nbl::asset;
using namespace nbl::video;


int main()
{
	nbl::SIrrlichtCreationParameters deviceParams;
	deviceParams.Bits = 24; //may have to set to 32bit for some platforms
	deviceParams.ZBufferBits = 24; //we'd like 32bit here
	deviceParams.DriverType = EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
	deviceParams.WindowSize = dimension2d<uint32_t>(1280, 720);
	deviceParams.Fullscreen = false;
	deviceParams.Vsync = true; //! If supported by target platform
	deviceParams.Doublebuffer = true;
	deviceParams.Stencilbuffer = false; //! This will not even be a choice soon

	auto device = createDeviceEx(deviceParams);
	if (!device)
		return 1; // could not create selected driver.

	QToQuitEventReceiver receiver;
	device->setEventReceiver(&receiver);

	IVideoDriver* driver = device->getVideoDriver();
	
	nbl::io::IFileSystem* filesystem = device->getFileSystem();
	IAssetManager* am = device->getAssetManager();

	constexpr VkExtent3D fftDim = VkExtent3D{256, 1, 1};

	using FFTClass = ext::FFT::FFT;
	auto fftGPUSpecializedShader = FFTClass::createShader(driver, EF_R8G8B8A8_UNORM);
	
	auto fftPipelineLayout = FFTClass::getDefaultPipelineLayout(driver);
	auto fftPipeline = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(fftPipelineLayout), std::move(fftGPUSpecializedShader));

	FFTClass::Uniforms_t fftUniform = {};
	auto fftDispatchInfo = FFTClass::buildParameters(&fftUniform, fftDim);

	// Allocate(and fill) uniform Buffer
	auto fftUniformBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(fftUniform), &fftUniform);

	// Allocate Output Buffer
	auto fftOutputBuffer = driver->createDeviceLocalGPUBufferOnDedMem(FFTClass::getOutputBufferSize(fftDim, sizeof(float)));

	// Allocate Input Buffer 
	uint32_t fftInputBufferSize = FFTClass::getInputBufferSize(fftDim, sizeof(float));
	auto fftInputBuffer = driver->createDeviceLocalGPUBufferOnDedMem(fftInputBufferSize);

	auto fftDescriptorSet = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(fftPipelineLayout->getDescriptorSetLayout(0u)));
	FFTClass::updateDescriptorSet(driver, fftDescriptorSet.get(), fftInputBuffer, fftUniformBuffer, fftUniformBuffer);
	
	// Create and Fill GPU Input Buffer
	float * fftInputMem = reinterpret_cast<float *>(_NBL_ALIGNED_MALLOC(fftInputBufferSize, 1));

	for(uint32_t i = 0; i < fftDim.width; ++i) {
		fftInputMem[i] = i;
	}

	video::IDriverMemoryAllocation::MemoryRange updateRange(0, fftInputBufferSize); 
	fftInputBuffer->updateSubRange(updateRange, fftInputMem);
		
	_NBL_ALIGNED_FREE(fftInputMem);





	IAssetLoader::SAssetLoadParams lp;
	auto imageBundle = am->getAsset("../../media/noises/spp_benchmark_4k_512.exr", lp);

	E_FORMAT inFormat;
	constexpr auto outFormat = EF_R8G8B8A8_SRGB;
	smart_refctd_ptr<IGPUImage> outImg;
	smart_refctd_ptr<IGPUImageView> imgToTonemapView,outImgView;
	{
		auto cpuImg = IAsset::castDown<ICPUImage>(imageBundle.getContents().begin()[0]);
		IGPUImage::SCreationParams imgInfo = cpuImg->getCreationParameters();
		inFormat = imgInfo.format;

		auto gpuImages = driver->getGPUObjectsFromAssets(&cpuImg.get(),&cpuImg.get()+1);
		auto gpuImage = gpuImages->operator[](0u);

		IGPUImageView::SCreationParams imgViewInfo;
		imgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
		imgViewInfo.image = std::move(gpuImage);
		imgViewInfo.viewType = IGPUImageView::ET_2D_ARRAY;
		imgViewInfo.format = inFormat;
		imgViewInfo.subresourceRange.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0u);
		imgViewInfo.subresourceRange.baseMipLevel = 0;
		imgViewInfo.subresourceRange.levelCount = 1;
		imgViewInfo.subresourceRange.baseArrayLayer = 0;
		imgViewInfo.subresourceRange.layerCount = 1;
		imgToTonemapView = driver->createGPUImageView(IGPUImageView::SCreationParams(imgViewInfo));

		imgInfo.format = outFormat;
		outImg = driver->createDeviceLocalGPUImageOnDedMem(std::move(imgInfo));

		imgViewInfo.image = outImg;
		imgViewInfo.format = outFormat;
		outImgView = driver->createGPUImageView(IGPUImageView::SCreationParams(imgViewInfo));
	}

	const auto inputColorSpace = std::make_tuple(inFormat,ECP_SRGB,EOTF_IDENTITY);

	using LumaMeterClass = ext::LumaMeter::CLumaMeter;
	constexpr auto MeterMode = LumaMeterClass::EMM_MEDIAN;
	const float minLuma = 1.f/2048.f;
	const float maxLuma = 65536.f;

	auto cpuLumaMeasureSpecializedShader = LumaMeterClass::createShader(nullptr, inputColorSpace,MeterMode,minLuma,maxLuma);
	auto gpuLumaMeasureShader = driver->createGPUShader(smart_refctd_ptr<const ICPUShader>(cpuLumaMeasureSpecializedShader->getUnspecialized()));
	auto gpuLumaMeasureSpecializedShader = driver->createGPUSpecializedShader(gpuLumaMeasureShader.get(), cpuLumaMeasureSpecializedShader->getSpecializationInfo());

	const float meteringMinUV[2] = { 0.1f,0.1f };
	const float meteringMaxUV[2] = { 0.9f,0.9f };
	LumaMeterClass::Uniforms_t<MeterMode> uniforms;
	auto lumaDispatchInfo = LumaMeterClass::buildParameters(uniforms, outImg->getCreationParameters().extent, meteringMinUV, meteringMaxUV);

	auto uniformBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(uniforms),&uniforms);


	using ToneMapperClass = ext::ToneMapper::CToneMapper;
	constexpr auto TMO = ToneMapperClass::EO_ACES;
	constexpr bool usingLumaMeter = MeterMode<LumaMeterClass::EMM_COUNT;

	auto cpuTonemappingSpecializedShader = ToneMapperClass::createShader(nullptr,
		inputColorSpace,
		std::make_tuple(outFormat,ECP_SRGB,OETF_sRGB),
		TMO,usingLumaMeter,MeterMode,minLuma,maxLuma,false
	);
	auto gpuTonemappingShader = driver->createGPUShader(smart_refctd_ptr<const ICPUShader>(cpuTonemappingSpecializedShader->getUnspecialized()));
	auto gpuTonemappingSpecializedShader = driver->createGPUSpecializedShader(gpuTonemappingShader.get(),cpuTonemappingSpecializedShader->getSpecializationInfo());

	auto outImgStorage = ToneMapperClass::createViewForImage(driver,false,core::smart_refctd_ptr(outImg),{static_cast<IImage::E_ASPECT_FLAGS>(0u),0,1,0,1});

	auto parameterBuffer = driver->createDeviceLocalGPUBufferOnDedMem(ToneMapperClass::getParameterBufferSize<TMO,MeterMode>());
	constexpr float Exposure = 0.f;
	constexpr float Key = 0.18;
	auto params = ToneMapperClass::Params_t<TMO>(Exposure, Key, 0.85f);
	{
		params.setAdaptationFactorFromFrameDelta(0.f);
		driver->updateBufferRangeViaStagingBuffer(parameterBuffer.get(),0u,sizeof(params),&params);
	}

	auto commonPipelineLayout = ToneMapperClass::getDefaultPipelineLayout(driver,usingLumaMeter);

	auto lumaMeteringPipeline = driver->createGPUComputePipeline(nullptr,core::smart_refctd_ptr(commonPipelineLayout),std::move(gpuLumaMeasureSpecializedShader));
	auto toneMappingPipeline = driver->createGPUComputePipeline(nullptr,core::smart_refctd_ptr(commonPipelineLayout),std::move(gpuTonemappingSpecializedShader));

	auto commonDescriptorSet = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(commonPipelineLayout->getDescriptorSetLayout(0u)));
	ToneMapperClass::updateDescriptorSet<TMO,MeterMode>(driver,commonDescriptorSet.get(),parameterBuffer,imgToTonemapView,outImgStorage,1u,2u,usingLumaMeter ? 3u:0u,uniformBuffer,0u,false);


	constexpr auto dynOffsetArrayLen = usingLumaMeter ? 2u : 1u;

	auto lumaDynamicOffsetArray = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<uint32_t> >(dynOffsetArrayLen,0u);
	lumaDynamicOffsetArray->back() = sizeof(ToneMapperClass::Params_t<TMO>);

	auto toneDynamicOffsetArray = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<uint32_t> >(dynOffsetArrayLen,0u);


	auto blitFBO = driver->addFrameBuffer();
	blitFBO->attach(video::EFAP_COLOR_ATTACHMENT0, std::move(outImgView));

	uint32_t outBufferIx = 0u;
	auto lastPresentStamp = std::chrono::high_resolution_clock::now();
	while (device->run() && receiver.keepOpen())
	{
		driver->beginScene(false, false);

		driver->bindComputePipeline(fftPipeline.get());
		driver->bindDescriptorSets(EPBP_COMPUTE, fftPipelineLayout.get(), 0u, 1u, &fftDescriptorSet.get(), nullptr);
		/// driver->pushConstants(commonPipelineLayout.get(),IGPUSpecializedShader::ESS_COMPUTE,0u,sizeof(outBufferIx),&outBufferIx); outBufferIx ^= 0x1u;
		FFTClass::dispatchHelper(driver, fftDispatchInfo, true);

		//driver->bindComputePipeline(toneMappingPipeline.get());
		//driver->bindDescriptorSets(EPBP_COMPUTE,commonPipelineLayout.get(),0u,1u,&commonDescriptorSet.get(),&toneDynamicOffsetArray);
		//ToneMapperClass::dispatchHelper(driver,outImgStorage.get(),true);

		driver->blitRenderTargets(blitFBO, nullptr, false, false);

		driver->endScene();
	}

	return 0;
}