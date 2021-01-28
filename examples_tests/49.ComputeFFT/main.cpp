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

	constexpr uint32_t num_channels = 1;
	constexpr VkExtent3D fftDim = VkExtent3D{128, 4, 1};
	constexpr uint32_t dataPointBytes = sizeof(float) * num_channels;

	using FFTClass = ext::FFT::FFT;
	auto fftGPUSpecializedShader = FFTClass::createShader(driver, EF_R8G8B8A8_UNORM);
	
	auto fftPipelineLayout = FFTClass::getDefaultPipelineLayout(driver);
	auto fftPipeline = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(fftPipelineLayout), std::move(fftGPUSpecializedShader));

	FFTClass::Uniforms_t fftUniform = {};
	auto fftDispatchInfo_Horizontal = FFTClass::buildParameters(&fftUniform, fftDim, FFTClass::Direction::_Y, num_channels);

	// Allocate(and fill) uniform Buffer
	auto fftUniformBuffer = driver->createFilledDeviceLocalGPUBufferOnDedMem(sizeof(fftUniform), &fftUniform);

	// Allocate Output Buffer
	auto fftOutputBuffer = driver->createDeviceLocalGPUBufferOnDedMem(FFTClass::getOutputBufferSize(fftDim, dataPointBytes));

	// Allocate Input Buffer 
	uint32_t fftInputBufferSize = FFTClass::getInputBufferSize(fftDim, dataPointBytes);
	auto fftInputBuffer = driver->createDeviceLocalGPUBufferOnDedMem(fftInputBufferSize);

	auto fftDescriptorSet = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(fftPipelineLayout->getDescriptorSetLayout(0u)));
	FFTClass::updateDescriptorSet(driver, fftDescriptorSet.get(), fftDim, fftInputBuffer, fftOutputBuffer, fftUniformBuffer);
	
	// Create and Fill GPU Input Buffer
	float * fftInputMem = reinterpret_cast<float *>(_NBL_ALIGNED_MALLOC(fftInputBufferSize, 1));

	for(uint32_t j = 0; j < fftDim.height; ++j) {
		for(uint32_t i = 0; i < fftDim.width; ++i) {
			fftInputMem[i + j * fftDim.width] = (j+1) * i;
		}
	}

	driver->updateBufferRangeViaStagingBuffer(fftInputBuffer.get(), 0, fftInputBufferSize, fftInputMem);

	_NBL_ALIGNED_FREE(fftInputMem);


	E_FORMAT inFormat;
	constexpr auto outFormat = EF_R8G8B8A8_SRGB;
	smart_refctd_ptr<IGPUImage> outImg;
	smart_refctd_ptr<IGPUImageView> outImgView;
	{
		IGPUImageView::SCreationParams imgViewInfo;
		imgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
		imgViewInfo.image = outImg;
		imgViewInfo.viewType = IGPUImageView::ET_2D_ARRAY;
		imgViewInfo.format = outFormat;
		imgViewInfo.subresourceRange.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0u);
		imgViewInfo.subresourceRange.baseMipLevel = 0;
		imgViewInfo.subresourceRange.levelCount = 1;
		imgViewInfo.subresourceRange.baseArrayLayer = 0;
		imgViewInfo.subresourceRange.layerCount = 1;
		// outImg = driver->createDeviceLocalGPUImageOnDedMem(std::move(imgInfo));
		// outImgView = driver->createGPUImageView(IGPUImageView::SCreationParams(imgViewInfo));
	}

	// auto blitFBO = driver->addFrameBuffer();
	// blitFBO->attach(video::EFAP_COLOR_ATTACHMENT0, std::move(outImgView));

	uint32_t outBufferIx = 0u;
	auto lastPresentStamp = std::chrono::high_resolution_clock::now();
	while (device->run() && receiver.keepOpen())
	{
		driver->beginScene(false, false);

		driver->bindComputePipeline(fftPipeline.get());
		driver->bindDescriptorSets(EPBP_COMPUTE, fftPipelineLayout.get(), 0u, 1u, &fftDescriptorSet.get(), nullptr);
		
		driver->pushConstants(fftPipelineLayout.get(), IGPUSpecializedShader::ESS_COMPUTE, 0u, sizeof(uint32_t), &fftDispatchInfo_Horizontal.direction);
		FFTClass::dispatchHelper(driver, fftDispatchInfo_Horizontal, true);

		// driver->blitRenderTargets(blitFBO, nullptr, false, false);

		driver->endScene();
	}

	return 0;
}