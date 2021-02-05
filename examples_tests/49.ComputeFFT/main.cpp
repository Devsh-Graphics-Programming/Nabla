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

#include "nbl/core/math/intutil.h"
#include "nbl/core/math/glslFunctions.h"

VkExtent3D padDimensionToNextPOT(VkExtent3D const & dimension, VkExtent3D const & minimum_dimension = VkExtent3D{ 0, 0, 0 }) {
	VkExtent3D ret = {};
	VkExtent3D extendedDim = dimension;

	if(dimension.width < minimum_dimension.width) {
		extendedDim.width = minimum_dimension.width;
	}
	if(dimension.height < minimum_dimension.height) {
		extendedDim.height = minimum_dimension.height;
	}
	if(dimension.depth < minimum_dimension.depth) {
		extendedDim.depth = minimum_dimension.depth;
	}

	ret.width = roundUpToPoT(extendedDim.width);
	ret.height = roundUpToPoT(extendedDim.height);
	ret.depth = roundUpToPoT(extendedDim.depth);

	return ret;
}

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
	
	using FFTClass = ext::FFT::FFT;

	constexpr uint32_t num_channels = 1;
	constexpr uint32_t dataPointBytes = sizeof(float) * num_channels;

	constexpr VkExtent3D fftDim = VkExtent3D{6, 1, 1};
	VkExtent3D fftPaddedDim = padDimensionToNextPOT(fftDim);
	uint32_t maxPaddedDimensionSize = core::max(core::max(fftPaddedDim.width, fftPaddedDim.height), fftPaddedDim.depth);
	
	auto fftGPUSpecializedShader = FFTClass::createShader(driver, FFTClass::DataType::SSBO, EF_UNKNOWN, maxPaddedDimensionSize);
	
	auto fftPipelineLayout = FFTClass::getDefaultPipelineLayout(driver, FFTClass::DataType::SSBO);
	auto fftPipeline = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(fftPipelineLayout), std::move(fftGPUSpecializedShader));

	auto fftDispatchInfo_Horizontal = FFTClass::buildParameters(fftPaddedDim, FFTClass::Direction::X, num_channels);

	// Allocate Output Buffer
	auto fftOutputBuffer = driver->createDeviceLocalGPUBufferOnDedMem(FFTClass::getOutputBufferSize(fftPaddedDim, dataPointBytes));

	// Allocate Input Buffer SSBO
	uint32_t fftInputBufferSize = dataPointBytes * fftDim.width * fftDim.height * fftDim.depth;
	auto fftInputBuffer = driver->createDeviceLocalGPUBufferOnDedMem(fftInputBufferSize);

	auto fftDescriptorSet = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(fftPipelineLayout->getDescriptorSetLayout(0u)));
	FFTClass::updateDescriptorSet(driver, fftDescriptorSet.get(), fftInputBuffer, fftOutputBuffer);
	
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
		
		FFTClass::pushConstants(driver, fftPipelineLayout.get(), fftDim, fftPaddedDim, FFTClass::Direction::X, false, FFTClass::PaddingType::FILL_WITH_ZERO);
		FFTClass::dispatchHelper(driver, fftDispatchInfo_Horizontal, true);

		// driver->blitRenderTargets(blitFBO, nullptr, false, false);

		driver->endScene();
	}

	return 0;
}