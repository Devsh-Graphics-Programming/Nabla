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

static inline core::smart_refctd_ptr<video::IGPUPipelineLayout> getPipelineLayout_RemovePadding(video::IVideoDriver* driver) {
	static const asset::SPushConstantRange ranges[1] =
	{
		{
			ISpecializedShader::ESS_COMPUTE,
			0u,
			sizeof(uint32_t)
		},
	};

	static IGPUDescriptorSetLayout::SBinding bnd[] =
	{
		{
			0u,
			EDT_STORAGE_BUFFER,
			1u,
			ISpecializedShader::ESS_COMPUTE,
			nullptr
		},
		{
			1u,
			EDT_STORAGE_IMAGE,
			1u,
			ISpecializedShader::ESS_COMPUTE,
			nullptr
		},
	};
	
	core::SRange<const asset::SPushConstantRange> pcRange = {ranges, ranges+1};
	core::SRange<const video::IGPUDescriptorSetLayout::SBinding> bindings = {bnd, bnd+sizeof(bnd)/sizeof(IGPUDescriptorSetLayout::SBinding)};;

	return driver->createGPUPipelineLayout(
		pcRange.begin(),pcRange.end(),
		driver->createGPUDescriptorSetLayout(bindings.begin(),bindings.end()),nullptr,nullptr,nullptr
	);
}
static inline core::smart_refctd_ptr<video::IGPUSpecializedShader> createShader_RemovePadding(
	video::IVideoDriver* driver,
	nbl::asset::IGLSLCompiler * glslc,
	nbl::io::IFileSystem * fs) {
    io::IReadFile* file = fs->createAndOpenFile("../remove_padding.comp");
    auto cpucs = glslc->resolveIncludeDirectives(file, asset::ISpecializedShader::ESS_COMPUTE, "../remove_padding.comp");
    auto cs = driver->createGPUShader(std::move(cpucs));
    file->drop();
    asset::ISpecializedShader::SInfo csinfo(nullptr, nullptr, "main", asset::ISpecializedShader::ESS_COMPUTE, "../remove_padding.vert");
    auto cs_spec = driver->createGPUSpecializedShader(cs.get(), csinfo);
	return cs_spec;
}
static inline void updateDescriptorSet_RemovePadding (
	video::IVideoDriver * driver,
	video::IGPUDescriptorSet * set,
	core::smart_refctd_ptr<video::IGPUBuffer> inputBufferDescriptor,
	core::smart_refctd_ptr<video::IGPUImageView> outputImageDescriptor)
{
	video::IGPUDescriptorSet::SDescriptorInfo pInfos[2];
	video::IGPUDescriptorSet::SWriteDescriptorSet pWrites[2];

	for (auto i = 0; i< 2; i++)
	{
		pWrites[i].dstSet = set;
		pWrites[i].arrayElement = 0u;
		pWrites[i].count = 1u;
		pWrites[i].info = pInfos+i;
	}

	// Input Buffer 
	pWrites[0].binding = 0;
	pWrites[0].descriptorType = asset::EDT_STORAGE_BUFFER;
	pWrites[0].count = 1;
	pInfos[0].desc = inputBufferDescriptor;
	pInfos[0].buffer.size = inputBufferDescriptor->getSize();
	pInfos[0].buffer.offset = 0u;

	// Output Buffer 
	pWrites[1].binding = 1;
	pWrites[1].descriptorType = asset::EDT_STORAGE_IMAGE;
	pWrites[1].count = 1;
	pInfos[1].desc = outputImageDescriptor;
	pInfos[1].image.sampler = nullptr;
	pInfos[1].image.imageLayout = static_cast<asset::E_IMAGE_LAYOUT>(0u);;

	driver->updateDescriptorSets(2u, pWrites, 0u, nullptr);
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
    auto* fs = am->getFileSystem();
    auto* glslc = am->getGLSLCompiler();
	
	// Loading SrcImage and Kernel Image from File

	IAssetLoader::SAssetLoadParams lp;
	auto imageBundle = am->getAsset("../../media/colorexr.exr", lp);
	
	IGPUImage::SCreationParams srcImgInfo;
	
	smart_refctd_ptr<IGPUImage> outImg;
	smart_refctd_ptr<IGPUImageView> srcImageView, outImgView;
	{
		auto cpuImg = IAsset::castDown<ICPUImage>(imageBundle.getContents().begin()[0]);
		srcImgInfo = cpuImg->getCreationParameters();

		auto gpuImages = driver->getGPUObjectsFromAssets(&cpuImg.get(),&cpuImg.get()+1);
		auto gpuImage = gpuImages->operator[](0u);

		IGPUImageView::SCreationParams imgViewInfo;
		imgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
		imgViewInfo.image = std::move(gpuImage);
		imgViewInfo.viewType = IGPUImageView::ET_2D;
		imgViewInfo.format = srcImgInfo.format;
		imgViewInfo.subresourceRange.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0u);
		imgViewInfo.subresourceRange.baseMipLevel = 0;
		imgViewInfo.subresourceRange.levelCount = 1;
		imgViewInfo.subresourceRange.baseArrayLayer = 0;
		imgViewInfo.subresourceRange.layerCount = 1;
		srcImageView = driver->createGPUImageView(IGPUImageView::SCreationParams(imgViewInfo));
		
		outImg = driver->createDeviceLocalGPUImageOnDedMem(std::move(srcImgInfo));

		imgViewInfo.image = outImg;
		imgViewInfo.format = srcImgInfo.format;
		outImgView = driver->createGPUImageView(IGPUImageView::SCreationParams(imgViewInfo));
	}
	using FFTClass = ext::FFT::FFT;
	
	E_FORMAT srcFormat = srcImgInfo.format;
	VkExtent3D srcDim = srcImgInfo.extent;
	uint32_t srcNumChannels = getFormatChannelCount(srcImgInfo.format);
	VkExtent3D srcPaddedDim = padDimensionToNextPOT(srcDim);

	uint32_t maxPaddedDimensionSize = core::max(core::max(srcPaddedDim.width, srcPaddedDim.height), srcPaddedDim.depth);
	
	auto fftGPUSpecializedShader_SSBOInput = FFTClass::createShader(driver, FFTClass::DataType::SSBO, maxPaddedDimensionSize);
	auto fftGPUSpecializedShader_ImageInput = FFTClass::createShader(driver, FFTClass::DataType::TEXTURE2D, maxPaddedDimensionSize);
	
	auto fftPipelineLayout_SSBOInput = FFTClass::getDefaultPipelineLayout(driver, FFTClass::DataType::SSBO);
	auto fftPipelineLayout_ImageInput = FFTClass::getDefaultPipelineLayout(driver, FFTClass::DataType::TEXTURE2D);

	auto fftPipeline_SSBOInput = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(fftPipelineLayout_SSBOInput), std::move(fftGPUSpecializedShader_SSBOInput));
	auto fftPipeline_ImageInput = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(fftPipelineLayout_ImageInput), std::move(fftGPUSpecializedShader_ImageInput));
	
	auto fftDispatchInfo_SrcImage_Horizontal = FFTClass::buildParameters(srcPaddedDim, FFTClass::Direction::X, srcNumChannels);
	auto fftDispatchInfo_SrcImage_Vertical = FFTClass::buildParameters(srcPaddedDim, FFTClass::Direction::Y, srcNumChannels);

	// Allocate Output Buffer
	auto fftOutputBuffer_0 = driver->createDeviceLocalGPUBufferOnDedMem(FFTClass::getOutputBufferSize(srcPaddedDim, srcFormat));
	auto fftOutputBuffer_1 = driver->createDeviceLocalGPUBufferOnDedMem(FFTClass::getOutputBufferSize(srcPaddedDim, srcFormat));

	// FFT X
	auto fftDescriptorSet_Src_FFT_X = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(fftPipelineLayout_ImageInput->getDescriptorSetLayout(0u)));
	FFTClass::updateDescriptorSet(driver, fftDescriptorSet_Src_FFT_X.get(), srcImageView, fftOutputBuffer_0);

	// FFT Y
	auto fftDescriptorSet_Src_FFT_Y = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(fftPipelineLayout_SSBOInput->getDescriptorSetLayout(0u)));
	FFTClass::updateDescriptorSet(driver, fftDescriptorSet_Src_FFT_Y.get(), fftOutputBuffer_0, fftOutputBuffer_1);
	
	// IFFT X
	auto fftDescriptorSet_IFFT_X = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(fftPipelineLayout_SSBOInput->getDescriptorSetLayout(0u)));
	FFTClass::updateDescriptorSet(driver, fftDescriptorSet_IFFT_X.get(), fftOutputBuffer_1, fftOutputBuffer_0);

	// IFFT Y
	auto fftDescriptorSet_IFFT_Y = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(fftPipelineLayout_SSBOInput->getDescriptorSetLayout(0u)));
	FFTClass::updateDescriptorSet(driver, fftDescriptorSet_IFFT_Y.get(), fftOutputBuffer_0, fftOutputBuffer_1);
	
	auto removePaddingShader = createShader_RemovePadding(driver, glslc, fs);
	auto removePaddingPipelineLayout = getPipelineLayout_RemovePadding(driver);
	auto removePaddingDescriptorSet = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(removePaddingPipelineLayout->getDescriptorSetLayout(0u)));
	updateDescriptorSet_RemovePadding(driver, removePaddingDescriptorSet.get(), fftOutputBuffer_1, outImgView);

	uint32_t outBufferIx = 0u;
	auto lastPresentStamp = std::chrono::high_resolution_clock::now();
	while (device->run() && receiver.keepOpen())
	{
		driver->beginScene(false, false);

		// Src Image FFT X
		driver->bindComputePipeline(fftPipeline_ImageInput.get());
		driver->bindDescriptorSets(EPBP_COMPUTE, fftPipelineLayout_ImageInput.get(), 0u, 1u, &fftDescriptorSet_Src_FFT_X.get(), nullptr);
		FFTClass::pushConstants(driver, fftPipelineLayout_ImageInput.get(), srcDim, srcPaddedDim, FFTClass::Direction::X, false, FFTClass::PaddingType::CLAMP_TO_EDGE);
		FFTClass::dispatchHelper(driver, fftDispatchInfo_SrcImage_Horizontal);

		driver->bindComputePipeline(fftPipeline_SSBOInput.get());

		// Src Image FFT Y
		driver->bindDescriptorSets(EPBP_COMPUTE, fftPipelineLayout_SSBOInput.get(), 0u, 1u, &fftDescriptorSet_Src_FFT_Y.get(), nullptr);
		FFTClass::pushConstants(driver, fftPipelineLayout_SSBOInput.get(), srcPaddedDim, srcPaddedDim, FFTClass::Direction::Y, false, FFTClass::PaddingType::FILL_WITH_ZERO);
		FFTClass::dispatchHelper(driver, fftDispatchInfo_SrcImage_Vertical);

		// Combined IFFT X
		driver->bindDescriptorSets(EPBP_COMPUTE, fftPipelineLayout_SSBOInput.get(), 0u, 1u, &fftDescriptorSet_IFFT_X.get(), nullptr);
		FFTClass::pushConstants(driver, fftPipelineLayout_SSBOInput.get(), srcPaddedDim, srcPaddedDim, FFTClass::Direction::X, true, FFTClass::PaddingType::FILL_WITH_ZERO);
		FFTClass::dispatchHelper(driver, fftDispatchInfo_SrcImage_Horizontal);
		
		// Combined IFFT Y
		driver->bindDescriptorSets(EPBP_COMPUTE, fftPipelineLayout_SSBOInput.get(), 0u, 1u, &fftDescriptorSet_IFFT_Y.get(), nullptr);
		FFTClass::pushConstants(driver, fftPipelineLayout_SSBOInput.get(), srcPaddedDim, srcPaddedDim, FFTClass::Direction::Y, true, FFTClass::PaddingType::FILL_WITH_ZERO);
		FFTClass::dispatchHelper(driver, fftDispatchInfo_SrcImage_Vertical);

		driver->endScene();
	}

	return 0;
}