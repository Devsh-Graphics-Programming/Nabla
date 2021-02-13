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
#include "../../../../source/Nabla/COpenGLExtensionHandler.h"

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

struct DispatchInfo_t
{
	uint32_t workGroupDims[3];
	uint32_t workGroupCount[3];
};

static inline core::smart_refctd_ptr<video::IGPUPipelineLayout> getPipelineLayout_Convolution(video::IVideoDriver* driver) {
	static const asset::SPushConstantRange ranges[2] =
	{
		{
			ISpecializedShader::ESS_COMPUTE,
			0u,
			sizeof(uint32_t) * 3
		},
		{
			ISpecializedShader::ESS_COMPUTE,
			sizeof(uint32_t) * 4,
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
			EDT_STORAGE_BUFFER,
			1u,
			ISpecializedShader::ESS_COMPUTE,
			nullptr
		},
		{
			2u,
			EDT_STORAGE_BUFFER,
			1u,
			ISpecializedShader::ESS_COMPUTE,
			nullptr
		},
	};
	
	core::SRange<const asset::SPushConstantRange> pcRange = {ranges, ranges+2};
	core::SRange<const video::IGPUDescriptorSetLayout::SBinding> bindings = {bnd, bnd+sizeof(bnd)/sizeof(IGPUDescriptorSetLayout::SBinding)};;

	return driver->createGPUPipelineLayout(
		pcRange.begin(),pcRange.end(),
		driver->createGPUDescriptorSetLayout(bindings.begin(),bindings.end()),nullptr,nullptr,nullptr
	);
}
static inline core::smart_refctd_ptr<video::IGPUSpecializedShader> createShader_Convolution(
	video::IVideoDriver* driver,
	IAssetManager* am) {
	IAssetLoader::SAssetLoadParams lp;
	auto file_path = "../convolve.comp";
	auto shaderAsset = am->getAsset(file_path, lp);
	auto cpucs = IAsset::castDown<ICPUSpecializedShader>(shaderAsset.getContents().begin()[0]);
	auto cs = driver->createGPUShader(nbl::core::smart_refctd_ptr<const ICPUShader>((cpucs->getUnspecialized())));
	asset::ISpecializedShader::SInfo csinfo(nullptr, nullptr, "main", asset::ISpecializedShader::ESS_COMPUTE, file_path);
	auto cs_spec = driver->createGPUSpecializedShader(cs.get(), csinfo);
	return cs_spec;
}
static inline void updateDescriptorSet_Convolution (
	video::IVideoDriver * driver,
	video::IGPUDescriptorSet * set,
	core::smart_refctd_ptr<video::IGPUBuffer> sourceBufferDescriptor,
	core::smart_refctd_ptr<video::IGPUBuffer> kernelBufferDescriptor,
	core::smart_refctd_ptr<video::IGPUBuffer> outputBufferDescriptor)
{
	video::IGPUDescriptorSet::SDescriptorInfo pInfos[3];
	video::IGPUDescriptorSet::SWriteDescriptorSet pWrites[3];

	for (auto i = 0; i < 3; i++)
	{
		pWrites[i].dstSet = set;
		pWrites[i].arrayElement = 0u;
		pWrites[i].count = 1u;
		pWrites[i].info = pInfos+i;
	}

	// Source Buffer 
	pWrites[0].binding = 0;
	pWrites[0].descriptorType = asset::EDT_STORAGE_BUFFER;
	pWrites[0].count = 1;
	pInfos[0].desc = sourceBufferDescriptor;
	pInfos[0].buffer.size = sourceBufferDescriptor->getSize();
	pInfos[0].buffer.offset = 0u;

	// Kernel Buffer 
	pWrites[1].binding = 1;
	pWrites[1].descriptorType = asset::EDT_STORAGE_BUFFER;
	pWrites[1].count = 1;
	pInfos[1].desc = kernelBufferDescriptor;
	pInfos[1].buffer.size = kernelBufferDescriptor->getSize();
	pInfos[1].buffer.offset = 0u;
	
	// Output Buffer 
	pWrites[2].binding = 2;
	pWrites[2].descriptorType = asset::EDT_STORAGE_BUFFER;
	pWrites[2].count = 1;
	pInfos[2].desc = outputBufferDescriptor;
	pInfos[2].buffer.size = outputBufferDescriptor->getSize();
	pInfos[2].buffer.offset = 0u;

	driver->updateDescriptorSets(3u, pWrites, 0u, nullptr);
}
static inline void dispatchHelper_Convolution(
	video::IVideoDriver* driver,
	const DispatchInfo_t& dispatchInfo)
{
	driver->dispatch(dispatchInfo.workGroupCount[0], dispatchInfo.workGroupCount[1], dispatchInfo.workGroupCount[2]);
	COpenGLExtensionHandler::pGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}
static inline DispatchInfo_t getDispatchInfo_Convolution(
	asset::VkExtent3D const & paddedDimension,
	uint32_t numChannels)
{
	DispatchInfo_t ret = {};

	ret.workGroupDims[0] = 256;
	ret.workGroupDims[1] = 1;
	ret.workGroupDims[2] = 1;

	ret.workGroupCount[0] = core::ceil(float(paddedDimension.width * paddedDimension.height * paddedDimension.depth * numChannels) / ret.workGroupDims[0]);
	ret.workGroupCount[1] = 1;
	ret.workGroupCount[2] = 1;

	return ret;
}


static inline core::smart_refctd_ptr<video::IGPUPipelineLayout> getPipelineLayout_RemovePadding(video::IVideoDriver* driver) {
	static const asset::SPushConstantRange ranges[3] =
	{
		{
			ISpecializedShader::ESS_COMPUTE,
			0u,
			sizeof(uint32_t) * 3
		},
		{
			ISpecializedShader::ESS_COMPUTE,
			sizeof(uint32_t) * 4,
			sizeof(uint32_t) * 3
		},
		{
			ISpecializedShader::ESS_COMPUTE,
			sizeof(uint32_t) * 8,
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
	
	core::SRange<const asset::SPushConstantRange> pcRange = {ranges, ranges+3};
	core::SRange<const video::IGPUDescriptorSetLayout::SBinding> bindings = {bnd, bnd+sizeof(bnd)/sizeof(IGPUDescriptorSetLayout::SBinding)};;

	return driver->createGPUPipelineLayout(
		pcRange.begin(),pcRange.end(),
		driver->createGPUDescriptorSetLayout(bindings.begin(),bindings.end()),nullptr,nullptr,nullptr
	);
}
static inline core::smart_refctd_ptr<video::IGPUSpecializedShader> createShader_RemovePadding(
	video::IVideoDriver* driver,
	IAssetManager* am) {

	IAssetLoader::SAssetLoadParams lp;
	auto file_path = "../remove_padding.comp";
	auto shaderAsset = am->getAsset(file_path, lp);
	auto cpucs = IAsset::castDown<ICPUSpecializedShader>(shaderAsset.getContents().begin()[0]);
	auto cs = driver->createGPUShader(nbl::core::smart_refctd_ptr<const ICPUShader>((cpucs->getUnspecialized())));
	asset::ISpecializedShader::SInfo csinfo(nullptr, nullptr, "main", asset::ISpecializedShader::ESS_COMPUTE, file_path);
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
static inline void dispatchHelper_RemovePadding(
	video::IVideoDriver* driver,
	const DispatchInfo_t& dispatchInfo)
{
	driver->dispatch(dispatchInfo.workGroupCount[0], dispatchInfo.workGroupCount[1], dispatchInfo.workGroupCount[2]);
	COpenGLExtensionHandler::pGlMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}
static inline DispatchInfo_t getDispatchInfo_RemovePadding(
	asset::VkExtent3D const & inputDimensions)
{
	DispatchInfo_t ret = {};

	ret.workGroupDims[0] = 16;
	ret.workGroupDims[1] = 16;
	ret.workGroupDims[2] = 1;

	ret.workGroupCount[0] = core::ceil(float(inputDimensions.width) / ret.workGroupDims[0]);
	ret.workGroupCount[1] = core::ceil(float(inputDimensions.height) / ret.workGroupDims[1]);
	ret.workGroupCount[2] = core::ceil(float(inputDimensions.depth) / ret.workGroupDims[2]);

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
	// Loading SrcImage and Kernel Image from File

	IAssetLoader::SAssetLoadParams lp;
	auto srcImageBundle = am->getAsset("../../media/colorexr.exr", lp);
	auto srcCpuImg = IAsset::castDown<ICPUImage>(srcImageBundle.getContents().begin()[0]);
	auto kerImageBundle = am->getAsset("../../media/kernels/gaussian_kernel_21x21.exr", lp);
	auto kerCpuImg = IAsset::castDown<ICPUImage>(kerImageBundle.getContents().begin()[0]);
	
	IGPUImage::SCreationParams srcImgInfo;
	IGPUImage::SCreationParams kerImgInfo;
	
	smart_refctd_ptr<IGPUImage> outImg;
	smart_refctd_ptr<IGPUImageView> outImgView;

	smart_refctd_ptr<IGPUImageView> srcImageView;
	IGPUImageView::SCreationParams srcImgViewInfo;
	{
		srcImgInfo = srcCpuImg->getCreationParameters();

		auto srcGpuImages = driver->getGPUObjectsFromAssets(&srcCpuImg.get(),&srcCpuImg.get()+1);
		auto srcGpuImage = srcGpuImages->operator[](0u);

		srcImgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
		srcImgViewInfo.image = std::move(srcGpuImage);
		srcImgViewInfo.viewType = IGPUImageView::ET_2D;
		srcImgViewInfo.format = srcImgInfo.format;
		srcImgViewInfo.subresourceRange.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0u);
		srcImgViewInfo.subresourceRange.baseMipLevel = 0;
		srcImgViewInfo.subresourceRange.levelCount = 1;
		srcImgViewInfo.subresourceRange.baseArrayLayer = 0;
		srcImgViewInfo.subresourceRange.layerCount = 1;
		srcImageView = driver->createGPUImageView(IGPUImageView::SCreationParams(srcImgViewInfo));
	}
	smart_refctd_ptr<IGPUImageView> kerImageView;
	{
		kerImgInfo = kerCpuImg->getCreationParameters();

		auto kerGpuImages = driver->getGPUObjectsFromAssets(&kerCpuImg.get(),&kerCpuImg.get()+1);
		auto kerGpuImage = kerGpuImages->operator[](0u);

		IGPUImageView::SCreationParams kerImgViewInfo;
		kerImgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
		kerImgViewInfo.image = std::move(kerGpuImage);
		kerImgViewInfo.viewType = IGPUImageView::ET_2D;
		kerImgViewInfo.format = kerImgInfo.format;
		kerImgViewInfo.subresourceRange.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0u);
		kerImgViewInfo.subresourceRange.baseMipLevel = 0;
		kerImgViewInfo.subresourceRange.levelCount = 1;
		kerImgViewInfo.subresourceRange.baseArrayLayer = 0;
		kerImgViewInfo.subresourceRange.layerCount = 1;
		kerImageView = driver->createGPUImageView(IGPUImageView::SCreationParams(kerImgViewInfo));
	}

	using FFTClass = ext::FFT::FFT;
	
	E_FORMAT srcFormat = srcImgInfo.format;
	E_FORMAT kerFormat = kerImgInfo.format;
	VkExtent3D srcDim = srcImgInfo.extent;
	VkExtent3D kerDim = kerImgInfo.extent;
	uint32_t srcNumChannels = getFormatChannelCount(srcFormat);
	uint32_t kerNumChannels = getFormatChannelCount(kerFormat);
	assert(srcNumChannels == kerNumChannels); // Just to make sure, because the other case is not handled in this example
	
	VkExtent3D paddedDim = padDimensionToNextPOT(srcDim, kerDim);
	uint32_t maxPaddedDimensionSize = core::max(core::max(paddedDim.width, paddedDim.height), paddedDim.depth);
	
	VkExtent3D outImageDim = srcDim;

	// Create Out Image
	{
		srcImgInfo.extent = outImageDim;
		outImg = driver->createDeviceLocalGPUImageOnDedMem(std::move(srcImgInfo));

		srcImgViewInfo.image = outImg;
		srcImgViewInfo.format = srcImgInfo.format;
		outImgView = driver->createGPUImageView(IGPUImageView::SCreationParams(srcImgViewInfo));
	}

	auto fftGPUSpecializedShader_SSBOInput = FFTClass::createShader(driver, FFTClass::DataType::SSBO, maxPaddedDimensionSize);
	auto fftGPUSpecializedShader_ImageInput = FFTClass::createShader(driver, FFTClass::DataType::TEXTURE2D, maxPaddedDimensionSize);
	auto fftGPUSpecializedShader_KernelNormalization = FFTClass::createKernelNormalizationShader(driver, am);
	
	auto fftPipelineLayout_SSBOInput = FFTClass::getDefaultPipelineLayout(driver, FFTClass::DataType::SSBO);
	auto fftPipelineLayout_ImageInput = FFTClass::getDefaultPipelineLayout(driver, FFTClass::DataType::TEXTURE2D);
	auto fftPipelineLayout_KernelNormalization = FFTClass::getPipelineLayout_KernelNormalization(driver);

	auto fftPipeline_SSBOInput = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(fftPipelineLayout_SSBOInput), std::move(fftGPUSpecializedShader_SSBOInput));
	auto fftPipeline_ImageInput = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(fftPipelineLayout_ImageInput), std::move(fftGPUSpecializedShader_ImageInput));
	auto fftPipeline_KernelNormalization = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(fftPipelineLayout_KernelNormalization), std::move(fftGPUSpecializedShader_KernelNormalization));
	
	auto fftDispatchInfo_Horizontal = FFTClass::buildParameters(paddedDim, FFTClass::Direction::X, srcNumChannels);
	auto fftDispatchInfo_Vertical = FFTClass::buildParameters(paddedDim, FFTClass::Direction::Y, srcNumChannels);

	auto convolveShader = createShader_Convolution(driver, am);
	auto convolvePipelineLayout = getPipelineLayout_Convolution(driver);
	auto convolvePipeline = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(convolvePipelineLayout), std::move(convolveShader));
	auto convolveDispatchInfo = getDispatchInfo_Convolution(paddedDim, srcNumChannels);
	

	// Allocate Output Buffer
	auto fftOutputBuffer_0 = driver->createDeviceLocalGPUBufferOnDedMem(FFTClass::getOutputBufferSize(paddedDim, srcNumChannels)); // result of: srcFFTX and kerFFTX and Convolution and IFFTY
	auto fftOutputBuffer_1 = driver->createDeviceLocalGPUBufferOnDedMem(FFTClass::getOutputBufferSize(paddedDim, srcNumChannels)); // result of: srcFFTY and IFFTX 
	auto fftOutputBuffer_KernelNormalized = driver->createDeviceLocalGPUBufferOnDedMem(FFTClass::getOutputBufferSize(paddedDim, srcNumChannels)); // result of: kerFFTY


	// Precompute Kernel FFT
	{
		// Ker FFT X 
		auto fftDescriptorSet_Ker_FFT_X = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(fftPipelineLayout_ImageInput->getDescriptorSetLayout(0u)));
		FFTClass::updateDescriptorSet(driver, fftDescriptorSet_Ker_FFT_X.get(), kerImageView, fftOutputBuffer_0);

		// Ker FFT Y
		auto fftDescriptorSet_Ker_FFT_Y = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(fftPipelineLayout_SSBOInput->getDescriptorSetLayout(0u)));
		FFTClass::updateDescriptorSet(driver, fftDescriptorSet_Ker_FFT_Y.get(), fftOutputBuffer_0, fftOutputBuffer_1);
		
		// Normalization of FFT Y result
		auto fftDescriptorSet_KernelNormalization = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(fftPipelineLayout_KernelNormalization->getDescriptorSetLayout(0u)));
		FFTClass::updateDescriptorSet_KernelNormalization(driver, fftDescriptorSet_KernelNormalization.get(), fftOutputBuffer_1, fftOutputBuffer_KernelNormalized);

		// Ker Image FFT X
		driver->bindComputePipeline(fftPipeline_ImageInput.get());
		driver->bindDescriptorSets(EPBP_COMPUTE, fftPipelineLayout_ImageInput.get(), 0u, 1u, &fftDescriptorSet_Ker_FFT_X.get(), nullptr);
		FFTClass::pushConstants(driver, fftPipelineLayout_ImageInput.get(), kerDim, paddedDim, FFTClass::Direction::X, false, FFTClass::PaddingType::FILL_WITH_ZERO);
		FFTClass::dispatchHelper(driver, fftDispatchInfo_Horizontal);

		// Ker Image FFT Y
		driver->bindComputePipeline(fftPipeline_SSBOInput.get());
		driver->bindDescriptorSets(EPBP_COMPUTE, fftPipelineLayout_SSBOInput.get(), 0u, 1u, &fftDescriptorSet_Ker_FFT_Y.get(), nullptr);
		FFTClass::pushConstants(driver, fftPipelineLayout_SSBOInput.get(), paddedDim, paddedDim, FFTClass::Direction::Y, false);
		FFTClass::dispatchHelper(driver, fftDispatchInfo_Vertical);
		
		// Ker Image FFT Y
		driver->bindComputePipeline(fftPipeline_SSBOInput.get());
		driver->bindDescriptorSets(EPBP_COMPUTE, fftPipelineLayout_SSBOInput.get(), 0u, 1u, &fftDescriptorSet_Ker_FFT_Y.get(), nullptr);
		FFTClass::pushConstants(driver, fftPipelineLayout_SSBOInput.get(), paddedDim, paddedDim, FFTClass::Direction::Y, false);
		FFTClass::dispatchHelper(driver, fftDispatchInfo_Vertical);
		
		// Ker Normalization
		driver->bindComputePipeline(fftPipeline_KernelNormalization.get());
		driver->bindDescriptorSets(EPBP_COMPUTE, fftPipelineLayout_KernelNormalization.get(), 0u, 1u, &fftDescriptorSet_KernelNormalization.get(), nullptr);
		FFTClass::dispatchKernelNormalization(driver, paddedDim, srcNumChannels);
	}

	// Src FFT X 
	auto fftDescriptorSet_Src_FFT_X = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(fftPipelineLayout_ImageInput->getDescriptorSetLayout(0u)));
	FFTClass::updateDescriptorSet(driver, fftDescriptorSet_Src_FFT_X.get(), srcImageView, fftOutputBuffer_0);

	// Src FFT Y
	auto fftDescriptorSet_Src_FFT_Y = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(fftPipelineLayout_SSBOInput->getDescriptorSetLayout(0u)));
	FFTClass::updateDescriptorSet(driver, fftDescriptorSet_Src_FFT_Y.get(), fftOutputBuffer_0, fftOutputBuffer_1);

	// Convolution
	auto convolveDescriptorSet = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(convolvePipelineLayout->getDescriptorSetLayout(0u)));
	updateDescriptorSet_Convolution(driver, convolveDescriptorSet.get(), fftOutputBuffer_1, fftOutputBuffer_KernelNormalized, fftOutputBuffer_0);

	// IFFT X
	auto fftDescriptorSet_IFFT_X = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(fftPipelineLayout_SSBOInput->getDescriptorSetLayout(0u)));
	FFTClass::updateDescriptorSet(driver, fftDescriptorSet_IFFT_X.get(), fftOutputBuffer_0, fftOutputBuffer_1);

	// IFFT Y
	auto fftDescriptorSet_IFFT_Y = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(fftPipelineLayout_SSBOInput->getDescriptorSetLayout(0u)));
	FFTClass::updateDescriptorSet(driver, fftDescriptorSet_IFFT_Y.get(), fftOutputBuffer_1, fftOutputBuffer_0);
	
	auto removePaddingShader = createShader_RemovePadding(driver, am);
	auto removePaddingPipelineLayout = getPipelineLayout_RemovePadding(driver);
	auto removePaddingPipeline = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(removePaddingPipelineLayout), std::move(removePaddingShader));
	auto removePaddingDescriptorSet = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(removePaddingPipelineLayout->getDescriptorSetLayout(0u)));
	updateDescriptorSet_RemovePadding(driver, removePaddingDescriptorSet.get(), fftOutputBuffer_0, outImgView);
	auto removePaddingDispatchInfo = getDispatchInfo_RemovePadding(outImageDim);

	uint32_t outBufferIx = 0u;
	auto lastPresentStamp = std::chrono::high_resolution_clock::now();
	bool savedToFile = false;
	
	auto downloadStagingArea = driver->getDefaultDownStreamingBuffer();
	
	auto blitFBO = driver->addFrameBuffer();
	blitFBO->attach(video::EFAP_COLOR_ATTACHMENT0, std::move(outImgView));

	while (device->run() && receiver.keepOpen())
	{
		driver->beginScene(false, false);

		// Src Image FFT X
		driver->bindComputePipeline(fftPipeline_ImageInput.get());
		driver->bindDescriptorSets(EPBP_COMPUTE, fftPipelineLayout_ImageInput.get(), 0u, 1u, &fftDescriptorSet_Src_FFT_X.get(), nullptr);
		FFTClass::pushConstants(driver, fftPipelineLayout_ImageInput.get(), srcDim, paddedDim, FFTClass::Direction::X, false, FFTClass::PaddingType::CLAMP_TO_EDGE);
		FFTClass::dispatchHelper(driver, fftDispatchInfo_Horizontal);

		// Src Image FFT Y
		driver->bindComputePipeline(fftPipeline_SSBOInput.get());
		driver->bindDescriptorSets(EPBP_COMPUTE, fftPipelineLayout_SSBOInput.get(), 0u, 1u, &fftDescriptorSet_Src_FFT_Y.get(), nullptr);
		FFTClass::pushConstants(driver, fftPipelineLayout_SSBOInput.get(), paddedDim, paddedDim, FFTClass::Direction::Y, false);
		FFTClass::dispatchHelper(driver, fftDispatchInfo_Vertical);

		// Convolution
		driver->bindComputePipeline(convolvePipeline.get());
		driver->bindDescriptorSets(EPBP_COMPUTE, convolvePipelineLayout.get(), 0u, 1u, &convolveDescriptorSet.get(), nullptr);
		driver->pushConstants(convolvePipelineLayout.get(), nbl::video::IGPUSpecializedShader::ESS_COMPUTE, 0u, sizeof(uint32_t) * 3, &paddedDim); // pc.numChannels
		driver->pushConstants(convolvePipelineLayout.get(), nbl::video::IGPUSpecializedShader::ESS_COMPUTE, sizeof(uint32_t) * 4, sizeof(uint32_t), &srcNumChannels); // numSrcChannels
		dispatchHelper_Convolution(driver, convolveDispatchInfo);
		
		// Convolved IFFT X
		driver->bindComputePipeline(fftPipeline_SSBOInput.get());
		driver->bindDescriptorSets(EPBP_COMPUTE, fftPipelineLayout_SSBOInput.get(), 0u, 1u, &fftDescriptorSet_IFFT_X.get(), nullptr);
		FFTClass::pushConstants(driver, fftPipelineLayout_SSBOInput.get(), paddedDim, paddedDim, FFTClass::Direction::X, true);
		FFTClass::dispatchHelper(driver, fftDispatchInfo_Horizontal);
		
		// Convolved IFFT Y
		driver->bindComputePipeline(fftPipeline_SSBOInput.get());
		driver->bindDescriptorSets(EPBP_COMPUTE, fftPipelineLayout_SSBOInput.get(), 0u, 1u, &fftDescriptorSet_IFFT_Y.get(), nullptr);
		FFTClass::pushConstants(driver, fftPipelineLayout_SSBOInput.get(), paddedDim, paddedDim, FFTClass::Direction::Y, true);
		FFTClass::dispatchHelper(driver, fftDispatchInfo_Vertical);

		// Remove Padding and Copy to GPU Image
		driver->bindComputePipeline(removePaddingPipeline.get());
		driver->bindDescriptorSets(EPBP_COMPUTE, removePaddingPipelineLayout.get(), 0u, 1u, &removePaddingDescriptorSet.get(), nullptr);
		driver->pushConstants(removePaddingPipelineLayout.get(), nbl::video::IGPUSpecializedShader::ESS_COMPUTE, 0u, sizeof(uint32_t) * 3, &paddedDim); // pc.numChannels
		driver->pushConstants(removePaddingPipelineLayout.get(), nbl::video::IGPUSpecializedShader::ESS_COMPUTE, sizeof(uint32_t) * 4, sizeof(uint32_t) * 3, &kerDim); // numSrcChannels
		driver->pushConstants(removePaddingPipelineLayout.get(), nbl::video::IGPUSpecializedShader::ESS_COMPUTE, sizeof(uint32_t) * 8, sizeof(uint32_t), &srcNumChannels); // numSrcChannels
		dispatchHelper_RemovePadding(driver, removePaddingDispatchInfo);
		
		if(false == savedToFile) {
			savedToFile = true;
			
			core::smart_refctd_ptr<ICPUImageView> imageView;
			const uint32_t colorBufferBytesize = srcDim.height * srcDim.width * asset::getTexelOrBlockBytesize(srcFormat);

			// create image
			ICPUImage::SCreationParams imgParams;
			imgParams.flags = static_cast<ICPUImage::E_CREATE_FLAGS>(0u); // no flags
			imgParams.type = ICPUImage::ET_2D;
			imgParams.format = srcFormat;
			imgParams.extent = srcDim;
			imgParams.mipLevels = 1u;
			imgParams.arrayLayers = 1u;
			imgParams.samples = ICPUImage::ESCF_1_BIT;
			auto image = ICPUImage::create(std::move(imgParams));

			constexpr uint64_t timeoutInNanoSeconds = 300000000000u;
			const auto waitPoint = std::chrono::high_resolution_clock::now()+std::chrono::nanoseconds(timeoutInNanoSeconds);

			uint32_t address = std::remove_pointer<decltype(downloadStagingArea)>::type::invalid_address; // remember without initializing the address to be allocated to invalid_address you won't get an allocation!
			const uint32_t alignment = 4096u; // common page size
			auto unallocatedSize = downloadStagingArea->multi_alloc(waitPoint, 1u, &address, &colorBufferBytesize, &alignment);
			if (unallocatedSize)
			{
				os::Printer::log("Could not download the buffer from the GPU!", ELL_ERROR);
			}

			// set up regions
			auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<IImage::SBufferCopy> >(1u);
			{
				auto& region = regions->front();

				region.bufferOffset = 0u;
				region.bufferRowLength = srcCpuImg->getRegions().begin()[0].bufferRowLength;
				region.bufferImageHeight = srcDim.height;
				//region.imageSubresource.aspectMask = wait for Vulkan;
				region.imageSubresource.mipLevel = 0u;
				region.imageSubresource.baseArrayLayer = 0u;
				region.imageSubresource.layerCount = 1u;
				region.imageOffset = { 0u,0u,0u };
				region.imageExtent = imgParams.extent;
			}

			driver->copyImageToBuffer(outImg.get(), downloadStagingArea->getBuffer(), 1, &regions->front());

			auto downloadFence = driver->placeFence(true);

			auto* data = reinterpret_cast<uint8_t*>(downloadStagingArea->getBufferPointer()) + address;
			auto cpubufferalias = core::make_smart_refctd_ptr<asset::CCustomAllocatorCPUBuffer<core::null_allocator<uint8_t> > >(colorBufferBytesize, data, core::adopt_memory);
			image->setBufferAndRegions(std::move(cpubufferalias),regions);
			
			// wait for download fence and then invalidate the CPU cache
			{
				auto result = downloadFence->waitCPU(timeoutInNanoSeconds,true);
				if (result==E_DRIVER_FENCE_RETVAL::EDFR_TIMEOUT_EXPIRED||result==E_DRIVER_FENCE_RETVAL::EDFR_FAIL)
				{
					os::Printer::log("Could not download the buffer from the GPU, fence not signalled!", ELL_ERROR);
					downloadStagingArea->multi_free(1u, &address, &colorBufferBytesize, nullptr);
					continue;
				}
				if (downloadStagingArea->needsManualFlushOrInvalidate())
					driver->invalidateMappedMemoryRanges({{downloadStagingArea->getBuffer()->getBoundMemory(),address,colorBufferBytesize}});
			}

			// create image view
			ICPUImageView::SCreationParams imgViewParams;
			imgViewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
			imgViewParams.format = image->getCreationParameters().format;
			imgViewParams.image = std::move(image);
			imgViewParams.viewType = ICPUImageView::ET_2D;
			imgViewParams.subresourceRange = {static_cast<IImage::E_ASPECT_FLAGS>(0u),0u,1u,0u,1u};
			imageView = ICPUImageView::create(std::move(imgViewParams));

			IAssetWriter::SAssetWriteParams wp(imageView.get());
			volatile bool success = am->writeAsset("convolved_exr.exr", wp);
			assert(success);
		}
		
		driver->blitRenderTargets(blitFBO, nullptr, false, false);

		driver->endScene();
	}

	return 0;
}