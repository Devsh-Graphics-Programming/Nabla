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

struct DispatchInfo_t
{
	uint32_t workGroupDims[3];
	uint32_t workGroupCount[3];
};

static inline core::smart_refctd_ptr<video::IGPUPipelineLayout> getPipelineLayout_Convolution(video::IVideoDriver* driver) {
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
	
	using FFTClass = ext::FFT::FFT;
	core::SRange<const asset::SPushConstantRange> pcRange = FFTClass::getDefaultPushConstantRanges();
	core::SRange<const video::IGPUDescriptorSetLayout::SBinding> bindings = {bnd, bnd+sizeof(bnd)/sizeof(IGPUDescriptorSetLayout::SBinding)};;

	return driver->createGPUPipelineLayout(
		pcRange.begin(),pcRange.end(),
		driver->createGPUDescriptorSetLayout(bindings.begin(),bindings.end()),nullptr,nullptr,nullptr
	);
}
static inline core::smart_refctd_ptr<video::IGPUSpecializedShader> createShader_Convolution(
	video::IVideoDriver* driver,
	IAssetManager* am,
	uint32_t maxDimensionSize) 
{
uint32_t const maxPaddedDimensionSize = core::roundUpToPoT(maxDimensionSize);

	const char* sourceFmt =
R"===(#version 430 core

#define _NBL_GLSL_WORKGROUP_SIZE_ %u
#define _NBL_GLSL_EXT_FFT_MAX_DIM_SIZE_ %u
#define _NBL_GLSL_EXT_FFT_MAX_ITEMS_PER_THREAD %u
 
#include "../fft_convolve_ifft.comp"

)===";

	const size_t extraSize = 32 + 32 + 32 + 32;
	
	constexpr uint32_t DEFAULT_WORK_GROUP_SIZE = 256u;
	const uint32_t maxItemsPerThread = (maxPaddedDimensionSize - 1u) / (DEFAULT_WORK_GROUP_SIZE) + 1u;
	auto shader = core::make_smart_refctd_ptr<ICPUBuffer>(strlen(sourceFmt)+extraSize+1u);
	snprintf(
		reinterpret_cast<char*>(shader->getPointer()),shader->getSize(), sourceFmt,
		DEFAULT_WORK_GROUP_SIZE,
		maxPaddedDimensionSize,
		maxItemsPerThread
	);

	auto cpuSpecializedShader = core::make_smart_refctd_ptr<ICPUSpecializedShader>(
		core::make_smart_refctd_ptr<ICPUShader>(std::move(shader),ICPUShader::buffer_contains_glsl),
		ISpecializedShader::SInfo{nullptr, nullptr, "main", asset::ISpecializedShader::ESS_COMPUTE}
	);
	
	auto gpuShader = driver->createGPUShader(nbl::core::smart_refctd_ptr<const ICPUShader>(cpuSpecializedShader->getUnspecialized()));
	
	auto gpuSpecializedShader = driver->createGPUSpecializedShader(gpuShader.get(), cpuSpecializedShader->getSpecializationInfo());

	return gpuSpecializedShader;
}
static inline void updateDescriptorSet_Convolution (
	video::IVideoDriver * driver,
	video::IGPUDescriptorSet * set,
	core::smart_refctd_ptr<video::IGPUBuffer> inputOutputBufferDescriptor,
	core::smart_refctd_ptr<video::IGPUBuffer> kernelBufferDescriptor)
{
	constexpr uint32_t descCount = 2u;
	video::IGPUDescriptorSet::SDescriptorInfo pInfos[descCount];
	video::IGPUDescriptorSet::SWriteDescriptorSet pWrites[descCount];

	for (auto i = 0; i < descCount; i++)
	{
		pWrites[i].dstSet = set;
		pWrites[i].arrayElement = 0u;
		pWrites[i].count = 1u;
		pWrites[i].info = pInfos+i;
	}

	// InputOutput Buffer 
	pWrites[0].binding = 0;
	pWrites[0].descriptorType = asset::EDT_STORAGE_BUFFER;
	pWrites[0].count = 1;
	pInfos[0].desc = inputOutputBufferDescriptor;
	pInfos[0].buffer.size = inputOutputBufferDescriptor->getSize();
	pInfos[0].buffer.offset = 0u;

	// Kernel Buffer 
	pWrites[1].binding = 1;
	pWrites[1].descriptorType = asset::EDT_STORAGE_BUFFER;
	pWrites[1].count = 1;
	pInfos[1].desc = kernelBufferDescriptor;
	pInfos[1].buffer.size = kernelBufferDescriptor->getSize();
	pInfos[1].buffer.offset = 0u;

	driver->updateDescriptorSets(descCount, pWrites, 0u, nullptr);
}

static inline core::smart_refctd_ptr<video::IGPUPipelineLayout> getPipelineLayout_LastFFT(video::IVideoDriver* driver) {
	
	using FFTClass = ext::FFT::FFT;

	static const asset::SPushConstantRange ranges[2] =
	{
		{
			ISpecializedShader::ESS_COMPUTE,
			0u,
			sizeof(FFTClass::Parameters_t)
		},
		{
			ISpecializedShader::ESS_COMPUTE,
			sizeof(FFTClass::Parameters_t),
			sizeof(uint32_t) * 3
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
	
	core::SRange<const asset::SPushConstantRange> pcRange = {ranges, ranges+2};
	core::SRange<const video::IGPUDescriptorSetLayout::SBinding> bindings = {bnd, bnd+sizeof(bnd)/sizeof(IGPUDescriptorSetLayout::SBinding)};;

	return driver->createGPUPipelineLayout(
		pcRange.begin(),pcRange.end(),
		driver->createGPUDescriptorSetLayout(bindings.begin(),bindings.end()),nullptr,nullptr,nullptr
	);
}
static inline core::smart_refctd_ptr<video::IGPUSpecializedShader> createShader_LastFFT(
	video::IVideoDriver* driver,
	IAssetManager* am,
	uint32_t maxDimensionSize) {
	
uint32_t const maxPaddedDimensionSize = core::roundUpToPoT(maxDimensionSize);

	const char* sourceFmt =
R"===(#version 430 core

#define _NBL_GLSL_WORKGROUP_SIZE_ %u
#define _NBL_GLSL_EXT_FFT_MAX_DIM_SIZE_ %u
#define _NBL_GLSL_EXT_FFT_MAX_ITEMS_PER_THREAD %u

#include "../last_fft.comp"

)===";

	const size_t extraSize = 32 + 32 + 32 + 32;
	
	constexpr uint32_t DEFAULT_WORK_GROUP_SIZE = 256u;
	const uint32_t maxItemsPerThread = (maxPaddedDimensionSize - 1u) / (DEFAULT_WORK_GROUP_SIZE) + 1u;
	auto shader = core::make_smart_refctd_ptr<ICPUBuffer>(strlen(sourceFmt)+extraSize+1u);
	snprintf(
		reinterpret_cast<char*>(shader->getPointer()),shader->getSize(), sourceFmt,
		DEFAULT_WORK_GROUP_SIZE,
		maxPaddedDimensionSize,
		maxItemsPerThread
	);

	auto cpuSpecializedShader = core::make_smart_refctd_ptr<ICPUSpecializedShader>(
		core::make_smart_refctd_ptr<ICPUShader>(std::move(shader),ICPUShader::buffer_contains_glsl),
		ISpecializedShader::SInfo{nullptr, nullptr, "main", asset::ISpecializedShader::ESS_COMPUTE}
	);
	
	auto gpuShader = driver->createGPUShader(nbl::core::smart_refctd_ptr<const ICPUShader>(cpuSpecializedShader->getUnspecialized()));
	
	auto gpuSpecializedShader = driver->createGPUSpecializedShader(gpuShader.get(), cpuSpecializedShader->getSpecializationInfo());

	return gpuSpecializedShader;
}
static inline void updateDescriptorSet_LastFFT (
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
	// Loading SrcImage and Kernel Image from File

	IAssetLoader::SAssetLoadParams lp;
	auto srcImageBundle = am->getAsset("../../media/colorexr.exr", lp);
	auto srcCpuImg = IAsset::castDown<ICPUImage>(srcImageBundle.getContents().begin()[0]);
	auto kerImageBundle = am->getAsset("../../media/kernels/physical_flare_512.exr", lp);
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
	
	VkExtent3D paddedDim = FFTClass::padDimensionToNextPOT(srcDim, kerDim);
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
	
	auto fftDispatchInfo_Horizontal = FFTClass::buildParameters(paddedDim, FFTClass::Direction::X);
	auto fftDispatchInfo_Vertical = FFTClass::buildParameters(paddedDim, FFTClass::Direction::Y);

	auto convolveShader = createShader_Convolution(driver, am, maxPaddedDimensionSize);
	auto convolvePipelineLayout = getPipelineLayout_Convolution(driver);
	auto convolvePipeline = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(convolvePipelineLayout), std::move(convolveShader));

	auto lastFFTShader = createShader_LastFFT(driver, am, maxPaddedDimensionSize);
	auto lastFFTPipelineLayout = getPipelineLayout_LastFFT(driver);
	auto lastFFTPipeline = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(lastFFTPipelineLayout), std::move(lastFFTShader));

	// Allocate Output Buffer
	auto fftOutputBuffer_0 = driver->createDeviceLocalGPUBufferOnDedMem(FFTClass::getOutputBufferSize(paddedDim, srcNumChannels)); // result of: srcFFTX and kerFFTX and Convolution and IFFTY
	auto fftOutputBuffer_1 = driver->createDeviceLocalGPUBufferOnDedMem(FFTClass::getOutputBufferSize(paddedDim, srcNumChannels)); // result of: srcFFTY and IFFTX 
	auto fftOutputBuffer_KernelNormalized = driver->createDeviceLocalGPUBufferOnDedMem(FFTClass::getOutputBufferSize(paddedDim, srcNumChannels)); // result of: kerFFTY


	// Precompute Kernel FFT
	{
		// Ker FFT X 
		auto fftDescriptorSet_Ker_FFT_X = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(fftPipelineLayout_ImageInput->getDescriptorSetLayout(0u)));
		FFTClass::updateDescriptorSet(driver, fftDescriptorSet_Ker_FFT_X.get(), kerImageView, fftOutputBuffer_0, ISampler::ETC_CLAMP_TO_BORDER);

		// Ker FFT Y
		auto fftDescriptorSet_Ker_FFT_Y = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(fftPipelineLayout_SSBOInput->getDescriptorSetLayout(0u)));
		FFTClass::updateDescriptorSet(driver, fftDescriptorSet_Ker_FFT_Y.get(), fftOutputBuffer_0, fftOutputBuffer_1);
		
		// Normalization of FFT Y result
		auto fftDescriptorSet_KernelNormalization = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(fftPipelineLayout_KernelNormalization->getDescriptorSetLayout(0u)));
		FFTClass::updateDescriptorSet_KernelNormalization(driver, fftDescriptorSet_KernelNormalization.get(), fftOutputBuffer_1, fftOutputBuffer_KernelNormalized);

		// Ker Image FFT X
		driver->bindComputePipeline(fftPipeline_ImageInput.get());
		driver->bindDescriptorSets(EPBP_COMPUTE, fftPipelineLayout_ImageInput.get(), 0u, 1u, &fftDescriptorSet_Ker_FFT_X.get(), nullptr);
		FFTClass::pushConstants(driver, fftPipelineLayout_ImageInput.get(), kerDim, paddedDim, FFTClass::Direction::X, false, srcNumChannels, FFTClass::PaddingType::FILL_WITH_ZERO);
		FFTClass::dispatchHelper(driver, fftDispatchInfo_Horizontal);

		// Ker Image FFT Y
		driver->bindComputePipeline(fftPipeline_SSBOInput.get());
		driver->bindDescriptorSets(EPBP_COMPUTE, fftPipelineLayout_SSBOInput.get(), 0u, 1u, &fftDescriptorSet_Ker_FFT_Y.get(), nullptr);
		FFTClass::pushConstants(driver, fftPipelineLayout_SSBOInput.get(), paddedDim, paddedDim, FFTClass::Direction::Y, false, srcNumChannels);
		FFTClass::dispatchHelper(driver, fftDispatchInfo_Vertical);
		
		// Ker Image FFT Y
		driver->bindComputePipeline(fftPipeline_SSBOInput.get());
		driver->bindDescriptorSets(EPBP_COMPUTE, fftPipelineLayout_SSBOInput.get(), 0u, 1u, &fftDescriptorSet_Ker_FFT_Y.get(), nullptr);
		FFTClass::pushConstants(driver, fftPipelineLayout_SSBOInput.get(), paddedDim, paddedDim, FFTClass::Direction::Y, false, srcNumChannels);
		FFTClass::dispatchHelper(driver, fftDispatchInfo_Vertical);
		
		// Ker Normalization
		driver->bindComputePipeline(fftPipeline_KernelNormalization.get());
		driver->bindDescriptorSets(EPBP_COMPUTE, fftPipelineLayout_KernelNormalization.get(), 0u, 1u, &fftDescriptorSet_KernelNormalization.get(), nullptr);
		FFTClass::dispatchKernelNormalization(driver, paddedDim, srcNumChannels);
	}

	// Src FFT X 
	auto fftDescriptorSet_Src_FFT_X = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(fftPipelineLayout_ImageInput->getDescriptorSetLayout(0u)));
	FFTClass::updateDescriptorSet(driver, fftDescriptorSet_Src_FFT_X.get(), srcImageView, fftOutputBuffer_0, ISampler::ETC_CLAMP_TO_EDGE);

	// Convolution
	auto convolveDescriptorSet = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(convolvePipelineLayout->getDescriptorSetLayout(0u)));
	updateDescriptorSet_Convolution(driver, convolveDescriptorSet.get(), fftOutputBuffer_0, fftOutputBuffer_KernelNormalized);

	// Last IFFTX 
	auto lastFFTDescriptorSet = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(lastFFTPipelineLayout->getDescriptorSetLayout(0u)));
	updateDescriptorSet_LastFFT(driver, lastFFTDescriptorSet.get(), fftOutputBuffer_0, outImgView);

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
		FFTClass::pushConstants(driver, fftPipelineLayout_ImageInput.get(), srcDim, paddedDim, FFTClass::Direction::X, false, srcNumChannels, FFTClass::PaddingType::CLAMP_TO_EDGE);
		FFTClass::dispatchHelper(driver, fftDispatchInfo_Horizontal);

		// Src Image FFT Y + Convolution + Convolved IFFT Y
		driver->bindComputePipeline(convolvePipeline.get());
		driver->bindDescriptorSets(EPBP_COMPUTE, convolvePipelineLayout.get(), 0u, 1u, &convolveDescriptorSet.get(), nullptr);
		FFTClass::pushConstants(driver, convolvePipelineLayout.get(), paddedDim, paddedDim, FFTClass::Direction::Y, false, srcNumChannels);
		FFTClass::dispatchHelper(driver, fftDispatchInfo_Vertical);

		// Last FFT Padding and Copy to GPU Image
		driver->bindComputePipeline(lastFFTPipeline.get());
		driver->bindDescriptorSets(EPBP_COMPUTE, lastFFTPipelineLayout.get(), 0u, 1u, &lastFFTDescriptorSet.get(), nullptr);
		FFTClass::pushConstants(driver, lastFFTPipelineLayout.get(), paddedDim, paddedDim, FFTClass::Direction::X, true, srcNumChannels);
		driver->pushConstants(lastFFTPipelineLayout.get(), nbl::video::IGPUSpecializedShader::ESS_COMPUTE, sizeof(FFTClass::Parameters_t), sizeof(uint32_t) * 3, &kerDim); // numSrcChannels
		FFTClass::dispatchHelper(driver, fftDispatchInfo_Horizontal);
		
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