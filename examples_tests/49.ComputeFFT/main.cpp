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

constexpr uint32_t channelCountOverride = 3u;

inline smart_refctd_ptr<IGPUPipelineLayout> getPipelineLayout_Convolution(IVideoDriver* driver)
{
	IGPUSampler::SParams params =
	{
		{
			ISampler::ETC_REPEAT,
			ISampler::ETC_REPEAT,
			ISampler::ETC_REPEAT,
			ISampler::ETBC_FLOAT_OPAQUE_BLACK,
			ISampler::ETF_LINEAR, // is it needed?
			ISampler::ETF_LINEAR,
			ISampler::ESMM_NEAREST,
			0u,
			0u,
			ISampler::ECO_ALWAYS
		}
	};
	auto sampler = driver->createGPUSampler(std::move(params));
	smart_refctd_ptr<IGPUSampler> samplers[channelCountOverride];
	std::fill_n(samplers,channelCountOverride,sampler);

	IGPUDescriptorSetLayout::SBinding bnd[] =
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
			EDT_COMBINED_IMAGE_SAMPLER,
			channelCountOverride,
			ISpecializedShader::ESS_COMPUTE,
			samplers
		}
	};
	
	using FFTClass = ext::FFT::FFT;
	core::SRange<const asset::SPushConstantRange> pcRange = FFTClass::getDefaultPushConstantRanges();
	core::SRange<const video::IGPUDescriptorSetLayout::SBinding> bindings = {bnd,bnd+sizeof(bnd)/sizeof(IGPUDescriptorSetLayout::SBinding)};

	return driver->createGPUPipelineLayout(
		pcRange.begin(),pcRange.end(),
		driver->createGPUDescriptorSetLayout(bindings.begin(),bindings.end()),nullptr,nullptr,nullptr
	);
}

inline core::smart_refctd_ptr<video::IGPUSpecializedShader> createShader_Convolution(
	video::IVideoDriver* driver,
	IAssetManager* am,
	uint32_t maxDimensionSize) 
{
	const uint32_t maxPaddedDimensionSize = core::roundUpToPoT(maxDimensionSize);

	const char* sourceFmt =
R"===(#version 430 core

#define _NBL_GLSL_WORKGROUP_SIZE_ %u
#define _NBL_GLSL_EXT_FFT_MAX_DIM_SIZE_ %u
 
#include "../fft_convolve_ifft.comp"

)===";

	const size_t extraSize = 32 + 32 + 32 + 32;
	
	constexpr uint32_t DEFAULT_WORK_GROUP_SIZE = 256u;
	auto shader = core::make_smart_refctd_ptr<ICPUBuffer>(strlen(sourceFmt)+extraSize+1u);
	snprintf(
		reinterpret_cast<char*>(shader->getPointer()),shader->getSize(), sourceFmt,
		DEFAULT_WORK_GROUP_SIZE,
		maxPaddedDimensionSize
	);

	auto cpuSpecializedShader = core::make_smart_refctd_ptr<ICPUSpecializedShader>(
		core::make_smart_refctd_ptr<ICPUShader>(std::move(shader),ICPUShader::buffer_contains_glsl),
		ISpecializedShader::SInfo{nullptr, nullptr, "main", asset::ISpecializedShader::ESS_COMPUTE}
	);
	
	auto gpuShader = driver->createGPUShader(nbl::core::smart_refctd_ptr<const ICPUShader>(cpuSpecializedShader->getUnspecialized()));
	
	auto gpuSpecializedShader = driver->createGPUSpecializedShader(gpuShader.get(), cpuSpecializedShader->getSpecializationInfo());

	return gpuSpecializedShader;
}
inline void updateDescriptorSet_Convolution (
	video::IVideoDriver * driver,
	video::IGPUDescriptorSet * set,
	core::smart_refctd_ptr<video::IGPUBuffer> inputOutputBufferDescriptor,
	const core::smart_refctd_ptr<video::IGPUImageView>* kernelNormalizedSpectrumImageDescriptors)
{
	constexpr uint32_t descCount = 2u;
	video::IGPUDescriptorSet::SDescriptorInfo pInfos[1u+channelCountOverride];
	video::IGPUDescriptorSet::SWriteDescriptorSet pWrites[descCount];

	for (auto i = 0; i < descCount; i++)
	{
		pWrites[i].dstSet = set;
		pWrites[i].arrayElement = 0u;
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
	pWrites[1].descriptorType = asset::EDT_COMBINED_IMAGE_SAMPLER;
	pWrites[1].count = channelCountOverride;
	for (uint32_t i=0u; i<channelCountOverride; i++)
	{
		auto& info = pInfos[1u+i];
		info.desc = kernelNormalizedSpectrumImageDescriptors[i];
		//info.image.imageLayout = ;
		info.image.sampler = nullptr;
	}

	driver->updateDescriptorSets(descCount, pWrites, 0u, nullptr);
}

static inline core::smart_refctd_ptr<video::IGPUPipelineLayout> getPipelineLayout_LastFFT(video::IVideoDriver* driver) {
	
	using FFTClass = ext::FFT::FFT;

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

	using FFTClass = ext::FFT::FFT;
	core::SRange<const asset::SPushConstantRange> pcRange = FFTClass::getDefaultPushConstantRanges();
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

#include "../last_fft.comp"

)===";

	const size_t extraSize = 32 + 32 + 32 + 32;
	
	constexpr uint32_t DEFAULT_WORK_GROUP_SIZE = 256u;
	auto shader = core::make_smart_refctd_ptr<ICPUBuffer>(strlen(sourceFmt)+extraSize+1u);
	snprintf(
		reinterpret_cast<char*>(shader->getPointer()),shader->getSize(), sourceFmt,
		DEFAULT_WORK_GROUP_SIZE,
		maxPaddedDimensionSize
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
	auto kerImageBundle = am->getAsset("../../media/kernels/physical_flare_512.exr", lp);

	smart_refctd_ptr<IGPUImageView> srcImageView;
	{
		auto srcGpuImages = driver->getGPUObjectsFromAssets<ICPUImage>(srcImageBundle.getContents());

		IGPUImageView::SCreationParams srcImgViewInfo;
		srcImgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
		srcImgViewInfo.image = srcGpuImages->operator[](0u);
		srcImgViewInfo.viewType = IGPUImageView::ET_2D;
		srcImgViewInfo.format = srcImgViewInfo.image->getCreationParameters().format;
		srcImgViewInfo.subresourceRange.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0u);
		srcImgViewInfo.subresourceRange.baseMipLevel = 0;
		srcImgViewInfo.subresourceRange.levelCount = 1;
		srcImgViewInfo.subresourceRange.baseArrayLayer = 0;
		srcImgViewInfo.subresourceRange.layerCount = 1;
		srcImageView = driver->createGPUImageView(std::move(srcImgViewInfo));
	}
	smart_refctd_ptr<IGPUImageView> kerImageView;
	{
		auto kerGpuImages = driver->getGPUObjectsFromAssets<ICPUImage>(kerImageBundle.getContents());

		IGPUImageView::SCreationParams kerImgViewInfo;
		kerImgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
		kerImgViewInfo.image = kerGpuImages->operator[](0u);
		kerImgViewInfo.viewType = IGPUImageView::ET_2D;
		kerImgViewInfo.format = kerImgViewInfo.image->getCreationParameters().format;
		kerImgViewInfo.subresourceRange.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0u);
		kerImgViewInfo.subresourceRange.baseMipLevel = 0;
		kerImgViewInfo.subresourceRange.levelCount = 1;
		kerImgViewInfo.subresourceRange.baseArrayLayer = 0;
		kerImgViewInfo.subresourceRange.layerCount = 1;
		kerImageView = driver->createGPUImageView(std::move(kerImgViewInfo));
	}

	using FFTClass = ext::FFT::FFT;
	
	const E_FORMAT srcFormat = srcImageView->getCreationParameters().format;
	uint32_t srcNumChannels = getFormatChannelCount(srcFormat);
	uint32_t kerNumChannels = getFormatChannelCount(kerImageView->getCreationParameters().format);
	//! OVERRIDE (we dont need alpha)
	srcNumChannels = channelCountOverride;
	kerNumChannels = channelCountOverride;
	assert(srcNumChannels == kerNumChannels); // Just to make sure, because the other case is not handled in this example
	
	const auto srcDim = srcImageView->getCreationParameters().image->getCreationParameters().extent;

	// Create Out Image
	smart_refctd_ptr<IGPUImage> outImg;
	smart_refctd_ptr<IGPUImageView> outImgView;
	{
		auto dstImgViewInfo = srcImageView->getCreationParameters();

		auto dstImgInfo = dstImgViewInfo.image->getCreationParameters();
		outImg = driver->createDeviceLocalGPUImageOnDedMem(std::move(dstImgInfo));

		dstImgViewInfo.image = outImg;
		outImgView = driver->createGPUImageView(IGPUImageView::SCreationParams(dstImgViewInfo));
	}
	 // TODO: re-examine
	const VkExtent3D paddedDim = FFTClass::padDimensionToNextPOT(srcDim);
	auto fftGPUSpecializedShader_ImageInput = FFTClass::createShader(driver, FFTClass::DataType::TEXTURE2D, paddedDim.width);
	
	auto fftPipelineLayout_ImageInput = FFTClass::getDefaultPipelineLayout(driver, FFTClass::DataType::TEXTURE2D);
	auto fftPipeline_ImageInput = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(fftPipelineLayout_ImageInput), std::move(fftGPUSpecializedShader_ImageInput));

	auto convolveShader = createShader_Convolution(driver, am, paddedDim.height);
	auto convolvePipelineLayout = getPipelineLayout_Convolution(driver);
	auto convolvePipeline = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(convolvePipelineLayout), std::move(convolveShader));

	auto lastFFTShader = createShader_LastFFT(driver, am, paddedDim.width);
	auto lastFFTPipelineLayout = getPipelineLayout_LastFFT(driver);
	auto lastFFTPipeline = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(lastFFTPipelineLayout), std::move(lastFFTShader));

	// Allocate Output Buffer
	auto fftOutputBuffer_0 = driver->createDeviceLocalGPUBufferOnDedMem(FFTClass::getOutputBufferSize(paddedDim, srcNumChannels)); // result of: srcFFTX and kerFFTX and Convolution and IFFTY
	auto fftOutputBuffer_1 = driver->createDeviceLocalGPUBufferOnDedMem(FFTClass::getOutputBufferSize(paddedDim, srcNumChannels)); // result of: srcFFTY and IFFTX 
	core::smart_refctd_ptr<IGPUImageView> kernelNormalizedSpectrums[channelCountOverride];

	// Precompute Kernel FFT
	{
		const auto kerDim = kerImageView->getCreationParameters().image->getCreationParameters().extent;
		const VkExtent3D paddedKerDim = FFTClass::padDimensionToNextPOT(kerDim);

		// create kernel spectrums
		auto createKernelSpectrum = [&]() -> auto
		{
			video::IGPUImage::SCreationParams imageParams;
			imageParams.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
			imageParams.type = asset::IImage::ET_2D;
			imageParams.format = asset::EF_R16G16_SFLOAT;
			imageParams.extent = { paddedKerDim.width,paddedKerDim.height,1u};
			imageParams.mipLevels = 1u;
			imageParams.arrayLayers = 1u;
			imageParams.samples = asset::IImage::ESCF_1_BIT;

			video::IGPUImageView::SCreationParams viewParams;
			viewParams.flags = static_cast<video::IGPUImageView::E_CREATE_FLAGS>(0u);
			viewParams.image = driver->createGPUImageOnDedMem(std::move(imageParams),driver->getDeviceLocalGPUMemoryReqs());
			viewParams.viewType = video::IGPUImageView::ET_2D;
			viewParams.format = asset::EF_R16G16_SFLOAT;
			viewParams.components = {};
			viewParams.subresourceRange = {};
			viewParams.subresourceRange.levelCount = 1u;
			viewParams.subresourceRange.layerCount = 1u;
			return driver->createGPUImageView(std::move(viewParams));
		};
		for (uint32_t i=0u; i<channelCountOverride; i++)
			kernelNormalizedSpectrums[i] = createKernelSpectrum();

		// Ker FFT X 
		auto fftDescriptorSet_Ker_FFT_X = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(fftPipelineLayout_ImageInput->getDescriptorSetLayout(0u)));
		FFTClass::updateDescriptorSet(driver, fftDescriptorSet_Ker_FFT_X.get(), kerImageView, fftOutputBuffer_0, ISampler::ETC_CLAMP_TO_BORDER);

		// Ker FFT Y
		auto fftPipelineLayout_SSBOInput = FFTClass::getDefaultPipelineLayout(driver, FFTClass::DataType::SSBO);
		auto fftDescriptorSet_Ker_FFT_Y = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(fftPipelineLayout_SSBOInput->getDescriptorSetLayout(0u)));
		FFTClass::updateDescriptorSet(driver, fftDescriptorSet_Ker_FFT_Y.get(), fftOutputBuffer_0, fftOutputBuffer_1);
		
		// Normalization of FFT Y result
		auto fftPipelineLayout_KernelNormalization = [&]() -> auto
		{
			IGPUDescriptorSetLayout::SBinding bnd[] =
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
					channelCountOverride,
					ISpecializedShader::ESS_COMPUTE,
					nullptr
				},
			};
			return driver->createGPUPipelineLayout(
				nullptr,nullptr,
				driver->createGPUDescriptorSetLayout(bnd,bnd+2),nullptr,nullptr,nullptr
			);
		}();
		auto fftDescriptorSet_KernelNormalization = [&]() -> auto
		{
			auto dset = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(fftPipelineLayout_KernelNormalization->getDescriptorSetLayout(0u)));

			video::IGPUDescriptorSet::SDescriptorInfo pInfos[1+channelCountOverride];
			video::IGPUDescriptorSet::SWriteDescriptorSet pWrites[2];

			for (auto i = 0; i < 2; i++)
			{
				pWrites[i].dstSet = dset.get();
				pWrites[i].arrayElement = 0u;
				pWrites[i].count = 1u;
				pWrites[i].info = pInfos + i;
			}

			// In Buffer 
			pWrites[0].binding = 0;
			pWrites[0].descriptorType = asset::EDT_STORAGE_BUFFER;
			pWrites[0].count = 1;
			pInfos[0].desc = fftOutputBuffer_1;
			pInfos[0].buffer.size = fftOutputBuffer_1->getSize();
			pInfos[0].buffer.offset = 0u;

			// Out Buffer 
			pWrites[1].binding = 1;
			pWrites[1].descriptorType = asset::EDT_STORAGE_IMAGE;
			pWrites[1].count = channelCountOverride;
			for (uint32_t i=0u; i<channelCountOverride; i++)
			{
				auto& info = pInfos[1u+i];
				info.desc = kernelNormalizedSpectrums[i];
				//info.image.imageLayout = ;
				info.image.sampler = nullptr;
			}

			driver->updateDescriptorSets(2u, pWrites, 0u, nullptr);
			return dset;
		}();

		auto fftDispatchInfo_Horizontal = FFTClass::buildParameters(paddedKerDim, FFTClass::Direction::X);
		auto fftDispatchInfo_Vertical = FFTClass::buildParameters(paddedKerDim, FFTClass::Direction::Y);

		// Ker Image FFT X
		driver->bindComputePipeline(fftPipeline_ImageInput.get());
		driver->bindDescriptorSets(EPBP_COMPUTE, fftPipelineLayout_ImageInput.get(), 0u, 1u, &fftDescriptorSet_Ker_FFT_X.get(), nullptr);
		FFTClass::pushConstants(driver, fftPipelineLayout_ImageInput.get(), kerDim, paddedKerDim, FFTClass::Direction::X, false, srcNumChannels, FFTClass::PaddingType::FILL_WITH_ZERO);
		FFTClass::dispatchHelper(driver, fftDispatchInfo_Horizontal);

		// Ker Image FFT Y
		auto fftPipeline_SSBOInput = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(fftPipelineLayout_SSBOInput), FFTClass::createShader(driver,FFTClass::DataType::SSBO,paddedKerDim.height));
		driver->bindComputePipeline(fftPipeline_SSBOInput.get());
		driver->bindDescriptorSets(EPBP_COMPUTE, fftPipelineLayout_SSBOInput.get(), 0u, 1u, &fftDescriptorSet_Ker_FFT_Y.get(), nullptr);
		FFTClass::pushConstants(driver, fftPipelineLayout_SSBOInput.get(), paddedKerDim, paddedKerDim, FFTClass::Direction::Y, false, srcNumChannels);
		FFTClass::dispatchHelper(driver, fftDispatchInfo_Vertical);
		
		// Ker Normalization
		auto fftPipeline_KernelNormalization = driver->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(fftPipelineLayout_KernelNormalization),
			[&]() -> auto
			{
				IAssetLoader::SAssetLoadParams lp;
				auto shaderAsset = am->getAsset("../normalization.comp", lp);
				auto stuff = driver->getGPUObjectsFromAssets<asset::ICPUSpecializedShader>(shaderAsset.getContents(),nullptr);
				return *stuff->begin();
			}()
		);
		driver->bindComputePipeline(fftPipeline_KernelNormalization.get());
		driver->bindDescriptorSets(EPBP_COMPUTE, fftPipelineLayout_KernelNormalization.get(), 0u, 1u, &fftDescriptorSet_KernelNormalization.get(), nullptr);
		{
			const uint32_t dispatchSizeX = (paddedKerDim.width-1u)/16u+1u;
			const uint32_t dispatchSizeY = (paddedKerDim.height-1u)/16u+1u;
			driver->dispatch(dispatchSizeX,dispatchSizeY,kerNumChannels);
			FFTClass::defaultBarrier();
		}
	}

	// Src FFT X 
	auto fftDescriptorSet_Src_FFT_X = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(fftPipelineLayout_ImageInput->getDescriptorSetLayout(0u)));
	FFTClass::updateDescriptorSet(driver, fftDescriptorSet_Src_FFT_X.get(), srcImageView, fftOutputBuffer_0, ISampler::ETC_MIRROR);

	// Convolution
	auto convolveDescriptorSet = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(convolvePipelineLayout->getDescriptorSetLayout(0u)));
	updateDescriptorSet_Convolution(driver, convolveDescriptorSet.get(), fftOutputBuffer_0, kernelNormalizedSpectrums);

	// Last IFFTX 
	auto lastFFTDescriptorSet = driver->createGPUDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(lastFFTPipelineLayout->getDescriptorSetLayout(0u)));
	updateDescriptorSet_LastFFT(driver, lastFFTDescriptorSet.get(), fftOutputBuffer_0, outImgView);

	uint32_t outBufferIx = 0u;
	auto lastPresentStamp = std::chrono::high_resolution_clock::now();
	bool savedToFile = false;
	
	auto downloadStagingArea = driver->getDefaultDownStreamingBuffer();
	
	auto blitFBO = driver->addFrameBuffer();
	blitFBO->attach(video::EFAP_COLOR_ATTACHMENT0, std::move(outImgView));


	auto fftDispatchInfo_Horizontal = FFTClass::buildParameters(paddedDim, FFTClass::Direction::X);
	auto fftDispatchInfo_Vertical = FFTClass::buildParameters(paddedDim, FFTClass::Direction::Y);
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
				region.bufferRowLength = 0u;
				region.bufferImageHeight = 0u;
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