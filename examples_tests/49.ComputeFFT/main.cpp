// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>
#include <iostream>
#include <cstdio>

#include "nbl/ext/FFT/FFT.h"
#include "../common/QToQuitEventReceiver.h"

using namespace nbl;
using namespace nbl::core;
using namespace nbl::asset;
using namespace nbl::video;

using FFTClass = ext::FFT::FFT;

constexpr uint32_t channelCountOverride = 3u;

inline core::smart_refctd_ptr<video::IGPUSpecializedShader> createShader(
	video::IVideoDriver* driver,
	const uint32_t maxFFTlen,
	const bool useHalfStorage,
	const char* includeMainName,
	float kernelScale = 1.f)
{
	const char* sourceFmt =
R"===(#version 430 core

#define _NBL_GLSL_WORKGROUP_SIZE_ %u
#define _NBL_GLSL_EXT_FFT_MAX_DIM_SIZE_ %u
#define _NBL_GLSL_EXT_FFT_HALF_STORAGE_ %u

#define KERNEL_SCALE %f
 
#include "%s"

)===";

	const size_t extraSize = 4u+8u+8u+128u;
	
	constexpr uint32_t DEFAULT_WORK_GROUP_SIZE = FFTClass::DEFAULT_WORK_GROUP_SIZE;
	auto shader = core::make_smart_refctd_ptr<ICPUBuffer>(strlen(sourceFmt)+extraSize+1u);
	snprintf(
		reinterpret_cast<char*>(shader->getPointer()),shader->getSize(), sourceFmt,
		DEFAULT_WORK_GROUP_SIZE,
		maxFFTlen,
		useHalfStorage ? 1u:0u,
		kernelScale,
		includeMainName
	);

	auto cpuSpecializedShader = core::make_smart_refctd_ptr<ICPUSpecializedShader>(
		core::make_smart_refctd_ptr<ICPUShader>(std::move(shader),ICPUShader::buffer_contains_glsl),
		ISpecializedShader::SInfo{nullptr, nullptr, "main", asset::ISpecializedShader::ESS_COMPUTE}
	);
	
	auto gpuShader = driver->createShader(nbl::core::smart_refctd_ptr<const ICPUShader>(cpuSpecializedShader->getUnspecialized()));
	
	auto gpuSpecializedShader = driver->createSpecializedShader(gpuShader.get(), cpuSpecializedShader->getSpecializationInfo());

	return gpuSpecializedShader;
}



inline void updateDescriptorSet_Convolution (
	video::IVideoDriver * driver,
	video::IGPUDescriptorSet * set,
	core::smart_refctd_ptr<video::IGPUBuffer> inputBufferDescriptor,
	core::smart_refctd_ptr<video::IGPUBuffer> outputBufferDescriptor,
	const core::smart_refctd_ptr<video::IGPUImageView>* kernelNormalizedSpectrumImageDescriptors)
{
	constexpr uint32_t descCount = 3u;
	video::IGPUDescriptorSet::SDescriptorInfo pInfos[descCount-1u+channelCountOverride];
	video::IGPUDescriptorSet::SWriteDescriptorSet pWrites[descCount];

	for (auto i = 0; i < descCount; i++)
	{
		pWrites[i].binding = i;
		pWrites[i].dstSet = set;
		pWrites[i].arrayElement = 0u;
		pWrites[i].info = pInfos+i;
	}

	// Input Buffer 
	pWrites[0].descriptorType = asset::EDT_STORAGE_BUFFER;
	pWrites[0].count = 1;
	pInfos[0].desc = inputBufferDescriptor;
	pInfos[0].buffer.size = inputBufferDescriptor->getSize();
	pInfos[0].buffer.offset = 0u;
	
	//
	pWrites[1].descriptorType = asset::EDT_STORAGE_BUFFER;
	pWrites[1].count = 1;
	pInfos[1].desc = outputBufferDescriptor;
	pInfos[1].buffer.size = outputBufferDescriptor->getSize();
	pInfos[1].buffer.offset = 0u;

	// Kernel Buffer 
	pWrites[2].descriptorType = asset::EDT_COMBINED_IMAGE_SAMPLER;
	pWrites[2].count = channelCountOverride;
	for (uint32_t i=0u; i<channelCountOverride; i++)
	{
		auto& info = pInfos[2u+i];
		info.desc = kernelNormalizedSpectrumImageDescriptors[i];
		//info.image.imageLayout = ;
		info.image.sampler = nullptr;
	}

	driver->updateDescriptorSets(descCount, pWrites, 0u, nullptr);
}
inline void updateDescriptorSet_LastFFT (
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

using nbl_glsl_ext_FFT_Parameters_t = ext::FFT::FFT::Parameters_t;
struct vec2
{
	float x,y;
};
struct ivec2
{
	int32_t x,y;
};
#include "extra_parameters.glsl"


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
	auto kerImageBundle = am->getAsset("../../media/kernels/physical_flare_256.exr", lp);

	// get GPU image views
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
		srcImageView = driver->createImageView(std::move(srcImgViewInfo));
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
		kerImgViewInfo.subresourceRange.levelCount = kerImgViewInfo.image->getCreationParameters().mipLevels;
		kerImgViewInfo.subresourceRange.baseArrayLayer = 0;
		kerImgViewInfo.subresourceRange.layerCount = 1;
		kerImageView = driver->createImageView(std::move(kerImgViewInfo));
	}

	// agree on formats
	const E_FORMAT srcFormat = srcImageView->getCreationParameters().format;
	uint32_t srcNumChannels = getFormatChannelCount(srcFormat);
	uint32_t kerNumChannels = getFormatChannelCount(kerImageView->getCreationParameters().format);
	//! OVERRIDE (we dont need alpha)
	srcNumChannels = channelCountOverride;
	kerNumChannels = channelCountOverride;
	assert(srcNumChannels == kerNumChannels); // Just to make sure, because the other case is not handled in this example

	// Create Out Image
	smart_refctd_ptr<IGPUImage> outImg;
	smart_refctd_ptr<IGPUImageView> outImgView;
	{
		auto dstImgViewInfo = srcImageView->getCreationParameters();

		auto dstImgInfo = dstImgViewInfo.image->getCreationParameters();
		outImg = driver->createDeviceLocalGPUImageOnDedMem(std::move(dstImgInfo));

		dstImgViewInfo.image = outImg;
		outImgView = driver->createImageView(IGPUImageView::SCreationParams(dstImgViewInfo));
	}

	// input pipeline
	auto imageFirstFFTPipelineLayout = [driver]() -> auto
	{
		IGPUDescriptorSetLayout::SBinding bnd[] =
		{
			{
				0u,
				EDT_COMBINED_IMAGE_SAMPLER,
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
			}
		};
	
		core::SRange<const asset::SPushConstantRange> pcRange = FFTClass::getDefaultPushConstantRanges();
		core::SRange<const video::IGPUDescriptorSetLayout::SBinding> bindings = {bnd,bnd+sizeof(bnd)/sizeof(IGPUDescriptorSetLayout::SBinding)};

		return driver->createPipelineLayout(
			pcRange.begin(),pcRange.end(),
			driver->createDescriptorSetLayout(bindings.begin(),bindings.end()),nullptr,nullptr,nullptr
		);
	}();
	auto convolvePipelineLayout = [driver]() -> auto
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
		auto sampler = driver->createSampler(std::move(params));
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
				EDT_STORAGE_BUFFER,
				1u,
				ISpecializedShader::ESS_COMPUTE,
				nullptr
			},
			{
				2u,
				EDT_COMBINED_IMAGE_SAMPLER,
				channelCountOverride,
				ISpecializedShader::ESS_COMPUTE,
				samplers
			}
		};
	
		const asset::SPushConstantRange pcRange = {ISpecializedShader::ESS_COMPUTE,0u,sizeof(convolve_parameters_t)};
		core::SRange<const video::IGPUDescriptorSetLayout::SBinding> bindings = {bnd,bnd+sizeof(bnd)/sizeof(IGPUDescriptorSetLayout::SBinding)};

		return driver->createPipelineLayout(
			&pcRange,&pcRange+1,
			driver->createDescriptorSetLayout(bindings.begin(),bindings.end()),nullptr,nullptr,nullptr
		);
	}();
	auto lastFFTPipelineLayout = [driver]() -> auto
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
				1u,
				ISpecializedShader::ESS_COMPUTE,
				nullptr
			},
		};
		
		const asset::SPushConstantRange pcRange = {ISpecializedShader::ESS_COMPUTE,0u,sizeof(image_store_parameters_t)};
		core::SRange<const video::IGPUDescriptorSetLayout::SBinding> bindings = {bnd, bnd+sizeof(bnd)/sizeof(IGPUDescriptorSetLayout::SBinding)};;

		return driver->createPipelineLayout(
			&pcRange,&pcRange+1,
			driver->createDescriptorSetLayout(bindings.begin(),bindings.end()),nullptr,nullptr,nullptr
		);
	}();

	const float bloomRelativeScale = 0.25f;
	const auto kerDim = kerImageView->getCreationParameters().image->getCreationParameters().extent;
	const auto srcDim = srcImageView->getCreationParameters().image->getCreationParameters().extent;
	const float bloomScale = core::min(float(srcDim.width)/float(kerDim.width),float(srcDim.height)/float(kerDim.height))*bloomRelativeScale;
	if (bloomScale>1.f)
		std::cout << "WARNING: Bloom Kernel will Clip and loose sharpness, increase resolution of bloom kernel!" << std::endl;
	const auto marginSrcDim = [srcDim,kerDim,bloomScale]() -> auto
	{
		auto tmp = srcDim;
		for (auto i=0u; i<3u; i++)
		{
			const auto coord = (&kerDim.width)[i];
			if (coord>1u)
				(&tmp.width)[i] += core::max(coord*bloomScale,1u)-1u;
		}
		return tmp;
	}();
	constexpr bool useHalfFloats = true;
	// Allocate Output Buffer
	auto fftOutputBuffer_0 = driver->createDeviceLocalGPUBufferOnDedMem(FFTClass::getOutputBufferSize(useHalfFloats,marginSrcDim,srcNumChannels));
	auto fftOutputBuffer_1 = driver->createDeviceLocalGPUBufferOnDedMem(FFTClass::getOutputBufferSize(useHalfFloats,marginSrcDim,srcNumChannels));
	core::smart_refctd_ptr<IGPUImageView> kernelNormalizedSpectrums[channelCountOverride];

	auto updateDescriptorSet = [driver](video::IGPUDescriptorSet* set, core::smart_refctd_ptr<IGPUImageView> inputImageDescriptor, asset::ISampler::E_TEXTURE_CLAMP textureWrap, core::smart_refctd_ptr<IGPUBuffer> outputBufferDescriptor) -> void
	{
		IGPUSampler::SParams params =
		{
			{
				textureWrap,
				textureWrap,
				textureWrap,
				ISampler::ETBC_FLOAT_OPAQUE_BLACK,
				ISampler::ETF_LINEAR,
				ISampler::ETF_LINEAR,
				ISampler::ESMM_LINEAR,
				8u,
				0u,
				ISampler::ECO_ALWAYS
			}
		};
		auto sampler = driver->createSampler(std::move(params));
		
		constexpr auto kDescriptorCount = 2u;
		video::IGPUDescriptorSet::SDescriptorInfo pInfos[kDescriptorCount];
		video::IGPUDescriptorSet::SWriteDescriptorSet pWrites[kDescriptorCount];

		for (auto i=0; i<kDescriptorCount; i++)
		{
			pWrites[i].dstSet = set;
			pWrites[i].arrayElement = 0u;
			pWrites[i].count = 1u;
			pWrites[i].info = pInfos+i;
		}

		// Input Buffer 
		pWrites[0].binding = 0;
		pWrites[0].descriptorType = asset::EDT_COMBINED_IMAGE_SAMPLER;
		pWrites[0].count = 1;
		pInfos[0].desc = inputImageDescriptor;
		pInfos[0].image.sampler = sampler;
		pInfos[0].image.imageLayout = static_cast<asset::E_IMAGE_LAYOUT>(0u);

		// Output Buffer 
		pWrites[1].binding = 1;
		pWrites[1].descriptorType = asset::EDT_STORAGE_BUFFER;
		pWrites[1].count = 1;
		pInfos[1].desc = outputBufferDescriptor;
		pInfos[1].buffer.size = outputBufferDescriptor->getSize();
		pInfos[1].buffer.offset = 0u;

		driver->updateDescriptorSets(2u, pWrites, 0u, nullptr);
	};

	// Precompute Kernel FFT
	{
		const VkExtent3D paddedKerDim = FFTClass::padDimensions(kerDim);

		// create kernel spectrums
		auto createKernelSpectrum = [&]() -> auto
		{
			video::IGPUImage::SCreationParams imageParams;
			imageParams.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
			imageParams.type = asset::IImage::ET_2D;
			imageParams.format = useHalfFloats ? EF_R16G16_SFLOAT:EF_R32G32_SFLOAT;
			imageParams.extent = { paddedKerDim.width,paddedKerDim.height,1u};
			imageParams.mipLevels = 1u;
			imageParams.arrayLayers = 1u;
			imageParams.samples = asset::IImage::ESCF_1_BIT;

			video::IGPUImageView::SCreationParams viewParams;
			viewParams.flags = static_cast<video::IGPUImageView::E_CREATE_FLAGS>(0u);
			viewParams.image = driver->createGPUImageOnDedMem(std::move(imageParams),driver->getDeviceLocalGPUMemoryReqs());
			viewParams.viewType = video::IGPUImageView::ET_2D;
			viewParams.format = useHalfFloats ? EF_R16G16_SFLOAT:EF_R32G32_SFLOAT;
			viewParams.components = {};
			viewParams.subresourceRange = {};
			viewParams.subresourceRange.levelCount = 1u;
			viewParams.subresourceRange.layerCount = 1u;
			return driver->createImageView(std::move(viewParams));
		};
		for (uint32_t i=0u; i<channelCountOverride; i++)
			kernelNormalizedSpectrums[i] = createKernelSpectrum();
		

		FFTClass::Parameters_t fftPushConstants[2];
		FFTClass::DispatchInfo_t fftDispatchInfo[2];
		const ISampler::E_TEXTURE_CLAMP fftPadding[2] = {ISampler::ETC_CLAMP_TO_BORDER,ISampler::ETC_CLAMP_TO_BORDER};
		const auto passes = FFTClass::buildParameters(false,srcNumChannels,kerDim,fftPushConstants,fftDispatchInfo,fftPadding);
		assert(passes==2u);
		// last FFT pipeline
		core::smart_refctd_ptr<IGPUComputePipeline> fftPipeline_SSBOInput(core::make_smart_refctd_ptr<FFTClass>(driver,0x1u<<fftPushConstants[1].getLog2FFTSize(),useHalfFloats)->getDefaultPipeline());

		// descriptor sets
		core::smart_refctd_ptr<IGPUDescriptorSet> fftDescriptorSet_Ker_FFT[2] =
		{
			driver->createDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(imageFirstFFTPipelineLayout->getDescriptorSetLayout(0u))),
			driver->createDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(fftPipeline_SSBOInput->getLayout()->getDescriptorSetLayout(0u)))
		};
		updateDescriptorSet(fftDescriptorSet_Ker_FFT[0].get(), kerImageView, ISampler::ETC_CLAMP_TO_BORDER, fftOutputBuffer_0);
		FFTClass::updateDescriptorSet(driver,fftDescriptorSet_Ker_FFT[1].get(), fftOutputBuffer_0, fftOutputBuffer_1);
		
		// Normalization of FFT spectrum
		struct NormalizationPushConstants
		{
			ext::FFT::uvec4 stride;
			ext::FFT::uvec4 bitreverse_shift;
		};
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
			SPushConstantRange pc_rng;
			pc_rng.offset = 0u;
			pc_rng.size = sizeof(NormalizationPushConstants);
			pc_rng.stageFlags = ISpecializedShader::ESS_COMPUTE;
			return driver->createPipelineLayout(
				&pc_rng,&pc_rng+1u,
				driver->createDescriptorSetLayout(bnd,bnd+2),nullptr,nullptr,nullptr
			);
		}();
		auto fftDescriptorSet_KernelNormalization = [&]() -> auto
		{
			auto dset = driver->createDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(fftPipelineLayout_KernelNormalization->getDescriptorSetLayout(0u)));

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

		// Ker Image First Axis FFT
		{
			auto fftPipeline_ImageInput = driver->createComputePipeline(nullptr,core::smart_refctd_ptr(imageFirstFFTPipelineLayout),createShader(driver,0x1u<<fftPushConstants[0].getLog2FFTSize(),useHalfFloats,"../image_first_fft.comp",bloomScale));
			driver->bindComputePipeline(fftPipeline_ImageInput.get());
			driver->bindDescriptorSets(EPBP_COMPUTE, imageFirstFFTPipelineLayout.get(), 0u, 1u, &fftDescriptorSet_Ker_FFT[0].get(), nullptr);
			FFTClass::dispatchHelper(driver, imageFirstFFTPipelineLayout.get(), fftPushConstants[0], fftDispatchInfo[0]);
		}

		// Ker Image Last Axis FFT
		driver->bindComputePipeline(fftPipeline_SSBOInput.get());
		driver->bindDescriptorSets(EPBP_COMPUTE, fftPipeline_SSBOInput->getLayout(), 0u, 1u, &fftDescriptorSet_Ker_FFT[1].get(), nullptr);
		FFTClass::dispatchHelper(driver, fftPipeline_SSBOInput->getLayout(), fftPushConstants[1], fftDispatchInfo[1]);
		
		// Ker Normalization
		auto fftPipeline_KernelNormalization = driver->createComputePipeline(nullptr,core::smart_refctd_ptr(fftPipelineLayout_KernelNormalization),createShader(driver,0xdeadbeefu,useHalfFloats,"../normalization.comp"));
		driver->bindComputePipeline(fftPipeline_KernelNormalization.get());
		driver->bindDescriptorSets(EPBP_COMPUTE, fftPipelineLayout_KernelNormalization.get(), 0u, 1u, &fftDescriptorSet_KernelNormalization.get(), nullptr);
		{
			NormalizationPushConstants normalizationPC;
			normalizationPC.stride = fftPushConstants[1].output_strides;
			normalizationPC.bitreverse_shift.x = 32-core::findMSB(paddedKerDim.width);
			normalizationPC.bitreverse_shift.y = 32-core::findMSB(paddedKerDim.height);
			normalizationPC.bitreverse_shift.z = 0;
			driver->pushConstants(fftPipelineLayout_KernelNormalization.get(),ICPUSpecializedShader::ESS_COMPUTE,0u,sizeof(normalizationPC),&normalizationPC);
		}
		{
			const uint32_t dispatchSizeX = (paddedKerDim.width-1u)/16u+1u;
			const uint32_t dispatchSizeY = (paddedKerDim.height-1u)/16u+1u;
			driver->dispatch(dispatchSizeX,dispatchSizeY,kerNumChannels);
			FFTClass::defaultBarrier();
		}
	}
	
	FFTClass::Parameters_t fftPushConstants[3];
	FFTClass::DispatchInfo_t fftDispatchInfo[3];
	const ISampler::E_TEXTURE_CLAMP fftPadding[2] = {ISampler::ETC_MIRROR,ISampler::ETC_MIRROR};
	const auto passes = FFTClass::buildParameters(false,srcNumChannels,srcDim,fftPushConstants,fftDispatchInfo,fftPadding,marginSrcDim);
	{
		// override for less work and storage (dont need to store the extra padding of the last axis after iFFT)
		fftPushConstants[1].output_strides.x = fftPushConstants[0].input_strides.x;
		fftPushConstants[1].output_strides.y = fftPushConstants[0].input_strides.y;
		fftPushConstants[1].output_strides.z = fftPushConstants[1].input_strides.z;
		fftPushConstants[1].output_strides.w = fftPushConstants[1].input_strides.w;
		// iFFT
		fftPushConstants[2].input_dimensions = fftPushConstants[1].input_dimensions;
		{
			fftPushConstants[2].input_dimensions.w = fftPushConstants[0].input_dimensions.w^0x80000000u;
			fftPushConstants[2].input_strides = fftPushConstants[1].output_strides;
			fftPushConstants[2].output_strides = fftPushConstants[0].input_strides;
		}
		fftDispatchInfo[2] = fftDispatchInfo[0];
	}
	assert(passes==2);
	// pipelines
	auto fftPipeline_ImageInput = driver->createComputePipeline(nullptr,core::smart_refctd_ptr(imageFirstFFTPipelineLayout),createShader(driver,0x1u<<fftPushConstants[0].getLog2FFTSize(),useHalfFloats,"../image_first_fft.comp"));
	auto convolvePipeline = driver->createComputePipeline(nullptr, std::move(convolvePipelineLayout), createShader(driver,0x1u<<fftPushConstants[1].getLog2FFTSize(),useHalfFloats, "../fft_convolve_ifft.comp"));
	auto lastFFTPipeline = driver->createComputePipeline(nullptr, std::move(lastFFTPipelineLayout), createShader(driver,0x1u<<fftPushConstants[0].getLog2FFTSize(),useHalfFloats,"../last_fft.comp"));

	// Src First Axis FFT
	auto fftDescriptorSet_Src_FirstFFT = driver->createDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(imageFirstFFTPipelineLayout->getDescriptorSetLayout(0u)));
	updateDescriptorSet(fftDescriptorSet_Src_FirstFFT.get(), srcImageView, ISampler::ETC_MIRROR, fftOutputBuffer_0);

	// Convolution
	auto convolveDescriptorSet = driver->createDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(convolvePipeline->getLayout()->getDescriptorSetLayout(0u)));
	updateDescriptorSet_Convolution(driver, convolveDescriptorSet.get(), fftOutputBuffer_0, fftOutputBuffer_1, kernelNormalizedSpectrums);

	// Last Axis IFFT
	auto lastFFTDescriptorSet = driver->createDescriptorSet(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(lastFFTPipeline->getLayout()->getDescriptorSetLayout(0u)));
	updateDescriptorSet_LastFFT(driver, lastFFTDescriptorSet.get(), fftOutputBuffer_1, outImgView);

	uint32_t outBufferIx = 0u;
	auto lastPresentStamp = std::chrono::high_resolution_clock::now();
	bool savedToFile = false;
	
	auto downloadStagingArea = driver->getDefaultDownStreamingBuffer();
	
	auto blitFBO = driver->addFrameBuffer();
	blitFBO->attach(video::EFAP_COLOR_ATTACHMENT0, std::move(outImgView));

	while (device->run() && receiver.keepOpen())
	{
		driver->beginScene(false, false);

		// Src Image First Axis FFT
		driver->bindComputePipeline(fftPipeline_ImageInput.get());
		driver->bindDescriptorSets(EPBP_COMPUTE, imageFirstFFTPipelineLayout.get(), 0u, 1u, &fftDescriptorSet_Src_FirstFFT.get(), nullptr);
		FFTClass::dispatchHelper(driver, imageFirstFFTPipelineLayout.get(), fftPushConstants[0], fftDispatchInfo[0]);

		// Src Image Last Axis FFT + Convolution + Convolved Last Axis IFFT Y
		driver->bindComputePipeline(convolvePipeline.get());
		driver->bindDescriptorSets(EPBP_COMPUTE, convolvePipeline->getLayout(), 0u, 1u, &convolveDescriptorSet.get(), nullptr);
		{
			const auto& kernelImgExtent = kernelNormalizedSpectrums[0]->getCreationParameters().image->getCreationParameters().extent;
			vec2 kernel_half_pixel_size{0.5f,0.5f};
			kernel_half_pixel_size.x /= kernelImgExtent.width;
			kernel_half_pixel_size.y /= kernelImgExtent.height;
			driver->pushConstants(convolvePipeline->getLayout(),ISpecializedShader::ESS_COMPUTE,offsetof(convolve_parameters_t,kernel_half_pixel_size),sizeof(convolve_parameters_t::kernel_half_pixel_size),&kernel_half_pixel_size);
		}
		FFTClass::dispatchHelper(driver, convolvePipeline->getLayout(), fftPushConstants[1], fftDispatchInfo[1]);

		// Last FFT Padding and Copy to GPU Image
		driver->bindComputePipeline(lastFFTPipeline.get());
		driver->bindDescriptorSets(EPBP_COMPUTE, lastFFTPipeline->getLayout(), 0u, 1u, &lastFFTDescriptorSet.get(), nullptr);
		{
			const auto paddedSrcDim = FFTClass::padDimensions(marginSrcDim);
			ivec2 unpad_offset = { 0,0 };
			for (auto i=0u; i<2u; i++)
			if (fftDispatchInfo[2].workGroupCount[i]==1u)
				(&unpad_offset.x)[i] = ((&paddedSrcDim.width)[i]-(&srcDim.width)[i])>>1u;
			driver->pushConstants(lastFFTPipeline->getLayout(),ISpecializedShader::ESS_COMPUTE,offsetof(image_store_parameters_t,unpad_offset),sizeof(image_store_parameters_t::unpad_offset),&unpad_offset);
		}
		FFTClass::dispatchHelper(driver, lastFFTPipeline->getLayout(), fftPushConstants[2], fftDispatchInfo[2]);
		
		if(!savedToFile) 
		{
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