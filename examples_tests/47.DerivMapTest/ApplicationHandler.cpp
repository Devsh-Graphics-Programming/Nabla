// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "ApplicationHandler.hpp"

#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"
#include "nbl/ext/ScreenShot/ScreenShot.h"

#include <nbl/asset/filters/kernels/CGaussianImageFilterKernel.h>
#include <nbl/asset/filters/kernels/CDerivativeImageFilterKernel.h>
#include <nbl/asset/filters/kernels/CBoxImageFilterKernel.h>
#include <nbl/asset/filters/kernels/CChannelIndependentImageFilterKernel.h>
#include <nbl/asset/filters/CMipMapGenerationImageFilter.h>

using namespace nbl;
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

void ApplicationHandler::presentImageOnTheScreen(nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> gpuImageView, std::string currentHandledImageFileName, std::string currentHandledImageExtension)
{
	auto samplerDescriptorSet3 = driver->createGPUDescriptorSet(core::smart_refctd_ptr(gpuDescriptorSetLayout3));

	IGPUDescriptorSet::SDescriptorInfo info;
	{
		info.desc = gpuImageView;
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

		driver->bindGraphicsPipeline(currentGpuPipelineFor2D.get());
		driver->bindDescriptorSets(EPBP_GRAPHICS, currentGpuPipelineFor2D->getLayout(), 3u, 1u, &samplerDescriptorSet3.get(), nullptr);
		driver->drawMeshBuffer(currentGpuMeshBuffer.get());

		driver->blitRenderTargets(nullptr, screenShotFrameBuffer, false, false);
		driver->endScene();
	}

	ext::ScreenShot::createScreenShot(device, screenShotFrameBuffer->getAttachment(video::EFAP_COLOR_ATTACHMENT0), "screenShot_" + currentHandledImageFileName + ".png");
}

namespace
{
template<class Kernel>
class MyKernel : public asset::CFloatingPointSeparableImageFilterKernelBase<MyKernel<Kernel>>
{
		using Base = asset::CFloatingPointSeparableImageFilterKernelBase<MyKernel<Kernel>>;

		Kernel kernel;
		float multiplier;

	public:
		using value_type = typename Base::value_type;

		MyKernel(Kernel&& k, float _imgExtent) : Base(k.negative_support.x, k.positive_support.x), kernel(std::move(k)), multiplier(_imgExtent) {}

		// no special user data by default
		inline const asset::IImageFilterKernel::UserData* getUserData() const { return nullptr; }

		inline float weight(float x, int32_t channel) const
		{
			return kernel.weight(x, channel) * multiplier;
		}
		
		// we need to ensure to override the default behaviour of `CFloatingPointSeparableImageFilterKernelBase` which applies the weight along every axis
		template<class PreFilter, class PostFilter>
		struct sample_functor_t
		{
				sample_functor_t(const MyKernel* _this, PreFilter& _preFilter, PostFilter& _postFilter) :
					_this(_this), preFilter(_preFilter), postFilter(_postFilter) {}

				inline void operator()(value_type* windowSample, core::vectorSIMDf& relativePos, const core::vectorSIMDi32& globalTexelCoord, const IImageFilterKernel::UserData* userData)
				{
					preFilter(windowSample, relativePos, globalTexelCoord, userData);
					auto* scale = IImageFilterKernel::ScaleFactorUserData::cast(userData);
					for (int32_t i=0; i<MaxChannels; i++)
					{
						// this differs from the `CFloatingPointSeparableImageFilterKernelBase`
						windowSample[i] *= _this->weight(relativePos.x, i);
						if (scale)
							windowSample[i] *= scale->factor[i];
					}
					postFilter(windowSample, relativePos, globalTexelCoord, userData);
				}

			private:
				const MyKernel* _this;
				PreFilter& preFilter;
				PostFilter& postFilter;
		};

		_NBL_STATIC_INLINE_CONSTEXPR bool has_derivative = false;

		NBL_DECLARE_DEFINE_CIMAGEFILTER_KERNEL_PASS_THROUGHS(Base)
};
	
template<class Kernel>
class SeparateOutXAxisKernel : public asset::CFloatingPointSeparableImageFilterKernelBase<SeparateOutXAxisKernel<Kernel>>
{
		using Base = asset::CFloatingPointSeparableImageFilterKernelBase<SeparateOutXAxisKernel<Kernel>>;

		Kernel kernel;

	public:
		// passthrough everything
		using value_type = typename Kernel::value_type;

		_NBL_STATIC_INLINE_CONSTEXPR auto MaxChannels = Kernel::MaxChannels; // derivative map only needs 2 channels

		SeparateOutXAxisKernel(Kernel&& k) : Base(k.negative_support.x, k.positive_support.x), kernel(std::move(k)) {}

		NBL_DECLARE_DEFINE_CIMAGEFILTER_KERNEL_PASS_THROUGHS(Base)
					
		// we need to ensure to override the default behaviour of `CFloatingPointSeparableImageFilterKernelBase` which applies the weight along every axis
		template<class PreFilter, class PostFilter>
		struct sample_functor_t
		{
				sample_functor_t(const SeparateOutXAxisKernel<Kernel>* _this, PreFilter& _preFilter, PostFilter& _postFilter) :
					_this(_this), preFilter(_preFilter), postFilter(_postFilter) {}

				inline void operator()(value_type* windowSample, core::vectorSIMDf& relativePos, const core::vectorSIMDi32& globalTexelCoord, const IImageFilterKernel::UserData* userData)
				{
					preFilter(windowSample, relativePos, globalTexelCoord, userData);
					auto* scale = IImageFilterKernel::ScaleFactorUserData::cast(userData);
					for (int32_t i=0; i<MaxChannels; i++)
					{
						// this differs from the `CFloatingPointSeparableImageFilterKernelBase`
						windowSample[i] *= _this->kernel.weight(relativePos.x, i);
						if (scale)
							windowSample[i] *= scale->factor[i];
					}
					postFilter(windowSample, relativePos, globalTexelCoord, userData);
				}

			private:
				const SeparateOutXAxisKernel<Kernel>* _this;
				PreFilter& preFilter;
				PostFilter& postFilter;
		};

		// the method all kernels must define and overload
		template<class PreFilter, class PostFilter>
		inline auto create_sample_functor_t(PreFilter& preFilter, PostFilter& postFilter) const
		{
			return sample_functor_t(this,preFilter,postFilter);
		}
};

}
static core::smart_refctd_ptr<asset::ICPUImage> createDerivMapFromHeightMap(asset::ICPUImage* _inImg, asset::ISampler::E_TEXTURE_CLAMP _uwrap, asset::ISampler::E_TEXTURE_CLAMP _vwrap, asset::ISampler::E_TEXTURE_BORDER_COLOR _borderColor)
{
	using namespace asset;

#define DERIV_MAP_FLOAT32
	auto getRGformat = [](asset::E_FORMAT f) -> asset::E_FORMAT {
		const uint32_t bytesPerChannel = (getBytesPerPixel(f) * core::rational(1, getFormatChannelCount(f))).getIntegerApprox();
		switch (bytesPerChannel)
		{
		case 1u:
#ifndef DERIV_MAP_FLOAT32
			return asset::EF_R8G8_UNORM;
#else
			[[fallthrough]];
#endif
		case 2u:
#ifndef DERIV_MAP_FLOAT32
			return asset::EF_R16G16_SFLOAT;
#else
			[[fallthrough]];
#endif
		case 4u:
			return asset::EF_R32G32_SFLOAT;
		case 8u:
			return asset::EF_R64G64_SFLOAT;
		default:
			return asset::EF_UNKNOWN;
		}
	};

	using ReconstructionKernel = CGaussianImageFilterKernel<>; // or Mitchell
	using DerivKernel_ = CDerivativeImageFilterKernel<ReconstructionKernel>;
	using DerivKernel = MyKernel<DerivKernel_>;
	using XDerivKernel_ = CChannelIndependentImageFilterKernel<DerivKernel, CBoxImageFilterKernel>;
	using YDerivKernel_ = CChannelIndependentImageFilterKernel<CBoxImageFilterKernel, DerivKernel>;
	using XDerivKernel = SeparateOutXAxisKernel<XDerivKernel_>;
	using YDerivKernel = SeparateOutXAxisKernel<YDerivKernel_>;
	using DerivativeMapFilter = CBlitImageFilter
		<
		false, false, DefaultSwizzle, IdentityDither, // (Criss, look at impl::CSwizzleAndConvertImageFilterBase)
		XDerivKernel,
		YDerivKernel,
		CBoxImageFilterKernel
		>;

	const auto extent = _inImg->getCreationParameters().extent;
	const float mlt = static_cast<float>(std::max(extent.width, extent.height));
	XDerivKernel xderiv(XDerivKernel_( DerivKernel(DerivKernel_(ReconstructionKernel()), mlt), CBoxImageFilterKernel() ));
	YDerivKernel yderiv(YDerivKernel_( CBoxImageFilterKernel(), DerivKernel(DerivKernel_(ReconstructionKernel()), mlt) ));

	using swizzle_t = asset::ICPUImageView::SComponentMapping;
	DerivativeMapFilter::state_type state(std::move(xderiv), std::move(yderiv), CBoxImageFilterKernel());

	state.swizzle = { swizzle_t::ES_R, swizzle_t::ES_R, swizzle_t::ES_R, swizzle_t::ES_R };

	const auto& inParams = _inImg->getCreationParameters();
	auto outParams = inParams;
	outParams.format = getRGformat(outParams.format);
	const uint32_t pitch = IImageAssetHandlerBase::calcPitchInBlocks(outParams.extent.width, asset::getTexelOrBlockBytesize(outParams.format));
	auto buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(asset::getTexelOrBlockBytesize(outParams.format) * pitch * outParams.extent.height);
	asset::ICPUImage::SBufferCopy region;
	region.imageOffset = { 0,0,0 };
	region.imageExtent = outParams.extent;
	region.imageSubresource.baseArrayLayer = 0u;
	region.imageSubresource.layerCount = 1u;
	region.imageSubresource.mipLevel = 0u;
	region.bufferRowLength = pitch;
	region.bufferImageHeight = 0u;
	region.bufferOffset = 0u;
	auto outImg = asset::ICPUImage::create(std::move(outParams));
	outImg->setBufferAndRegions(std::move(buffer), core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<IImage::SBufferCopy>>(1ull, region));

	state.inOffset = { 0,0,0 };
	state.inBaseLayer = 0u;
	state.outOffset = { 0,0,0 };
	state.outBaseLayer = 0u;
	state.inExtent = inParams.extent;
	state.outExtent = state.inExtent;
	state.inLayerCount = 1u;
	state.outLayerCount = 1u;
	state.inMipLevel = 0u;
	state.outMipLevel = 0u;
	state.inImage = _inImg;
	state.outImage = outImg.get();
	state.axisWraps[0] = _uwrap;
	state.axisWraps[1] = _vwrap;
	state.axisWraps[2] = asset::ISampler::ETC_CLAMP_TO_EDGE;
	state.borderColor = _borderColor;
	state.scratchMemoryByteSize = DerivativeMapFilter::getRequiredScratchByteSize(&state);
	state.scratchMemory = reinterpret_cast<uint8_t*>(_NBL_ALIGNED_MALLOC(state.scratchMemoryByteSize, _NBL_SIMD_ALIGNMENT));

	DerivativeMapFilter::execute(std::execution::par_unseq,&state);

	_NBL_ALIGNED_FREE(state.scratchMemory);

	return outImg;
}
void ApplicationHandler::performImageTest(std::string path)
{
	os::Printer::log("Reading", path);

	auto assetManager = device->getAssetManager();

	smart_refctd_ptr<ICPUImageView> cpuImageView;

	IAssetLoader::SAssetLoadParams lp(0ull,nullptr,IAssetLoader::ECF_DONT_CACHE_REFERENCES);
	auto cpuTexture = assetManager->getAsset(path, lp);
	auto cpuTextureContents = cpuTexture.getContents();
	
	if (cpuTextureContents.begin() == cpuTextureContents.end())
	{
		os::Printer::log("CANNOT PERFORM THE TEST - SKIPPING. LOADING WENT WRONG", ELL_WARNING);
		return;
	}

	io::path filename, extension, finalFileNameWithExtension;
	core::splitFilename(path.c_str(), nullptr, &filename, &extension);
	finalFileNameWithExtension = filename + ".";
	finalFileNameWithExtension += extension;

	smart_refctd_ptr<ICPUImageView> copyImageView;

	auto asset = *cpuTextureContents.begin();
	assert(asset->getAssetType()==IAsset::ET_IMAGE);
	auto cpuimage = core::smart_refctd_ptr_static_cast<ICPUImage>(std::move(asset));
	cpuimage = createDerivMapFromHeightMap(cpuimage.get(), ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK);
	core::smart_refctd_ptr<video::IGPUImageView> gpuImageView;
	{
		ICPUImageView::SCreationParams viewParams;
		viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
		viewParams.image = cpuimage;
		viewParams.format = viewParams.image->getCreationParameters().format;
		viewParams.viewType = IImageView<ICPUImage>::ET_2D;
		viewParams.subresourceRange.baseArrayLayer = 0u;
		viewParams.subresourceRange.layerCount = 1u;
		viewParams.subresourceRange.baseMipLevel = 0u;
		viewParams.subresourceRange.levelCount = 1u;

		cpuImageView = ICPUImageView::create(std::move(viewParams));
	}

	copyImageView = core::smart_refctd_ptr_static_cast<ICPUImageView>(cpuImageView->clone());
	gpuImageView = driver->getGPUObjectsFromAssets(&cpuImageView.get(), &cpuImageView.get() + 1u)->front();

	if (gpuImageView)
		presentImageOnTheScreen(gpuImageView, std::string(filename.c_str()), std::string(extension.c_str()));

	auto tryToWrite = [&](asset::IAsset* asset)
	{
		asset::IAssetWriter::SAssetWriteParams wparams(asset);
		return assetManager->writeAsset((io::path("imageAsset_") + finalFileNameWithExtension).c_str(), wparams);
	};

	if(!tryToWrite(copyImageView->getCreationParameters().image.get()))
		if(!tryToWrite(copyImageView.get()))
			os::Printer::log("An unexcepted error occoured while trying to write the asset!", nbl::ELL_WARNING);

	assetManager->removeCachedGPUObject(cpuImageView.get(), gpuImageView);
	assetManager->removeAssetFromCache(cpuTexture);
}


bool ApplicationHandler::initializeApplication()
{
	nbl::SIrrlichtCreationParameters params;
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

	auto createGPUPipeline = [&](IImageView<ICPUImage>::E_TYPE type) -> gpuPipeline
	{
		auto getPathToFragmentShader = [&]()
		{
			switch (type)
			{
				case IImageView<ICPUImage>::E_TYPE::ET_2D:
					return "../present2D.frag";
				case IImageView<ICPUImage>::E_TYPE::ET_2D_ARRAY:
					return "../present2DArray.frag";
				case IImageView<ICPUImage>::E_TYPE::ET_CUBE_MAP:
					return "../presentCubemap.frag";
				default:
				{
					os::Printer::log("Not supported image view in the example!", ELL_ERROR);
					return "";
				}
			}
		};

		IAssetLoader::SAssetLoadParams lp;
		auto fs_bundle = device->getAssetManager()->getAsset(getPathToFragmentShader(), lp);
		auto fs_contents = fs_bundle.getContents();
		if (fs_contents.begin() == fs_contents.end())
			return false;

		ICPUSpecializedShader* fs = static_cast<ICPUSpecializedShader*>(fs_contents.begin()->get());

		auto fragShader = driver->getGPUObjectsFromAssets(&fs, &fs + 1)->front();
		if (!fragShader)
			return {};

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

		auto gpuPipelineLayout = driver->createGPUPipelineLayout(nullptr, nullptr, nullptr, nullptr, nullptr, core::smart_refctd_ptr(gpuDescriptorSetLayout3));

		return driver->createGPURenderpassIndependentPipeline(nullptr, std::move(gpuPipelineLayout), shaders, shaders + sizeof(shaders) / sizeof(IGPUSpecializedShader*),
			std::get<SVertexInputParams>(fullScreenTriangle), blendParams,
			std::get<SPrimitiveAssemblyParams>(fullScreenTriangle), rasterParams);
	};

	currentGpuPipelineFor2D = createGPUPipeline(IImageView<ICPUImage>::E_TYPE::ET_2D);

	{
		SBufferBinding<IGPUBuffer> idxBinding{ 0ull, nullptr };
		currentGpuMeshBuffer = core::make_smart_refctd_ptr<IGPUMeshBuffer>(nullptr, nullptr, nullptr, std::move(idxBinding));
		currentGpuMeshBuffer->setIndexCount(3u);
		currentGpuMeshBuffer->setInstanceCount(1u);
	}

	return true;
}