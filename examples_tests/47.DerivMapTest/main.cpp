// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nabla.h>

#include <nbl/asset/filters/kernels/CGaussianImageFilterKernel.h>
#include <nbl/asset/filters/kernels/CDerivativeImageFilterKernel.h>
#include <nbl/asset/filters/kernels/CBoxImageFilterKernel.h>
#include <nbl/asset/filters/kernels/CChannelIndependentImageFilterKernel.h>
#include <nbl/asset/filters/CMipMapGenerationImageFilter.h>

#include "../common/CommonAPI.h"
#include "nbl/ext/ScreenShot/ScreenShot.h"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"

using namespace nbl;
using namespace core;
using namespace asset;
using namespace video;

#define SWITCH_IMAGES_PER_X_MILISECONDS 500
constexpr std::string_view testingImagePathsFile = "../imagesTestList.txt";

struct NBL_CAPTION_DATA_TO_DISPLAY
{
	std::string viewType;
	std::string name;
	std::string extension;
};

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
			for (int32_t i = 0; i < Kernel::MaxChannels; i++)
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

	//_NBL_STATIC_INLINE_CONSTEXPR auto MaxChannels = Kernel::MaxChannels; // derivative map only needs 2 channels

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
			for (int32_t i = 0; i < Kernel::MaxChannels; i++)
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
		return sample_functor_t(this, preFilter, postFilter);
	}
};

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
		false, false, DefaultSwizzle, IdentityDither,
		XDerivKernel,
		YDerivKernel,
		CBoxImageFilterKernel
		>;

	const auto extent = _inImg->getCreationParameters().extent;
	const float mlt = static_cast<float>(std::max(extent.width, extent.height));
	XDerivKernel xderiv(XDerivKernel_(DerivKernel(DerivKernel_(ReconstructionKernel()), mlt), CBoxImageFilterKernel()));
	YDerivKernel yderiv(YDerivKernel_(CBoxImageFilterKernel(), DerivKernel(DerivKernel_(ReconstructionKernel()), mlt)));

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

	DerivativeMapFilter::execute(&state);

	_NBL_ALIGNED_FREE(state.scratchMemory);

	return outImg;
}

int main()
{
	constexpr uint32_t NBL_WINDOW_WIDTH = 1280;
	constexpr uint32_t NBL_WINDOW_HEIGHT = 720;
	constexpr uint32_t FBO_COUNT = 1u;

	auto initOutput = CommonAPI::Init<NBL_WINDOW_WIDTH, NBL_WINDOW_HEIGHT, FBO_COUNT>(video::EAT_OPENGL, "DerivMapTest");
	auto window = std::move(initOutput.window);
	auto gl = std::move(initOutput.apiConnection);
	auto surface = std::move(initOutput.surface);
	auto gpuPhysicalDevice = std::move(initOutput.physicalDevice);
	auto logicalDevice = std::move(initOutput.logicalDevice);
	auto queues = std::move(initOutput.queues);
	auto swapchain = std::move(initOutput.swapchain);
	auto renderpass = std::move(initOutput.renderpass);
	auto fbo = std::move(initOutput.fbo[0]);
	auto commandPool = std::move(initOutput.commandPool);
	auto assetManager = std::move(initOutput.assetManager);
	auto cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
	nbl::video::IGPUObjectFromAssetConverter cpu2gpu;

	core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> commandBuffer;
	logicalDevice->createCommandBuffers(commandPool.get(), nbl::video::IGPUCommandBuffer::EL_PRIMARY, 1, &commandBuffer);

	auto createDescriptorPool = [&](const uint32_t textureCount)
	{
		constexpr uint32_t maxItemCount = 256u;
		{
			nbl::video::IDescriptorPool::SDescriptorPoolSize poolSize;
			poolSize.count = textureCount;
			poolSize.type = nbl::asset::EDT_COMBINED_IMAGE_SAMPLER;
			return logicalDevice->createDescriptorPool(static_cast<nbl::video::IDescriptorPool::E_CREATE_FLAGS>(0), maxItemCount, 1u, &poolSize);
		}
	};

	nbl::video::IGPUDescriptorSetLayout::SBinding binding{ 0u, nbl::asset::EDT_COMBINED_IMAGE_SAMPLER, 1u, nbl::video::IGPUSpecializedShader::ESS_FRAGMENT, nullptr };
	auto gpuDescriptorSetLayout3 = logicalDevice->createGPUDescriptorSetLayout(&binding, &binding + 1u);
	auto gpuDescriptorPool = createDescriptorPool(1u); // per single texture
	auto fullScreenTriangle = nbl::ext::FullScreenTriangle::createFullScreenTriangle(cpu2gpuParams);

	auto createGPUPipeline = [&](nbl::asset::IImageView<nbl::asset::ICPUImage>::E_TYPE typeOfImage)
	{
		auto getPathToFragmentShader = [&]()
		{
			switch (typeOfImage)
			{
			case nbl::asset::IImageView<nbl::asset::ICPUImage>::ET_2D:
				return "../present2D.frag";
			case nbl::asset::IImageView<nbl::asset::ICPUImage>::ET_2D_ARRAY:
				return "../present2DArray.frag";
			case nbl::asset::IImageView<nbl::asset::ICPUImage>::ET_CUBE_MAP:
				return "../presentCubemap.frag";
			default:
			{
				assert(false);
			}
			}
		};

		auto fs_bundle = assetManager->getAsset(getPathToFragmentShader(), {});
		auto fs_contents = fs_bundle.getContents();
		if (fs_contents.begin() == fs_contents.end())
			assert(false);

		asset::ICPUSpecializedShader* cpuFragmentShader = static_cast<nbl::asset::ICPUSpecializedShader*>(fs_contents.begin()->get());

		nbl::core::smart_refctd_ptr<video::IGPUSpecializedShader> gpuFragmentShader;
		{
			auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&cpuFragmentShader, &cpuFragmentShader + 1, cpu2gpuParams);
			if (!gpu_array.get() || gpu_array->size() < 1u || !(*gpu_array)[0])
				assert(false);

			gpuFragmentShader = (*gpu_array)[0];
		}

		nbl::video::IGPUSpecializedShader* gpuShaders[2] = { std::get<0>(fullScreenTriangle).get(), gpuFragmentShader.get() };
		nbl::asset::SBlendParams blendParams;
		blendParams.logicOpEnable = false;
		blendParams.logicOp = nbl::asset::ELO_NO_OP;
		for (size_t i = 0ull; i < nbl::asset::SBlendParams::MAX_COLOR_ATTACHMENT_COUNT; i++)
			blendParams.blendParams[i].attachmentEnabled = (i == 0ull);
		nbl::asset::SRasterizationParams rasterParams;
		rasterParams.faceCullingMode = nbl::asset::EFCM_NONE;
		rasterParams.depthCompareOp = nbl::asset::ECO_ALWAYS;
		rasterParams.minSampleShading = 1.f;
		rasterParams.depthWriteEnable = false;
		rasterParams.depthTestEnable = false;

		auto gpuPipelineLayout = logicalDevice->createGPUPipelineLayout(nullptr, nullptr, nullptr, nullptr, nullptr, core::smart_refctd_ptr(gpuDescriptorSetLayout3));

		return logicalDevice->createGPURenderpassIndependentPipeline // TODO: crashes due to shaders
		(
			nullptr,
			std::move(gpuPipelineLayout),
			gpuShaders,
			gpuShaders + sizeof(gpuShaders) / sizeof(nbl::video::IGPUSpecializedShader*),
			std::get<nbl::asset::SVertexInputParams>(fullScreenTriangle),
			blendParams,
			std::get<nbl::asset::SPrimitiveAssemblyParams>(fullScreenTriangle),
			rasterParams
		);
	};

	auto gpuPipelineFor2D = createGPUPipeline(nbl::asset::IImageView<nbl::asset::ICPUImage>::E_TYPE::ET_2D);
	auto gpuPipelineFor2DArrays = createGPUPipeline(nbl::asset::IImageView<nbl::asset::ICPUImage>::E_TYPE::ET_2D_ARRAY);
	auto gpuPipelineForCubemaps = createGPUPipeline(nbl::asset::IImageView<nbl::asset::ICPUImage>::E_TYPE::ET_CUBE_MAP);

	nbl::core::smart_refctd_ptr<nbl::video::IGPUMeshBuffer> gpuMeshBuffer;
	{
		nbl::asset::SBufferBinding<nbl::video::IGPUBuffer> idxBinding{ 0ull, nullptr };
		gpuMeshBuffer = core::make_smart_refctd_ptr<nbl::video::IGPUMeshBuffer>(nullptr, nullptr, nullptr, std::move(idxBinding));
		gpuMeshBuffer->setIndexCount(3u);
		gpuMeshBuffer->setInstanceCount(1u);
	}

	nbl::core::vector<nbl::core::smart_refctd_ptr<nbl::asset::ICPUImageView>> cpuImageViews;
	nbl::core::vector<NBL_CAPTION_DATA_TO_DISPLAY> captionTexturesData;
	{
		std::ifstream list(testingImagePathsFile.data());
		if (list.is_open())
		{
			std::string line;
			for (; std::getline(list, line); )
			{
				if (line != "" && line[0] != ';')
				{
					auto& pathToTexture = line;
					auto& newCpuImageViewTexture = cpuImageViews.emplace_back();

					constexpr auto cachingFlags = static_cast<nbl::asset::IAssetLoader::E_CACHING_FLAGS>(nbl::asset::IAssetLoader::ECF_DONT_CACHE_REFERENCES & nbl::asset::IAssetLoader::ECF_DONT_CACHE_TOP_LEVEL);
					nbl::asset::IAssetLoader::SAssetLoadParams loadParams(0ull, nullptr, cachingFlags);
					auto cpuTextureBundle = assetManager->getAsset(pathToTexture, loadParams);
					auto cpuTextureContents = cpuTextureBundle.getContents();

					if (cpuTextureContents.begin() == cpuTextureContents.end())
						assert(false); // cannot perform test in this scenario

					auto asset = *cpuTextureContents.begin();
					{
						bool status = asset->getAssetType() == IAsset::ET_IMAGE;
						assert(status);
					}
				
					auto cpuImage = core::smart_refctd_ptr_static_cast<ICPUImage>(std::move(asset));
					cpuImage = createDerivMapFromHeightMap(cpuImage.get(), ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETC_CLAMP_TO_EDGE, ISampler::ETBC_FLOAT_OPAQUE_BLACK);
					{
						nbl::asset::ICPUImageView::SCreationParams viewParams;
						viewParams.flags = static_cast<decltype(viewParams.flags)>(0u);
						viewParams.image = core::smart_refctd_ptr(cpuImage);
						viewParams.format = viewParams.image->getCreationParameters().format;
						viewParams.viewType = decltype(viewParams.viewType)::ET_2D;
						viewParams.subresourceRange.baseArrayLayer = 0u;
						viewParams.subresourceRange.layerCount = 1u;
						viewParams.subresourceRange.baseMipLevel = 0u;
						viewParams.subresourceRange.levelCount = 1u;

						newCpuImageViewTexture = nbl::asset::ICPUImageView::create(std::move(viewParams));
					}
					
					std::string finalFileNameWithExtension;
					std::filesystem::path filename, extension;
					core::splitFilename(pathToTexture.c_str(), nullptr, &filename, &extension);

					auto& captionData = captionTexturesData.emplace_back();
					captionData.name = filename.string();
					captionData.extension = extension.string();
					captionData.viewType = [&]()
					{
						const auto& viewType = newCpuImageViewTexture->getCreationParameters().viewType;

						if (viewType == nbl::asset::IImageView<nbl::video::IGPUImage>::ET_2D)
							return std::string("ET_2D");
						else if (viewType == nbl::asset::IImageView<nbl::video::IGPUImage>::ET_2D_ARRAY)
							return std::string("ET_2D_ARRAY");
						else if (viewType == nbl::asset::IImageView<nbl::video::IGPUImage>::ET_CUBE_MAP)
							return std::string("ET_CUBE_MAP");
						else
							assert(false);
					}();

					finalFileNameWithExtension = filename.string() + ".";
					finalFileNameWithExtension += extension.string();

					auto tryToWrite = [&](asset::IAsset* asset)
					{
						asset::IAssetWriter::SAssetWriteParams wparams(asset);
						std::string assetPath = "imageAsset_" + finalFileNameWithExtension;
						return assetManager->writeAsset(assetPath, wparams);
					};

					if (!tryToWrite(newCpuImageViewTexture->getCreationParameters().image.get()))
						if (!tryToWrite(newCpuImageViewTexture.get()))
							assert(false); // could not write an asset
				}
			}
		}
	}

	auto gpuImageViews = cpu2gpu.getGPUObjectsFromAssets(cpuImageViews.data(), cpuImageViews.data() + cpuImageViews.size(), cpu2gpuParams);
	if (!gpuImageViews || gpuImageViews->size() < cpuImageViews.size())
		assert(false);

	auto getCurrentGPURenderpassIndependentPipeline = [&](nbl::video::IGPUImageView* gpuImageView)
	{
		switch (gpuImageView->getCreationParameters().viewType)
		{
			case nbl::asset::IImageView<nbl::video::IGPUImage>::ET_2D:
			{
				return gpuPipelineFor2D;
			}

			case nbl::asset::IImageView<nbl::video::IGPUImage>::ET_2D_ARRAY:
			{
				return gpuPipelineFor2DArrays;
			}

			case nbl::asset::IImageView<nbl::video::IGPUImage>::ET_CUBE_MAP:
			{
				return gpuPipelineForCubemaps;
			}

			default:
			{
				assert(false);
			}
		}
	};

	auto presentImageOnTheScreen = [&](nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> gpuImageView, const NBL_CAPTION_DATA_TO_DISPLAY& captionData)
	{
		auto gpuSamplerDescriptorSet3 = logicalDevice->createGPUDescriptorSet(gpuDescriptorPool.get(), nbl::core::smart_refctd_ptr(gpuDescriptorSetLayout3));

		nbl::video::IGPUDescriptorSet::SDescriptorInfo info;
		{
			info.desc = gpuImageView;
			nbl::asset::ISampler::SParams samplerParams = { nbl::asset::ISampler::ETC_CLAMP_TO_EDGE, nbl::asset::ISampler::ETC_CLAMP_TO_EDGE, nbl::asset::ISampler::ETC_CLAMP_TO_EDGE, nbl::asset::ISampler::ETBC_FLOAT_OPAQUE_BLACK, nbl::asset::ISampler::ETF_LINEAR, nbl::asset::ISampler::ETF_LINEAR, nbl::asset::ISampler::ESMM_LINEAR, 0u, false, nbl::asset::ECO_ALWAYS };
			info.image.sampler = logicalDevice->createGPUSampler(samplerParams);
			info.image.imageLayout = nbl::asset::EIL_SHADER_READ_ONLY_OPTIMAL;
		}

		nbl::video::IGPUDescriptorSet::SWriteDescriptorSet write;
		write.dstSet = gpuSamplerDescriptorSet3.get();
		write.binding = 0u;
		write.arrayElement = 0u;
		write.count = 1u;
		write.descriptorType = nbl::asset::EDT_COMBINED_IMAGE_SAMPLER;
		write.info = &info;

		logicalDevice->updateDescriptorSets(1u, &write, 0u, nullptr);

		auto currentGpuRenderpassIndependentPipeline = getCurrentGPURenderpassIndependentPipeline(gpuImageView.get());
		nbl::core::smart_refctd_ptr<nbl::video::IGPUGraphicsPipeline> gpuGraphicsPipeline;
		{
			nbl::video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams;
			graphicsPipelineParams.renderpassIndependent = core::smart_refctd_ptr<nbl::video::IGPURenderpassIndependentPipeline>(const_cast<video::IGPURenderpassIndependentPipeline*>(currentGpuRenderpassIndependentPipeline.get()));
			graphicsPipelineParams.renderpass = core::smart_refctd_ptr(renderpass);

			gpuGraphicsPipeline = logicalDevice->createGPUGraphicsPipeline(nullptr, std::move(graphicsPipelineParams));
		}

		const std::string windowCaption = "[Nabla Engine] Deriv Map Test Demo - CURRENT IMAGE: " + captionData.name + " - VIEW TYPE: " + captionData.viewType + " - EXTENSION: " + captionData.extension;
		window->setCaption(windowCaption);

		auto startPoint = std::chrono::high_resolution_clock::now();
		nbl::core::smart_refctd_ptr<nbl::video::IGPUSemaphore> render_finished_sem;

		while (true)
		{
			auto aPoint = std::chrono::high_resolution_clock::now();
			if (std::chrono::duration_cast<std::chrono::milliseconds>(aPoint - startPoint).count() > SWITCH_IMAGES_PER_X_MILISECONDS)
				break;

			commandBuffer->reset(nbl::video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
			commandBuffer->begin(0);

			asset::SViewport viewport;
			viewport.minDepth = 1.f;
			viewport.maxDepth = 0.f;
			viewport.x = 0u;
			viewport.y = 0u;
			viewport.width = NBL_WINDOW_WIDTH;
			viewport.height = NBL_WINDOW_HEIGHT;
			commandBuffer->setViewport(0u, 1u, &viewport);

			nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
			nbl::asset::VkRect2D area;
			area.offset = { 0,0 };
			area.extent = { NBL_WINDOW_WIDTH, NBL_WINDOW_HEIGHT };
			nbl::asset::SClearValue clear;
			clear.color.float32[0] = 1.f;
			clear.color.float32[1] = 1.f;
			clear.color.float32[2] = 1.f;
			clear.color.float32[3] = 1.f;
			beginInfo.clearValueCount = 1u;
			beginInfo.framebuffer = fbo;
			beginInfo.renderpass = renderpass;
			beginInfo.renderArea = area;
			beginInfo.clearValues = &clear;

			commandBuffer->beginRenderPass(&beginInfo, nbl::asset::ESC_INLINE);
			commandBuffer->bindGraphicsPipeline(gpuGraphicsPipeline.get());
			commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, currentGpuRenderpassIndependentPipeline->getLayout(), 3u, 1u, &gpuSamplerDescriptorSet3.get(), nullptr);
			commandBuffer->drawMeshBuffer(gpuMeshBuffer.get());
			commandBuffer->endRenderPass();
			commandBuffer->end();

			auto img_acq_sem = logicalDevice->createSemaphore();
			render_finished_sem = logicalDevice->createSemaphore();

			uint32_t imgnum = 0u;
			constexpr uint64_t MAX_TIMEOUT = 99999999999999ull; // ns
			swapchain->acquireNextImage(MAX_TIMEOUT, img_acq_sem.get(), nullptr, &imgnum);

			CommonAPI::Submit(logicalDevice.get(), swapchain.get(), commandBuffer.get(), queues[decltype(initOutput)::EQT_GRAPHICS], img_acq_sem.get(), render_finished_sem.get());
			CommonAPI::Present(logicalDevice.get(), swapchain.get(), queues[decltype(initOutput)::EQT_GRAPHICS], render_finished_sem.get(), imgnum);
		}

		const auto& fboCreationParams = fbo->getCreationParameters();
		auto gpuSourceImageView = fboCreationParams.attachments[0];

		const std::string writePath = "screenShot_deriv_" + captionData.name + ".png";
		bool status = ext::ScreenShot::createScreenShot(logicalDevice.get(), queues[decltype(initOutput)::EQT_TRANSFER_UP], render_finished_sem.get(), gpuSourceImageView.get(), assetManager.get(), writePath);
		return status;
	};

	for (size_t i = 0; i < gpuImageViews->size(); ++i)
	{
		auto gpuImageView = (*gpuImageViews)[i];
		auto& captionData = captionTexturesData[i];

		bool status = presentImageOnTheScreen(nbl::core::smart_refctd_ptr(gpuImageView), captionData);
		assert(status);
	}
}