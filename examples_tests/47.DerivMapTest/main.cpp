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

class DerivMapTestApp : public ApplicationBase
{
	static constexpr uint32_t NBL_WINDOW_WIDTH = 1280;
	static constexpr uint32_t NBL_WINDOW_HEIGHT = 720;
	static constexpr uint32_t SC_IMG_COUNT = 3u;
	static constexpr uint32_t FRAMES_IN_FLIGHT = 5u;
	static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

public:
	struct Nabla : IUserData
	{
		nbl::core::smart_refctd_ptr<nbl::ui::IWindowManager> windowManager;
		nbl::core::smart_refctd_ptr<nbl::ui::IWindow> window;
		nbl::core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCb;
		nbl::core::smart_refctd_ptr<nbl::video::IAPIConnection> gl;
		nbl::core::smart_refctd_ptr<nbl::video::ISurface> surface;
		nbl::core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
		nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
		nbl::video::IPhysicalDevice* gpuPhysicalDevice;
		std::array<nbl::video::IGPUQueue*, CommonAPI::InitOutput<SC_IMG_COUNT>::EQT_COUNT> queues = { nullptr, nullptr, nullptr, nullptr };
		nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
		nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
		std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, SC_IMG_COUNT> fbos;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool> commandPool;
		nbl::core::smart_refctd_ptr<nbl::system::ISystem> system;
		nbl::core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
		nbl::video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
		nbl::core::smart_refctd_ptr<nbl::system::ILogger> logger;
		nbl::core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;

		nbl::core::smart_refctd_ptr<nbl::video::IGPUFence> gpuTransferFence;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUSemaphore> gpuTransferSemaphore;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUFence> gpuComputeFence;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUSemaphore> gpuComputeSemaphore;
		nbl::video::IGPUObjectFromAssetConverter cpu2gpu;

		video::created_gpu_object_array<ICPUImageView> gpuImageViews;
		nbl::core::vector<NBL_CAPTION_DATA_TO_DISPLAY> captionTexturesData;
		uint32_t imagesPresented = 0u;

		core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> gpuPipelineFor2D;
		core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> gpuPipelineFor2DArrays;
		core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> gpuPipelineForCubemaps;
		core::smart_refctd_ptr<IDescriptorPool> gpuDescriptorPool;
		core::smart_refctd_ptr<IGPUDescriptorSetLayout> gpuDescriptorSetLayout3;

		core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> getCurrentGPURenderpassIndependentPipeline(nbl::video::IGPUImageView* gpuImageView)
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

		bool presentImageOnTheScreen(nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> gpuImageView, const NBL_CAPTION_DATA_TO_DISPLAY& captionData)
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

			const std::string windowCaption = "[Nabla Engine] Derivative Map Test Demo - CURRENT IMAGE: " + captionData.name + " - VIEW TYPE: " + captionData.viewType + " - EXTENSION: " + captionData.extension;
			window->setCaption(windowCaption);

			core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[FRAMES_IN_FLIGHT];
			logicalDevice->createCommandBuffers(commandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, commandBuffers);

			core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
			core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
			core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };

			for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
			{
				imageAcquire[i] = logicalDevice->createSemaphore();
				renderFinished[i] = logicalDevice->createSemaphore();
			}

			auto startPoint = std::chrono::high_resolution_clock::now();

			constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;
			uint32_t acquiredNextFBO = {};
			auto resourceIx = -1;
			while (true)
			{
				++resourceIx;
				if (resourceIx >= FRAMES_IN_FLIGHT)
					resourceIx = 0;

				auto& commandBuffer = commandBuffers[resourceIx];
				auto& fence = frameComplete[resourceIx];

				if (fence)
					while (logicalDevice->waitForFences(1u, &fence.get(), false, MAX_TIMEOUT) == video::IGPUFence::ES_TIMEOUT) {}
				else
					fence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

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

				swapchain->acquireNextImage(MAX_TIMEOUT, imageAcquire[resourceIx].get(), nullptr, &acquiredNextFBO);

				nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
				{
					VkRect2D area;
					area.offset = { 0,0 };
					area.extent = { NBL_WINDOW_WIDTH, NBL_WINDOW_HEIGHT };
					nbl::asset::SClearValue clear;
					clear.color.float32[0] = 1.f;
					clear.color.float32[1] = 1.f;
					clear.color.float32[2] = 1.f;
					clear.color.float32[3] = 1.f;
					beginInfo.clearValueCount = 1u;
					beginInfo.framebuffer = fbos[acquiredNextFBO];
					beginInfo.renderpass = renderpass;
					beginInfo.renderArea = area;
					beginInfo.clearValues = &clear;
				}

				commandBuffer->beginRenderPass(&beginInfo, nbl::asset::ESC_INLINE);
				commandBuffer->bindGraphicsPipeline(gpuGraphicsPipeline.get());
				commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuGraphicsPipeline->getRenderpassIndependentPipeline()->getLayout(), 3, 1, &gpuSamplerDescriptorSet3.get(), nullptr);
				ext::FullScreenTriangle::recordDrawCalls(commandBuffer.get());
				commandBuffer->endRenderPass();
				commandBuffer->end();

				CommonAPI::Submit(logicalDevice.get(), swapchain.get(), commandBuffer.get(), queues[CommonAPI::InitOutput<1>::EQT_GRAPHICS], imageAcquire[resourceIx].get(), renderFinished[resourceIx].get(), fence.get());
				CommonAPI::Present(logicalDevice.get(), swapchain.get(), queues[CommonAPI::InitOutput<1>::EQT_GRAPHICS], renderFinished[resourceIx].get(), acquiredNextFBO);
			}

			const auto& fboCreationParams = fbos[acquiredNextFBO]->getCreationParameters();
			auto gpuSourceImageView = fboCreationParams.attachments[0];

			const std::string writePath = "screenShot_" + captionData.name + ".png";
			bool status = ext::ScreenShot::createScreenShot(logicalDevice.get(), queues[CommonAPI::InitOutput<1>::EQT_TRANSFER_UP], renderFinished[resourceIx].get(), gpuSourceImageView.get(), assetManager.get(), writePath);
			return status;
		};

		void setWindow(core::smart_refctd_ptr<nbl::ui::IWindow>&& wnd) override
		{
			window = std::move(wnd);
		}
	};

	APP_CONSTRUCTOR(DerivMapTestApp)

	void onAppInitialized_impl(void* data) override
	{
		Nabla* engine = static_cast<Nabla*>(data);

		
		CommonAPI::InitOutput<SC_IMG_COUNT> initOutput;
		initOutput.window = core::smart_refctd_ptr(engine->window);
		CommonAPI::Init<NBL_WINDOW_WIDTH, NBL_WINDOW_HEIGHT, SC_IMG_COUNT>(initOutput, video::EAT_OPENGL, "DerivMapTest", nbl::asset::EF_D32_SFLOAT);
		engine->window = std::move(initOutput.window);
		engine->gl = std::move(initOutput.apiConnection);
		engine->surface = std::move(initOutput.surface);
		engine->gpuPhysicalDevice = std::move(initOutput.physicalDevice);
		engine->logicalDevice = std::move(initOutput.logicalDevice);
		engine->queues = std::move(initOutput.queues);
		engine->swapchain = std::move(initOutput.swapchain);
		engine->renderpass = std::move(initOutput.renderpass);
		engine->fbos = std::move(initOutput.fbo);
		engine->commandPool = std::move(initOutput.commandPool);
		engine->assetManager = std::move(initOutput.assetManager);

		{
			engine->gpuTransferFence = engine->logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));
			engine->gpuTransferSemaphore = engine->logicalDevice->createSemaphore();

			engine->gpuComputeFence = engine->logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));
			engine->gpuComputeSemaphore = engine->logicalDevice->createSemaphore();

			engine->cpu2gpuParams.assetManager = engine->assetManager.get();
			engine->cpu2gpuParams.device = engine->logicalDevice.get();
			engine->cpu2gpuParams.finalQueueFamIx = engine->queues[decltype(initOutput)::EQT_GRAPHICS]->getFamilyIndex();
			engine->cpu2gpuParams.limits = engine->gpuPhysicalDevice->getLimits();
			engine->cpu2gpuParams.pipelineCache = nullptr;
			engine->cpu2gpuParams.sharingMode = nbl::asset::ESM_EXCLUSIVE;

			engine->cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].fence = &engine->gpuTransferFence;
			engine->cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].semaphore = &engine->gpuTransferSemaphore;
			engine->cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].queue = engine->queues[decltype(initOutput)::EQT_TRANSFER_UP];

			engine->cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].fence = &engine->gpuComputeFence;
			engine->cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].semaphore = &engine->gpuComputeSemaphore;
			engine->cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].queue = engine->queues[decltype(initOutput)::EQT_COMPUTE];
		}

		auto createDescriptorPool = [&](const uint32_t textureCount)
		{
			constexpr uint32_t maxItemCount = 256u;
			{
				nbl::video::IDescriptorPool::SDescriptorPoolSize poolSize;
				poolSize.count = textureCount;
				poolSize.type = nbl::asset::EDT_COMBINED_IMAGE_SAMPLER;
				return engine->logicalDevice->createDescriptorPool(static_cast<nbl::video::IDescriptorPool::E_CREATE_FLAGS>(0), maxItemCount, 1u, &poolSize);
			}
		};

		nbl::video::IGPUDescriptorSetLayout::SBinding binding{ 0u, nbl::asset::EDT_COMBINED_IMAGE_SAMPLER, 1u, nbl::video::IGPUSpecializedShader::ESS_FRAGMENT, nullptr };
		engine->gpuDescriptorSetLayout3 = engine->logicalDevice->createGPUDescriptorSetLayout(&binding, &binding + 1u);
		engine->gpuDescriptorPool = createDescriptorPool(1u); // per single texture
		auto fstProtoPipeline = nbl::ext::FullScreenTriangle::createProtoPipeline(engine->cpu2gpuParams);
		{
			engine->gpuTransferFence = std::move(engine->logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0)));
			engine->gpuTransferSemaphore = std::move(engine->logicalDevice->createSemaphore());

			engine->gpuComputeFence = std::move(engine->logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0)));
			engine->gpuComputeSemaphore = std::move(engine->logicalDevice->createSemaphore());
		}

		auto createGPUPipeline = [&](nbl::asset::IImageView<nbl::asset::ICPUImage>::E_TYPE typeOfImage) -> core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline>
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

			auto fs_bundle = engine->assetManager->getAsset(getPathToFragmentShader(), {});
			auto fs_contents = fs_bundle.getContents();
			if (fs_contents.begin() == fs_contents.end())
				assert(false);

			asset::ICPUSpecializedShader* cpuFragmentShader = static_cast<nbl::asset::ICPUSpecializedShader*>(fs_contents.begin()->get());

			nbl::core::smart_refctd_ptr<video::IGPUSpecializedShader> gpuFragmentShader;
			{
				auto gpu_array = engine->cpu2gpu.getGPUObjectsFromAssets(&cpuFragmentShader, &cpuFragmentShader + 1, engine->cpu2gpuParams);
				if (!gpu_array.get() || gpu_array->size() < 1u || !(*gpu_array)[0])
					assert(false);

				gpuFragmentShader = (*gpu_array)[0];
			}

			auto gpuPipelineLayout = engine->logicalDevice->createGPUPipelineLayout(nullptr, nullptr, nullptr, nullptr, nullptr, core::smart_refctd_ptr(engine->gpuDescriptorSetLayout3));
			return ext::FullScreenTriangle::createRenderpassIndependentPipeline(engine->logicalDevice.get(), fstProtoPipeline, std::move(gpuFragmentShader), std::move(gpuPipelineLayout));
		};

		engine->gpuPipelineFor2D = createGPUPipeline(nbl::asset::IImageView<nbl::asset::ICPUImage>::E_TYPE::ET_2D);
		engine->gpuPipelineFor2DArrays = nullptr; // createGPUPipeline(nbl::asset::IImageView<nbl::asset::ICPUImage>::E_TYPE::ET_2D_ARRAY);
		engine->gpuPipelineForCubemaps = nullptr; // createGPUPipeline(nbl::asset::IImageView<nbl::asset::ICPUImage>::E_TYPE::ET_CUBE_MAP);

		nbl::core::vector<nbl::core::smart_refctd_ptr<nbl::asset::ICPUImageView>> cpuImageViews;
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
						auto cpuTextureBundle = engine->assetManager->getAsset(pathToTexture, loadParams);
						auto cpuTextureContents = cpuTextureBundle.getContents();
						{
							bool status = !cpuTextureContents.empty();
							assert(status);
						}

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

						std::filesystem::path filename, extension;
						core::splitFilename(pathToTexture.c_str(), nullptr, &filename, &extension);

						auto& captionData = engine->captionTexturesData.emplace_back();
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

						const std::string finalFileNameWithExtension = captionData.name + captionData.extension;
						std::cout << finalFileNameWithExtension << "\n";

						auto tryToWrite = [&](asset::IAsset* asset)
						{
							asset::IAssetWriter::SAssetWriteParams wparams(asset);
							std::string assetPath = "imageAsset_" + finalFileNameWithExtension;
							return engine->assetManager->writeAsset(assetPath, wparams);
						};

						if (!tryToWrite(newCpuImageViewTexture->getCreationParameters().image.get()))
							if (!tryToWrite(newCpuImageViewTexture.get()))
								assert(false); // could not write an asset
					}
				}
			}
		}
		
		engine->gpuImageViews = engine->cpu2gpu.getGPUObjectsFromAssets(cpuImageViews.data(), cpuImageViews.data() + cpuImageViews.size(), engine->cpu2gpuParams);
		if (!engine->gpuImageViews || engine->gpuImageViews->size() < cpuImageViews.size())
			assert(false);
	}

	void workLoopBody(void* data) override
	{
		Nabla* engine = static_cast<Nabla*>(data);

		auto gpuImageView = (*engine->gpuImageViews)[engine->imagesPresented];
		auto& captionData = engine->captionTexturesData[engine->imagesPresented];

		bool status = engine->presentImageOnTheScreen(nbl::core::smart_refctd_ptr(gpuImageView), captionData);
		assert(status);

		engine->imagesPresented++;
	}

	bool keepRunning(void* params) override
	{
		Nabla* engine = static_cast<Nabla*>(params);
		return engine->imagesPresented < engine->gpuImageViews->size();
	}
};

NBL_COMMON_API_MAIN(DerivMapTestApp, DerivMapTestApp::Nabla)