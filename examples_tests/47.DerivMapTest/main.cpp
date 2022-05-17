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
		DefaultSwizzle,
		IdentityDither,
		void, //TODO: fix
		true,
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

	using blit_utils_t = asset::CBlitUtilities<XDerivKernel, YDerivKernel, CBoxImageFilterKernel>;
	if (!blit_utils_t::computePhaseSupportLUT(state.scratchMemory + DerivativeMapFilter::getPhaseSupportLUTByteOffset(&state), state.inExtentLayerCount, state.outExtentLayerCount, state.inImage->getCreationParameters().type, xderiv, yderiv, CBoxImageFilterKernel()))
		return nullptr;

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
	nbl::core::smart_refctd_ptr<nbl::ui::IWindowManager> windowManager;
	nbl::core::smart_refctd_ptr<nbl::ui::IWindow> window;
	nbl::core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCb;
	nbl::core::smart_refctd_ptr<nbl::video::IAPIConnection> gl;
	nbl::core::smart_refctd_ptr<nbl::video::ISurface> surface;
	nbl::core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
	nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
	nbl::video::IPhysicalDevice* gpuPhysicalDevice;
	std::array<nbl::video::IGPUQueue*, CommonAPI::InitOutput::MaxQueuesCount> queues = { nullptr, nullptr, nullptr, nullptr };
	nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
	nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
	std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, CommonAPI::InitOutput::MaxSwapChainImageCount> fbos;
	std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, CommonAPI::InitOutput::MaxQueuesCount> commandPools;
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
		logicalDevice->createCommandBuffers(commandPools[CommonAPI::InitOutput::EQT_GRAPHICS].get(), video::IGPUCommandBuffer::EL_PRIMARY, FRAMES_IN_FLIGHT, commandBuffers);

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

			{
				nbl::video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
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
				commandBuffer->beginRenderPass(&beginInfo, nbl::asset::ESC_INLINE);
			}

			commandBuffer->bindGraphicsPipeline(gpuGraphicsPipeline.get());
			commandBuffer->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuGraphicsPipeline->getRenderpassIndependentPipeline()->getLayout(), 3, 1, &gpuSamplerDescriptorSet3.get(), 0u);
			ext::FullScreenTriangle::recordDrawCalls(commandBuffer.get());
			commandBuffer->endRenderPass();
			commandBuffer->end();

			CommonAPI::Submit(logicalDevice.get(), swapchain.get(), commandBuffer.get(), queues[CommonAPI::InitOutput::EQT_GRAPHICS], imageAcquire[resourceIx].get(), renderFinished[resourceIx].get(), fence.get());
			CommonAPI::Present(logicalDevice.get(), swapchain.get(), queues[CommonAPI::InitOutput::EQT_GRAPHICS], renderFinished[resourceIx].get(), acquiredNextFBO);
		}

		const auto& fboCreationParams = fbos[acquiredNextFBO]->getCreationParameters();
		auto gpuSourceImageView = fboCreationParams.attachments[0];

		const std::string writePath = "screenShot_" + captionData.name + ".png";
		//TODO: what should be last parameter here?
		bool status = ext::ScreenShot::createScreenShot(
			logicalDevice.get(), 
			queues[CommonAPI::InitOutput::EQT_TRANSFER_UP],
			renderFinished[resourceIx].get(),
			gpuSourceImageView.get(),
			assetManager.get(),
			writePath, 
			asset::EIL_PRESENT_SRC,
			static_cast<asset::E_ACCESS_FLAGS>(0u));

		return status;
	};

	void setWindow(core::smart_refctd_ptr<nbl::ui::IWindow>&& wnd) override
	{
		window = std::move(wnd);
	}
	void setSystem(core::smart_refctd_ptr<nbl::system::ISystem>&& s) override
	{
		system = std::move(s);
	}
	nbl::ui::IWindow* getWindow() override
	{
		return window.get();
	}
	video::IAPIConnection* getAPIConnection() override
	{
		return gl.get();
	}
	video::ILogicalDevice* getLogicalDevice()  override
	{
		return logicalDevice.get();
	}
	video::IGPURenderpass* getRenderpass() override
	{
		return renderpass.get();
	}
	void setSurface(core::smart_refctd_ptr<video::ISurface>&& s) override
	{
		surface = std::move(s);
	}
	void setFBOs(std::vector<core::smart_refctd_ptr<video::IGPUFramebuffer>>& f) override
	{
		for (int i = 0; i < f.size(); i++)
		{
			fbos[i] = core::smart_refctd_ptr(f[i]);
		}
	}
	void setSwapchain(core::smart_refctd_ptr<video::ISwapchain>&& s) override
	{
		swapchain = std::move(s);
	}
	uint32_t getSwapchainImageCount() override
	{
		return SC_IMG_COUNT;
	}
	virtual nbl::asset::E_FORMAT getDepthFormat() override
	{
		return nbl::asset::EF_D32_SFLOAT;
	}

	APP_CONSTRUCTOR(DerivMapTestApp)

	void onAppInitialized_impl() override
	{
		CommonAPI::InitOutput initOutput;
		initOutput.window = core::smart_refctd_ptr(window);
		initOutput.system = core::smart_refctd_ptr(system);

		const auto swapchainImageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_COLOR_ATTACHMENT_BIT);
		const video::ISurface::SFormat surfaceFormat(asset::EF_R8G8B8A8_SRGB, asset::ECP_COUNT, asset::EOTF_UNKNOWN);

		CommonAPI::InitWithDefaultExt(initOutput, video::EAT_OPENGL, "MeshLoaders", NBL_WINDOW_WIDTH, NBL_WINDOW_HEIGHT, SC_IMG_COUNT, swapchainImageUsage, surfaceFormat);
		window = std::move(initOutput.window);
		gl = std::move(initOutput.apiConnection);
		surface = std::move(initOutput.surface);
		gpuPhysicalDevice = std::move(initOutput.physicalDevice);
		logicalDevice = std::move(initOutput.logicalDevice);
		queues = std::move(initOutput.queues);
		swapchain = std::move(initOutput.swapchain);
		renderpass = std::move(initOutput.renderpass);
		fbos = std::move(initOutput.fbo);
		commandPools = std::move(initOutput.commandPools);
		assetManager = std::move(initOutput.assetManager);
		cpu2gpuParams = std::move(initOutput.cpu2gpuParams);

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

		nbl::video::IGPUDescriptorSetLayout::SBinding binding{ 0u, nbl::asset::EDT_COMBINED_IMAGE_SAMPLER, 1u, nbl::video::IGPUShader::ESS_FRAGMENT, nullptr };
		gpuDescriptorSetLayout3 = logicalDevice->createGPUDescriptorSetLayout(&binding, &binding + 1u);
		gpuDescriptorPool = createDescriptorPool(1u); // per single texture
		auto fstProtoPipeline = nbl::ext::FullScreenTriangle::createProtoPipeline(cpu2gpuParams);
		{
			gpuTransferFence = std::move(logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0)));
			gpuTransferSemaphore = std::move(logicalDevice->createSemaphore());

			gpuComputeFence = std::move(logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0)));
			gpuComputeSemaphore = std::move(logicalDevice->createSemaphore());
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
						break;
					}
				}
				return "";
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

			auto gpuPipelineLayout = logicalDevice->createGPUPipelineLayout(nullptr, nullptr, nullptr, nullptr, nullptr, core::smart_refctd_ptr(gpuDescriptorSetLayout3));
			return ext::FullScreenTriangle::createRenderpassIndependentPipeline(logicalDevice.get(), fstProtoPipeline, std::move(gpuFragmentShader), std::move(gpuPipelineLayout));
		};

		gpuPipelineFor2D = createGPUPipeline(nbl::asset::IImageView<nbl::asset::ICPUImage>::E_TYPE::ET_2D);
		gpuPipelineFor2DArrays = nullptr; // createGPUPipeline(nbl::asset::IImageView<nbl::asset::ICPUImage>::E_TYPE::ET_2D_ARRAY);
		gpuPipelineForCubemaps = nullptr; // createGPUPipeline(nbl::asset::IImageView<nbl::asset::ICPUImage>::E_TYPE::ET_CUBE_MAP);

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
						auto cpuTextureBundle = assetManager->getAsset(pathToTexture, loadParams);
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

						auto& captionData = captionTexturesData.emplace_back();
						captionData.name = filename.string();
						captionData.extension = extension.string();
						captionData.viewType = [&]() -> std::string
						{
							const auto& viewType = newCpuImageViewTexture->getCreationParameters().viewType;

							if (viewType == nbl::asset::IImageView<nbl::video::IGPUImage>::ET_2D)
								return "ET_2D";
							else if (viewType == nbl::asset::IImageView<nbl::video::IGPUImage>::ET_2D_ARRAY)
								return "ET_2D_ARRAY";
							else if (viewType == nbl::asset::IImageView<nbl::video::IGPUImage>::ET_CUBE_MAP)
								return "ET_CUBE_MAP";
							else
							{
								assert(false);
								return "";
							}
						}();

						const std::string finalFileNameWithExtension = captionData.name + captionData.extension;
						std::cout << finalFileNameWithExtension << "\n";

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
		
		cpu2gpuParams.beginCommandBuffers();
		gpuImageViews = cpu2gpu.getGPUObjectsFromAssets(cpuImageViews.data(), cpuImageViews.data()+cpuImageViews.size(), cpu2gpuParams);
		cpu2gpuParams.waitForCreationToComplete(false);
		assert(gpuImageViews && gpuImageViews->size()==cpuImageViews.size());
	}

	void workLoopBody() override
	{
		auto gpuImageView = gpuImageViews->operator[](imagesPresented);
		auto& captionData = captionTexturesData[imagesPresented];

		bool status = presentImageOnTheScreen(nbl::core::smart_refctd_ptr(gpuImageView), captionData);
		assert(status);

		imagesPresented++;
	}

	bool keepRunning() override
	{
		return imagesPresented < gpuImageViews->size();
	}
};

NBL_COMMON_API_MAIN(DerivMapTestApp, DerivMapTestApp::Nabla)