#include "nbl/asset/utils/CDerivativeMapCreator.h"

#include "nbl/asset/filters/CSwizzleAndConvertImageFilter.h"
#include "nbl/asset/filters/kernels/CChannelIndependentImageFilterKernel.h"
#include "nbl/asset/filters/kernels/CDerivativeImageFilterKernel.h"
#include "nbl/asset/filters/kernels/CGaussianImageFilterKernel.h"
#include "nbl/asset/filters/CBlitImageFilter.h"
#include "nbl/asset/interchange/IImageAssetHandlerBase.h"

namespace nbl {
namespace asset
{

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

				inline void operator()(value_type* windowSample, core::vectorSIMDf& relativePos, const core::vectorSIMDi32& globalTexelCoord, const asset::IImageFilterKernel::UserData* userData)
				{
					preFilter(windowSample, relativePos, globalTexelCoord, userData);
					auto* scale = asset::IImageFilterKernel::ScaleFactorUserData::cast(userData);
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

				inline void operator()(value_type* windowSample, core::vectorSIMDf& relativePos, const core::vectorSIMDi32& globalTexelCoord, const asset::IImageFilterKernel::UserData* userData)
				{
					preFilter(windowSample, relativePos, globalTexelCoord, userData);
					auto* scale = asset::IImageFilterKernel::ScaleFactorUserData::cast(userData);
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

core::smart_refctd_ptr<asset::ICPUImage> nbl::asset::CDerivativeMapCreator::createDerivativeMapFromHeightMap(asset::ICPUImage* _inImg, asset::ISampler::E_TEXTURE_CLAMP _uwrap, asset::ISampler::E_TEXTURE_CLAMP _vwrap, asset::ISampler::E_TEXTURE_BORDER_COLOR _borderColor)
{
	using namespace asset;

	auto getRGformat = [](asset::E_FORMAT f) -> asset::E_FORMAT {
		const uint32_t bytesPerChannel = (getBytesPerPixel(f) * core::rational(1, getFormatChannelCount(f))).getIntegerApprox();
		switch (bytesPerChannel)
		{
		case 1u:
			return asset::EF_R8G8_SNORM;
		case 2u:
			return asset::EF_R16G16_SNORM;
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
	constexpr bool NORMALIZE = false;
	using DerivativeMapFilter = CBlitImageFilter
		<
		NORMALIZE, false, DefaultSwizzle, IdentityDither, // (Criss, look at impl::CSwizzleAndConvertImageFilterBase)
		XDerivKernel,
		YDerivKernel,
		CBoxImageFilterKernel
		>;

	const auto extent = _inImg->getCreationParameters().extent;
	const float mlt = 1.f;// static_cast<float>(std::max(extent.width, extent.height));
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

	DerivativeMapFilter::execute(std::execution::par_unseq,&state);

	_NBL_ALIGNED_FREE(state.scratchMemory);

	return outImg;
}

core::smart_refctd_ptr<asset::ICPUImageView> CDerivativeMapCreator::createDerivativeMapViewFromHeightMap(asset::ICPUImage* _inImg, asset::ISampler::E_TEXTURE_CLAMP _uwrap, asset::ISampler::E_TEXTURE_CLAMP _vwrap, asset::ISampler::E_TEXTURE_BORDER_COLOR _borderColor)
{
	auto img = createDerivativeMapFromHeightMap(_inImg, _uwrap, _vwrap, _borderColor);
	const auto& iparams = img->getCreationParameters();

	asset::ICPUImageView::SCreationParams params;
	params.format = iparams.format;
	params.subresourceRange.baseArrayLayer = 0u;
	params.subresourceRange.layerCount = iparams.arrayLayers;
	assert(params.subresourceRange.layerCount == 1u);
	params.subresourceRange.baseMipLevel = 0u;
	params.subresourceRange.levelCount = iparams.mipLevels;
	params.viewType = asset::IImageView<asset::ICPUImage>::ET_2D;
	params.flags = static_cast<asset::IImageView<asset::ICPUImage>::E_CREATE_FLAGS>(0);
	params.image = std::move(img);

	return asset::ICPUImageView::create(std::move(params));
}

core::smart_refctd_ptr<asset::ICPUImage> nbl::asset::CDerivativeMapCreator::createDerivativeMapFromNormalMap(asset::ICPUImage* _inImg)
{
	core::smart_refctd_ptr<ICPUImage> newDerivativeNormalMapImage;
	{
		bool status = _inImg->getCreationParameters().type == IImage::E_TYPE::ET_2D;
		assert(status);
	}

	auto cpuImageNormalTexture = _inImg;
	const auto referenceImageParams = cpuImageNormalTexture->getCreationParameters();
	const auto referenceBuffer = cpuImageNormalTexture->getBuffer();
	const auto referenceRegions = cpuImageNormalTexture->getRegions();
	const auto* referenceRegion = referenceRegions.begin();

	auto newImageParams = referenceImageParams;
	newImageParams.format = getRGformat(referenceImageParams.format);

	const uint32_t pitch = IImageAssetHandlerBase::calcPitchInBlocks(referenceImageParams.extent.width, asset::getTexelOrBlockBytesize(newImageParams.format));
	core::smart_refctd_ptr<ICPUBuffer> newCpuBuffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(asset::getTexelOrBlockBytesize(newImageParams.format) * pitch * newImageParams.extent.height);

	asset::ICPUImage::SBufferCopy region;
	region.imageOffset = { 0,0,0 };
	region.imageExtent = newImageParams.extent;
	region.imageSubresource.baseArrayLayer = 0u;
	region.imageSubresource.layerCount = 1u;
	region.imageSubresource.mipLevel = 0u;
	region.bufferRowLength = pitch;
	region.bufferImageHeight = 0u;
	region.bufferOffset = 0u;

	newDerivativeNormalMapImage = ICPUImage::create(std::move(newImageParams));
	newDerivativeNormalMapImage->setBufferAndRegions(std::move(newCpuBuffer), core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<IImage::SBufferCopy>>(1ull, region));

	using DerivativeNormalMapFilter = CNormalMapToDerivativeFilter<asset::DefaultSwizzle, asset::IdentityDither>;
	DerivativeNormalMapFilter derivativeNormalFilter;
	DerivativeNormalMapFilter::state_type state;

	state.inImage = cpuImageNormalTexture;
	state.outImage = newDerivativeNormalMapImage.get();
	state.inOffset = { 0, 0, 0 };
	state.inBaseLayer = 0;
	state.outOffset = { 0, 0, 0 };
	state.outBaseLayer = 0;
	state.extent = { referenceImageParams.extent.width, referenceImageParams.extent.height, referenceImageParams.extent.depth };
	state.layerCount = newDerivativeNormalMapImage->getCreationParameters().arrayLayers;

	state.scratchMemoryByteSize = state.getRequiredScratchByteSize(state.layerCount, state.extent);
	state.scratchMemory = reinterpret_cast<uint8_t*>(_NBL_ALIGNED_MALLOC(state.scratchMemoryByteSize, 32));

	state.inMipLevel = 0;
	state.outMipLevel = 0;

	if (!derivativeNormalFilter.execute(&state))
		os::Printer::log("Something went wrong while performing derivative filter operations!", ELL_ERROR);

	_NBL_ALIGNED_FREE(state.scratchMemory);

	return core::smart_refctd_ptr<ICPUImage>(state.outImage);
}

core::smart_refctd_ptr<asset::ICPUImageView> nbl::asset::CDerivativeMapCreator::createDerivativeMapViewFromNormalMap(asset::ICPUImage* _inImg)
{
	auto cpuDerivativeImage = createDerivativeMapFromNormalMap(_inImg);

	ICPUImageView::SCreationParams imageViewInfo;
	imageViewInfo.image = core::smart_refctd_ptr(cpuDerivativeImage);
	imageViewInfo.format = imageViewInfo.image->getCreationParameters().format;
	imageViewInfo.viewType = decltype(imageViewInfo.viewType)::ET_2D;
	imageViewInfo.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
	imageViewInfo.subresourceRange.baseArrayLayer = 0u;
	imageViewInfo.subresourceRange.baseMipLevel = 0u;
	imageViewInfo.subresourceRange.layerCount = imageViewInfo.image->getCreationParameters().arrayLayers;
	imageViewInfo.subresourceRange.levelCount = imageViewInfo.image->getCreationParameters().mipLevels;

	auto imageView = ICPUImageView::create(std::move(imageViewInfo));

	if (!imageView.get())
		os::Printer::log("Something went wrong while creating image view for derivative normal map!", ELL_ERROR);

	return imageView;
}

}
}