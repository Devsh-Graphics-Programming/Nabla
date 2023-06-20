#include "nbl/asset/utils/CDerivativeMapCreator.h"

#include "nbl/asset/filters/CSwizzleAndConvertImageFilter.h"
#include "nbl/asset/filters/CBlitImageFilter.h"
#include "nbl/asset/interchange/IImageAssetHandlerBase.h"

using namespace nbl;
using namespace nbl::asset;

template<bool isotropicNormalization>
core::smart_refctd_ptr<ICPUImage> CDerivativeMapCreator::createDerivativeMapFromHeightMap(ICPUImage* _inImg, ISampler::E_TEXTURE_CLAMP _uwrap, ISampler::E_TEXTURE_CLAMP _vwrap, ISampler::E_TEXTURE_BORDER_COLOR _borderColor, float* out_normalizationFactor)
{
	using namespace asset;

	// or Mitchell
	using ReconstructionFunction = CWeightFunction1D<SDiracFunction>;
	using DerivativeFunction = CWeightFunction1D<SGaussianFunction<>, 1>;

	using DerivativeMapFilter = CBlitImageFilter
	<
		StaticSwizzle<ICPUImageView::SComponentMapping::ES_R,ICPUImageView::SComponentMapping::ES_R>,
		IdentityDither,CDerivativeMapNormalizationState<isotropicNormalization>,true,
		CBlitUtilities<
			CChannelIndependentWeightFunction1D<
				CConvolutionWeightFunction1D<ReconstructionFunction, DerivativeFunction>,
				CConvolutionWeightFunction1D<ReconstructionFunction, CWeightFunction1D<SBoxFunction>>,
				CConvolutionWeightFunction1D<ReconstructionFunction, CWeightFunction1D<SBoxFunction>>,
				CConvolutionWeightFunction1D<ReconstructionFunction, CWeightFunction1D<SBoxFunction>>
			>,

			CChannelIndependentWeightFunction1D<
				CConvolutionWeightFunction1D<ReconstructionFunction, CWeightFunction1D<SBoxFunction>>,
				CConvolutionWeightFunction1D<ReconstructionFunction, DerivativeFunction>,
				CConvolutionWeightFunction1D<ReconstructionFunction, CWeightFunction1D<SBoxFunction>>,
				CConvolutionWeightFunction1D<ReconstructionFunction, CWeightFunction1D<SBoxFunction>>
			>,

			CChannelIndependentWeightFunction1D<
				CConvolutionWeightFunction1D<ReconstructionFunction, CWeightFunction1D<SBoxFunction>>,
				CConvolutionWeightFunction1D<ReconstructionFunction, CWeightFunction1D<SBoxFunction>>,
				CConvolutionWeightFunction1D<ReconstructionFunction, CWeightFunction1D<SBoxFunction>>,
				CConvolutionWeightFunction1D<ReconstructionFunction, CWeightFunction1D<SBoxFunction>>
			>
		>
	>;

	const auto extent = _inImg->getCreationParameters().extent;

	// derivative values should not change depending on resolution of the texture, so they need to be done w.r.t. normalized UV coordinates  
	DerivativeFunction derivX;
	derivX.scale(extent.width);

	DerivativeFunction derivY;
	derivY.scale(extent.height);

	const core::vectorSIMDu32 extent_vector(extent.width, extent.height, extent.depth);

	auto convolutionKernels = DerivativeMapFilter::blit_utils_t::getConvolutionKernels(
		extent_vector,
		extent_vector,

		ReconstructionFunction(), std::move(derivX),
		ReconstructionFunction(), CWeightFunction1D<SBoxFunction>(),
		ReconstructionFunction(), CWeightFunction1D<SBoxFunction>(),
		ReconstructionFunction(), CWeightFunction1D<SBoxFunction>(),

		ReconstructionFunction(), CWeightFunction1D<SBoxFunction>(),
		ReconstructionFunction(), std::move(derivY),
		ReconstructionFunction(), CWeightFunction1D<SBoxFunction>(),
		ReconstructionFunction(), CWeightFunction1D<SBoxFunction>(),

		ReconstructionFunction(), CWeightFunction1D<SBoxFunction>(),
		ReconstructionFunction(), CWeightFunction1D<SBoxFunction>(),
		ReconstructionFunction(), CWeightFunction1D<SBoxFunction>(),
		ReconstructionFunction(), CWeightFunction1D<SBoxFunction>());

	typename DerivativeMapFilter::state_type state(convolutionKernels);

	const auto& inParams = _inImg->getCreationParameters();
	auto outParams = inParams;
	outParams.format = getRGformat(outParams.format);
	const uint32_t pitch = IImageAssetHandlerBase::calcPitchInBlocks(outParams.extent.width, getTexelOrBlockBytesize(outParams.format));
	auto buffer = core::make_smart_refctd_ptr<ICPUBuffer>(getTexelOrBlockBytesize(outParams.format) * pitch * outParams.extent.height);
	ICPUImage::SBufferCopy region;
	region.imageOffset = { 0,0,0 };
	region.imageExtent = outParams.extent;
	region.imageSubresource.aspectMask = asset::IImage::EAF_COLOR_BIT;
	region.imageSubresource.baseArrayLayer = 0u;
	region.imageSubresource.layerCount = 1u;
	region.imageSubresource.mipLevel = 0u;
	region.bufferRowLength = pitch;
	region.bufferImageHeight = 0u;
	region.bufferOffset = 0u;
	auto outImg = ICPUImage::create(std::move(outParams));
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
	state.axisWraps[2] = ISampler::ETC_CLAMP_TO_EDGE;
	state.borderColor = _borderColor;
	state.scratchMemoryByteSize = DerivativeMapFilter::getRequiredScratchByteSize(&state);
	state.scratchMemory = reinterpret_cast<uint8_t*>(_NBL_ALIGNED_MALLOC(state.scratchMemoryByteSize, _NBL_SIMD_ALIGNMENT));

	if (!DerivativeMapFilter::blit_utils_t:: template computeScaledKernelPhasedLUT<float>(state.scratchMemory + DerivativeMapFilter::getScratchOffset(&state, DerivativeMapFilter::ESU_SCALED_KERNEL_PHASED_LUT), state.inExtentLayerCount, state.outExtentLayerCount, state.inImage->getCreationParameters().type, convolutionKernels))
		return nullptr;

	const bool result = DerivativeMapFilter::execute(core::execution::par_unseq,&state);
	if (result)
	{
		out_normalizationFactor[0] = state.normalization.maxAbsPerChannel[0];
		if constexpr (!isotropicNormalization)
			out_normalizationFactor[1] = state.normalization.maxAbsPerChannel[1];
	}

	_NBL_ALIGNED_FREE(state.scratchMemory);

	return outImg;
}

template<bool isotropicNormalization>
core::smart_refctd_ptr<ICPUImageView> CDerivativeMapCreator::createDerivativeMapViewFromHeightMap(ICPUImage* _inImg, ISampler::E_TEXTURE_CLAMP _uwrap, ISampler::E_TEXTURE_CLAMP _vwrap, ISampler::E_TEXTURE_BORDER_COLOR _borderColor, float* out_normalizationFactor)
{
	auto img = createDerivativeMapFromHeightMap<isotropicNormalization>(_inImg, _uwrap, _vwrap, _borderColor, out_normalizationFactor);
	const auto& iparams = img->getCreationParameters();

	ICPUImageView::SCreationParams params;
	params.format = iparams.format;
	params.subresourceRange.baseArrayLayer = 0u;
	params.subresourceRange.layerCount = iparams.arrayLayers;
	assert(params.subresourceRange.layerCount == 1u);
	params.subresourceRange.baseMipLevel = 0u;
	params.subresourceRange.levelCount = iparams.mipLevels;
	params.viewType = IImageView<ICPUImage>::ET_2D;
	params.flags = static_cast<IImageView<ICPUImage>::E_CREATE_FLAGS>(0);
	params.image = std::move(img);

	return ICPUImageView::create(std::move(params));
}

template<bool isotropicNormalization>
core::smart_refctd_ptr<ICPUImage> CDerivativeMapCreator::createDerivativeMapFromNormalMap(ICPUImage* _inImg, float* out_normalizationFactor)
{
	auto formatOverrideCreationParams = _inImg->getCreationParameters();
	assert(formatOverrideCreationParams.type == IImage::E_TYPE::ET_2D);
	// tools produce normalmaps with non SRGB encoding but use SRGB formats to store them (WTF!?)
	switch (formatOverrideCreationParams.format)
	{
		case EF_R8G8B8_SRGB:
			formatOverrideCreationParams.format = EF_R8G8B8_UNORM;
			break;
		case EF_R8G8B8A8_SRGB:
			formatOverrideCreationParams.format = EF_R8G8B8A8_UNORM;
			break;
		default:
			break;
	}
	auto newImageParams = formatOverrideCreationParams;
	newImageParams.format = getRGformat(newImageParams.format);

	auto cpuImageNormalTexture = ICPUImage::create(std::move(formatOverrideCreationParams));
	{
		const auto& referenceRegions = _inImg->getRegions();
		auto regionList = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<IImage::SBufferCopy>>(referenceRegions.size());
		std::copy(referenceRegions.begin(),referenceRegions.end(),regionList->data());
		cpuImageNormalTexture->setBufferAndRegions(
			core::smart_refctd_ptr<ICPUBuffer>(_inImg->getBuffer()),
			std::move(regionList)
		);
	}

	CNormalMapToDerivativeFilter<true> derivativeNormalFilter;
	decltype(derivativeNormalFilter)::state_type state;
	state.inOffset = { 0, 0, 0 };
	state.inBaseLayer = 0;
	state.outOffset = { 0, 0, 0 };
	state.outBaseLayer = 0;
	state.extent = { newImageParams.extent.width,newImageParams.extent.height,newImageParams.extent.depth };
	state.layerCount = newImageParams.arrayLayers;
	state.inMipLevel = 0;
	state.outMipLevel = 0;

	core::smart_refctd_ptr<ICPUImage> newDerivativeNormalMapImage;
	{
		const uint32_t pitch = IImageAssetHandlerBase::calcPitchInBlocks(newImageParams.extent.width,getTexelOrBlockBytesize(newImageParams.format));
		core::smart_refctd_ptr<ICPUBuffer> newCpuBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(getTexelOrBlockBytesize(newImageParams.format) * pitch * newImageParams.extent.height);

		ICPUImage::SBufferCopy region;
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
	}

	state.inImage = cpuImageNormalTexture.get();
	state.outImage = newDerivativeNormalMapImage.get();
	const bool result = derivativeNormalFilter.execute(&state);
	if (result)
	{
		out_normalizationFactor[0] = state.normalization.maxAbsPerChannel[0];
		if (!isotropicNormalization)
			out_normalizationFactor[1] = state.normalization.maxAbsPerChannel[1];
	}
	else
	{
		_NBL_DEBUG_BREAK_IF(true);
		// TODO: use logger
		// os::Printer::log("Something went wrong while performing derivative filter operations!", ELL_ERROR);
		return nullptr;
	}

	return newDerivativeNormalMapImage;
}

template<bool isotropicNormalization>
core::smart_refctd_ptr<ICPUImageView> CDerivativeMapCreator::createDerivativeMapViewFromNormalMap(ICPUImage* _inImg, float* out_normalizationFactor)
{
	auto cpuDerivativeImage = createDerivativeMapFromNormalMap<isotropicNormalization>(_inImg,out_normalizationFactor);

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
	{
		_NBL_DEBUG_BREAK_IF(true);
		// TODO: use logger
		// os::Printer::log("Something went wrong while creating image view for derivative normal map!", ELL_ERROR);
	}

	return imageView;
}


//explicit instantiation
template core::smart_refctd_ptr<ICPUImageView> CDerivativeMapCreator::createDerivativeMapViewFromHeightMap<false>(ICPUImage* _inImg, ISampler::E_TEXTURE_CLAMP _uwrap, ISampler::E_TEXTURE_CLAMP _vwrap, ISampler::E_TEXTURE_BORDER_COLOR _borderColor, float* out_normalizationFactor);
template core::smart_refctd_ptr<ICPUImageView> CDerivativeMapCreator::createDerivativeMapViewFromHeightMap<true>(ICPUImage* _inImg, ISampler::E_TEXTURE_CLAMP _uwrap, ISampler::E_TEXTURE_CLAMP _vwrap, ISampler::E_TEXTURE_BORDER_COLOR _borderColor, float* out_normalizationFactor);
template core::smart_refctd_ptr<ICPUImageView> CDerivativeMapCreator::createDerivativeMapViewFromNormalMap<false>(ICPUImage* _inImg, float* out_normalizationFactor);
template core::smart_refctd_ptr<ICPUImageView> CDerivativeMapCreator::createDerivativeMapViewFromNormalMap<true>(ICPUImage* _inImg, float* out_normalizationFactor);