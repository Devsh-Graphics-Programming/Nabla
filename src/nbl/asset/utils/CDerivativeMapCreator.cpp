#include "nbl/asset/utils/CDerivativeMapCreator.h"

#include "nbl/asset/filters/CSwizzleAndConvertImageFilter.h"
#include "nbl/asset/filters/kernels/CChannelIndependentImageFilterKernel.h"
#include "nbl/asset/filters/kernels/CDerivativeImageFilterKernel.h"
#include "nbl/asset/filters/kernels/CGaussianImageFilterKernel.h"
#include "nbl/asset/filters/CBlitImageFilter.h"
#include "nbl/asset/interchange/IImageAssetHandlerBase.h"

using namespace nbl;
using namespace nbl::asset;

namespace
{
template<class Kernel>
class MyKernel : public CFloatingPointSeparableImageFilterKernelBase<MyKernel<Kernel>>
{
    using Base = CFloatingPointSeparableImageFilterKernelBase<MyKernel<Kernel>>;

    Kernel kernel;
    float multiplier;

public:
    using value_type = typename Base::value_type;

    MyKernel(Kernel&& k, uint32_t _imgExtent)
        : Base(k.negative_support.x, k.positive_support.x), kernel(std::move(k)), multiplier(float(_imgExtent)) {}

    // no special user data by default
    inline const IImageFilterKernel::UserData* getUserData() const { return nullptr; }

    inline float weight(float x, int32_t channel) const
    {
        return kernel.weight(x, channel) * multiplier;
    }

    // we need to ensure to override the default behaviour of `CFloatingPointSeparableImageFilterKernelBase` which applies the weight along every axis
    template<class PreFilter, class PostFilter>
    struct sample_functor_t
    {
        sample_functor_t(const MyKernel* _this, PreFilter& _preFilter, PostFilter& _postFilter)
            : _this(_this), preFilter(_preFilter), postFilter(_postFilter) {}

        inline void operator()(value_type* windowSample, core::vectorSIMDf& relativePos, const core::vectorSIMDi32& globalTexelCoord, const IImageFilterKernel::UserData* userData)
        {
            preFilter(windowSample, relativePos, globalTexelCoord, userData);
            auto* scale = IImageFilterKernel::ScaleFactorUserData::cast(userData);
            for(int32_t i = 0; i < MaxChannels; i++)
            {
                // this differs from the `CFloatingPointSeparableImageFilterKernelBase`
                windowSample[i] *= _this->weight(relativePos.x, i);
                if(scale)
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
class SeparateOutXAxisKernel : public CFloatingPointSeparableImageFilterKernelBase<SeparateOutXAxisKernel<Kernel>>
{
    using Base = CFloatingPointSeparableImageFilterKernelBase<SeparateOutXAxisKernel<Kernel>>;

    Kernel kernel;

public:
    // passthrough everything
    using value_type = typename Kernel::value_type;

    _NBL_STATIC_INLINE_CONSTEXPR auto MaxChannels = Kernel::MaxChannels;  // derivative map only needs 2 channels

    SeparateOutXAxisKernel(Kernel&& k)
        : Base(k.negative_support.x, k.positive_support.x), kernel(std::move(k)) {}

    NBL_DECLARE_DEFINE_CIMAGEFILTER_KERNEL_PASS_THROUGHS(Base)

    // we need to ensure to override the default behaviour of `CFloatingPointSeparableImageFilterKernelBase` which applies the weight along every axis
    template<class PreFilter, class PostFilter>
    struct sample_functor_t
    {
        sample_functor_t(const SeparateOutXAxisKernel<Kernel>* _this, PreFilter& _preFilter, PostFilter& _postFilter)
            : _this(_this), preFilter(_preFilter), postFilter(_postFilter) {}

        inline void operator()(value_type* windowSample, core::vectorSIMDf& relativePos, const core::vectorSIMDi32& globalTexelCoord, const IImageFilterKernel::UserData* userData)
        {
            preFilter(windowSample, relativePos, globalTexelCoord, userData);
            auto* scale = IImageFilterKernel::ScaleFactorUserData::cast(userData);
            for(int32_t i = 0; i < MaxChannels; i++)
            {
                // this differs from the `CFloatingPointSeparableImageFilterKernelBase`
                windowSample[i] *= _this->kernel.weight(relativePos.x, i);
                if(scale)
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

}

template<bool isotropicNormalization>
core::smart_refctd_ptr<ICPUImage> CDerivativeMapCreator::createDerivativeMapFromHeightMap(ICPUImage* _inImg, ISampler::E_TEXTURE_CLAMP _uwrap, ISampler::E_TEXTURE_CLAMP _vwrap, ISampler::E_TEXTURE_BORDER_COLOR _borderColor, float* out_normalizationFactor)
{
    using namespace asset;

    using ReconstructionKernel = CGaussianImageFilterKernel<>;  // or Mitchell
    using DerivKernel_ = CDerivativeImageFilterKernel<ReconstructionKernel>;
    using DerivKernel = MyKernel<DerivKernel_>;
    using XDerivKernel_ = CChannelIndependentImageFilterKernel<DerivKernel, CBoxImageFilterKernel>;
    using YDerivKernel_ = CChannelIndependentImageFilterKernel<CBoxImageFilterKernel, DerivKernel>;
    using XDerivKernel = SeparateOutXAxisKernel<XDerivKernel_>;
    using YDerivKernel = SeparateOutXAxisKernel<YDerivKernel_>;
    using DerivativeMapFilter = CBlitImageFilter<
        StaticSwizzle<ICPUImageView::SComponentMapping::ES_R, ICPUImageView::SComponentMapping::ES_R>,
        IdentityDither, CDerivativeMapNormalizationState<isotropicNormalization>, true,
        XDerivKernel, YDerivKernel, CBoxImageFilterKernel>;

    const auto extent = _inImg->getCreationParameters().extent;
    // derivative values should not change depending on resolution of the texture, so they need to be done w.r.t. normalized UV coordinates
    XDerivKernel xderiv(XDerivKernel_(DerivKernel(DerivKernel_(ReconstructionKernel()), extent.width), CBoxImageFilterKernel()));
    YDerivKernel yderiv(YDerivKernel_(CBoxImageFilterKernel(), DerivKernel(DerivKernel_(ReconstructionKernel()), extent.height)));

    DerivativeMapFilter::state_type state(std::move(xderiv), std::move(yderiv), CBoxImageFilterKernel());

    const auto& inParams = _inImg->getCreationParameters();
    auto outParams = inParams;
    outParams.format = getRGformat(outParams.format);
    const uint32_t pitch = IImageAssetHandlerBase::calcPitchInBlocks(outParams.extent.width, getTexelOrBlockBytesize(outParams.format));
    auto buffer = core::make_smart_refctd_ptr<ICPUBuffer>(getTexelOrBlockBytesize(outParams.format) * pitch * outParams.extent.height);
    ICPUImage::SBufferCopy region;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = outParams.extent;
    region.imageSubresource.baseArrayLayer = 0u;
    region.imageSubresource.layerCount = 1u;
    region.imageSubresource.mipLevel = 0u;
    region.bufferRowLength = pitch;
    region.bufferImageHeight = 0u;
    region.bufferOffset = 0u;
    auto outImg = ICPUImage::create(std::move(outParams));
    outImg->setBufferAndRegions(std::move(buffer), core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<IImage::SBufferCopy>>(1ull, region));

    state.inOffset = {0, 0, 0};
    state.inBaseLayer = 0u;
    state.outOffset = {0, 0, 0};
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

    const bool result = DerivativeMapFilter::execute(std::execution::par_unseq, &state);
    if(result)
    {
        out_normalizationFactor[0] = state.normalization.maxAbsPerChannel[0];
        if constexpr(!isotropicNormalization)
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
    switch(formatOverrideCreationParams.format)
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
        std::copy(referenceRegions.begin(), referenceRegions.end(), regionList->data());
        cpuImageNormalTexture->setBufferAndRegions(
            core::smart_refctd_ptr<ICPUBuffer>(_inImg->getBuffer()),
            std::move(regionList));
    }

    CNormalMapToDerivativeFilter<true> derivativeNormalFilter;
    decltype(derivativeNormalFilter)::state_type state;
    state.inOffset = {0, 0, 0};
    state.inBaseLayer = 0;
    state.outOffset = {0, 0, 0};
    state.outBaseLayer = 0;
    state.extent = {newImageParams.extent.width, newImageParams.extent.height, newImageParams.extent.depth};
    state.layerCount = newImageParams.arrayLayers;
    state.inMipLevel = 0;
    state.outMipLevel = 0;

    core::smart_refctd_ptr<ICPUImage> newDerivativeNormalMapImage;
    {
        const uint32_t pitch = IImageAssetHandlerBase::calcPitchInBlocks(newImageParams.extent.width, getTexelOrBlockBytesize(newImageParams.format));
        core::smart_refctd_ptr<ICPUBuffer> newCpuBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(getTexelOrBlockBytesize(newImageParams.format) * pitch * newImageParams.extent.height);

        ICPUImage::SBufferCopy region;
        region.imageOffset = {0, 0, 0};
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
    if(result)
    {
        out_normalizationFactor[0] = state.normalization.maxAbsPerChannel[0];
        if(!isotropicNormalization)
            out_normalizationFactor[1] = state.normalization.maxAbsPerChannel[1];
    }
    else
    {
        os::Printer::log("Something went wrong while performing derivative filter operations!", ELL_ERROR);
        return nullptr;
    }

    return newDerivativeNormalMapImage;
}

template<bool isotropicNormalization>
core::smart_refctd_ptr<ICPUImageView> CDerivativeMapCreator::createDerivativeMapViewFromNormalMap(ICPUImage* _inImg, float* out_normalizationFactor)
{
    auto cpuDerivativeImage = createDerivativeMapFromNormalMap<isotropicNormalization>(_inImg, out_normalizationFactor);

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

    if(!imageView.get())
        os::Printer::log("Something went wrong while creating image view for derivative normal map!", ELL_ERROR);

    return imageView;
}

//explicit instantiation
template core::smart_refctd_ptr<ICPUImageView> CDerivativeMapCreator::createDerivativeMapViewFromHeightMap<false>(ICPUImage* _inImg, ISampler::E_TEXTURE_CLAMP _uwrap, ISampler::E_TEXTURE_CLAMP _vwrap, ISampler::E_TEXTURE_BORDER_COLOR _borderColor, float* out_normalizationFactor);
template core::smart_refctd_ptr<ICPUImageView> CDerivativeMapCreator::createDerivativeMapViewFromHeightMap<true>(ICPUImage* _inImg, ISampler::E_TEXTURE_CLAMP _uwrap, ISampler::E_TEXTURE_CLAMP _vwrap, ISampler::E_TEXTURE_BORDER_COLOR _borderColor, float* out_normalizationFactor);
template core::smart_refctd_ptr<ICPUImageView> CDerivativeMapCreator::createDerivativeMapViewFromNormalMap<false>(ICPUImage* _inImg, float* out_normalizationFactor);
template core::smart_refctd_ptr<ICPUImageView> CDerivativeMapCreator::createDerivativeMapViewFromNormalMap<true>(ICPUImage* _inImg, float* out_normalizationFactor);