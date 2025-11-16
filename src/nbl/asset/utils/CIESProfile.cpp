#include "CIESProfile.h"

#include <atomic>
#include "nbl/asset/filters/CBasicImageFilterCommon.h"
#include "nbl/builtin/hlsl/math/octahedral.hlsl"
#include "nbl/builtin/hlsl/math/polar.hlsl"

using namespace nbl;
using namespace asset;

template<class ExecutionPolicy>
core::smart_refctd_ptr<asset::ICPUImageView> CIESProfile::createIESTexture(ExecutionPolicy&& policy, const float flatten, const bool fullDomainFlatten, uint32_t width, uint32_t height) const
{
    const bool inFlattenDomain = flatten >= 0.0 && flatten <= 1.0; // [0, 1] range for blend equation, 1 is normally invalid but we use it to for special implied domain flatten mode
    assert(inFlattenDomain);

    if (width > properties_t::CDC_MAX_TEXTURE_WIDTH)
        width = properties_t::CDC_MAX_TEXTURE_WIDTH;

    if (height > properties_t::CDC_MAX_TEXTURE_HEIGHT)
        height = properties_t::CDC_MAX_TEXTURE_HEIGHT;

    // TODO: If no symmetry (no folding in half and abuse of mirror sampler) make dimensions odd-sized so middle texel taps the south pole

    // TODO: This is hack because the mitsuba loader and its material compiler use Virtual Texturing, and there's some bug with IES not sampling sub 128x128 mip levels
    // don't want to spend time to fix this since we'll be using descriptor indexing for the next iteration
    width = core::max(width,128);
    height = core::max(height,128);

    asset::ICPUImage::SCreationParams imgInfo;
    imgInfo.type = asset::ICPUImage::ET_2D;
    imgInfo.extent.width = width;
    imgInfo.extent.height = height;
    imgInfo.extent.depth = 1u;
    imgInfo.mipLevels = 1u;
    imgInfo.arrayLayers = 1u;
    imgInfo.samples = asset::ICPUImage::ESCF_1_BIT;
    imgInfo.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
    imgInfo.format = properties_t::IES_TEXTURE_STORAGE_FORMAT;
    auto outImg = asset::ICPUImage::create(std::move(imgInfo));

    asset::ICPUImage::SBufferCopy region;
    constexpr auto texelBytesz = asset::getTexelOrBlockBytesize<properties_t::IES_TEXTURE_STORAGE_FORMAT>();
    const size_t bufferRowLength = asset::IImageAssetHandlerBase::calcPitchInBlocks(width, texelBytesz);
    region.bufferRowLength = bufferRowLength;
    region.imageExtent = imgInfo.extent;
    region.imageSubresource.baseArrayLayer = 0u;
    region.imageSubresource.layerCount = 1u;
    region.imageSubresource.mipLevel = 0u;
    region.imageSubresource.aspectMask = core::bitflag(asset::IImage::EAF_COLOR_BIT);
    region.bufferImageHeight = 0u;
    region.bufferOffset = 0u;

    asset::ICPUBuffer::SCreationParams bParams;
    bParams.size = texelBytesz * bufferRowLength * height;
    auto buffer = asset::ICPUBuffer::create(std::move(bParams));

    if (!outImg->setBufferAndRegions(std::move(buffer), core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::IImage::SBufferCopy>>(1ull, region)))
        return {};

    //! Generate 2D IES grid data CandelaPower Distribution Curve texture can be created from
    {
        const auto& creationParams = outImg->getCreationParameters();

        CFillImageFilter::state_type state;
        state.outImage = outImg.get();
        state.subresource.aspectMask = core::bitflag(asset::IImage::EAF_COLOR_BIT);
        state.subresource.baseArrayLayer = 0u;
        state.subresource.layerCount = 1u;
        state.outRange.extent = creationParams.extent;

        const IImageFilter::IState::ColorValue::WriteMemoryInfo wInfo(creationParams.format, outImg->getBuffer()->getPointer());

        // Late Optimization TODO: Modify the Max Value for the UNORM texture to be the Max Value after flatten blending 
        const auto maxValue = accessor.properties.maxCandelaValue;
        const auto maxValueRecip = 1.f / maxValue;

        // There is one huge issue, the IES files love to give us values for degrees 0, 90, 180 an 360
        // So standard octahedral mapping won't work, because for above data points you need corner sampled images.
        const float vertInv = 1.0 / (height-1);
        const float horiInv = 1.0 / (width-1);

        const double flattenTarget = getAvgEmmision(fullDomainFlatten);
        const double domainLo = core::radians(accessor.vAngles.front());
        const double domainHi = core::radians(accessor.vAngles.back());
        auto fill = [&](uint32_t blockArrayOffset, core::vectorSIMDu32 position) -> void
        {
            // We don't currently support generating IES images that exploit symmetries or reduced domains, all are full octahederal mappings of a sphere.
            // If we did, we'd rely on MIRROR and CLAMP samplers to do some of the work for us while handling the discontinuity due to corner sampling. 
            
            using Octahedral = hlsl::math::OctahedralTransform<hlsl::float32_t>;
            using Polar = hlsl::math::Polar<hlsl::float32_t>;
            const auto uv = Octahedral::vector2_type(position.x * vertInv, position.y * horiInv);
            const auto dir = Octahedral::uvToDir(uv);
            const auto polar = Polar::createFromCartesian(dir);
            const auto intensity = sampler_t::sample(accessor, polar);

            //! blend the IES texture with "flatten"
            float blendV = intensity * (1.f - flatten);
            if (fullDomainFlatten && domainLo<= polar.theta && polar.theta<=domainHi || intensity >0.0)
                blendV += flattenTarget * flatten;

            blendV *= maxValueRecip;

            asset::IImageFilter::IState::ColorValue color;
            //asset::encodePixels<CIESProfile::IES_TEXTURE_STORAGE_FORMAT>(color.asDouble, &blendV); TODO: FIX THIS ENCODE, GIVES ARTIFACTS
            constexpr float UI16_MAX_D = static_cast<float>(std::numeric_limits<std::uint16_t>::max());
            const uint16_t encodeV = static_cast<uint16_t>(std::clamp(blendV * UI16_MAX_D + 0.5f, 0.f, UI16_MAX_D));
            *color.asUShort = encodeV;
            color.writeMemory(wInfo, blockArrayOffset);
        };

        CBasicImageFilterCommon::clip_region_functor_t clip(state.subresource, state.outRange, creationParams.format);
        const auto& regions = outImg->getRegions(state.subresource.mipLevel);
        CBasicImageFilterCommon::executePerRegion(std::forward<ExecutionPolicy>(policy), outImg.get(), fill, regions, clip);
    }

    ICPUImageView::SCreationParams viewParams = {};
    viewParams.image = outImg;
    viewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0);
    viewParams.viewType = IImageView<ICPUImage>::ET_2D;
    viewParams.format = viewParams.image->getCreationParameters().format;
    viewParams.subresourceRange.aspectMask = core::bitflag(asset::IImage::EAF_COLOR_BIT);
    viewParams.subresourceRange.levelCount = viewParams.image->getCreationParameters().mipLevels;
    viewParams.subresourceRange.layerCount = 1u;

    auto imageView = ICPUImageView::create(std::move(viewParams));
    return core::smart_refctd_ptr(imageView);
}

//! Explicit instantiations
template core::smart_refctd_ptr<asset::ICPUImageView> CIESProfile::createIESTexture(const std::execution::sequenced_policy&, const float, const bool, uint32_t, uint32_t) const;
template core::smart_refctd_ptr<asset::ICPUImageView> CIESProfile::createIESTexture(const std::execution::parallel_policy&, const float, const bool, uint32_t, uint32_t) const;
template core::smart_refctd_ptr<asset::ICPUImageView> CIESProfile::createIESTexture(const std::execution::parallel_unsequenced_policy&, const float, const bool, uint32_t, uint32_t) const;

core::smart_refctd_ptr<asset::ICPUImageView> CIESProfile::createIESTexture(const float flatten, const bool fullDomainFlatten, uint32_t width, uint32_t height) const
{
    return createIESTexture(std::execution::seq, flatten, fullDomainFlatten, width, height);
}