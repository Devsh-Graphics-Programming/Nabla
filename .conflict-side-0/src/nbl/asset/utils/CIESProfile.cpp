#include "CIESProfile.h"

#include <atomic>
#include "nbl/asset/filters/CBasicImageFilterCommon.h"

using namespace nbl;
using namespace asset;

template<class ExecutionPolicy>
core::smart_refctd_ptr<asset::ICPUImageView> CIESProfile::createIESTexture(ExecutionPolicy&& policy, hlsl::uint32_t2 resolution) const
{
    uint32_t width = resolution.x;
    uint32_t height = resolution.y;

    if (width > texture_t::MaxTextureWidth)
        width = texture_t::MaxTextureWidth;

    if (height > texture_t::MaxTextureHeight)
        height = texture_t::MaxTextureHeight;

    width = core::max(width, texture_t::MinTextureWidth);
    height = core::max(height, texture_t::MinTextureHeight);

    auto makeOdd = [](uint32_t value, const uint32_t maxValue) -> uint32_t
    {
        if (value & 1u)
            return value;
        return (value < maxValue) ? (value + 1u) : (value - 1u);
    };
    // TODO: remove this once we exploit symmetries and fold the domain.
    width = makeOdd(width, texture_t::MaxTextureWidth);
    height = makeOdd(height, texture_t::MaxTextureHeight);

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
		const auto texture = texture_t::create(accessor.properties.maxCandelaValue, hlsl::uint32_t2(width, height));

        auto fill = [&](uint32_t blockArrayOffset, core::vectorSIMDu32 position) -> void
        {
            const auto texel = texture.__call(accessor, hlsl::uint32_t2(position.x, position.y));

            asset::IImageFilter::IState::ColorValue color;
            constexpr float UI16_MAX_D = static_cast<float>(std::numeric_limits<std::uint16_t>::max());
            const uint16_t encodeV = static_cast<uint16_t>(std::clamp(texel * UI16_MAX_D + 0.5f, 0.f, UI16_MAX_D)); // TODO: use asset::encodePixels when its fixed (no artifacts)
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
template core::smart_refctd_ptr<asset::ICPUImageView> CIESProfile::createIESTexture(const std::execution::sequenced_policy&, hlsl::uint32_t2) const;
template core::smart_refctd_ptr<asset::ICPUImageView> CIESProfile::createIESTexture(const std::execution::parallel_policy&, hlsl::uint32_t2) const;
template core::smart_refctd_ptr<asset::ICPUImageView> CIESProfile::createIESTexture(const std::execution::parallel_unsequenced_policy&, hlsl::uint32_t2) const;

core::smart_refctd_ptr<asset::ICPUImageView> CIESProfile::createIESTexture(hlsl::uint32_t2 resolution) const
{
    return createIESTexture(std::execution::seq, resolution);
}
