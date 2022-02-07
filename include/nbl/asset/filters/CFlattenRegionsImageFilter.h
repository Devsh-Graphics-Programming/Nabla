// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_FLATTEN_REGIONS_IMAGE_FILTER_H_INCLUDED__
#define __NBL_ASSET_C_FLATTEN_REGIONS_IMAGE_FILTER_H_INCLUDED__

#include "nbl/core/declarations.h"

#include <type_traits>

#include "nbl/asset/filters/CFillImageFilter.h"
#include "nbl/asset/filters/CCopyImageFilter.h"

namespace nbl
{
namespace asset
{
// respecifies the image in terms of the least amount of region entries
class CFlattenRegionsImageFilter : public CImageFilter<CFlattenRegionsImageFilter>, public CBasicImageFilterCommon
{
public:
    virtual ~CFlattenRegionsImageFilter() {}

    class CState : public IImageFilter::IState
    {
    public:
        virtual ~CState() {}

        const ICPUImage* inImage = nullptr;
        core::smart_refctd_ptr<ICPUImage> outImage = nullptr;  //!< outImage pointer might change after execution, \bcan be null\b, we'll just make a new texture
        bool preFill = true;  //!< state whether to fill values using fillValue if there is a pixel and any region doesn't cover it with. If false - copy filter will be executed.
        IImageFilter::IState::ColorValue fillValue;  //! values for a pixel for which any region doesn't cover it with
    };
    using state_type = CState;

    static inline bool validate(state_type* state)
    {
        if(!state)
            return false;

        const auto& inParams = state->inImage->getCreationParameters();
        // TODO: remove this later when we can actually handle multi samples
        if(inParams.samples != IImage::ESCF_1_BIT)
            return false;

        // TODO: remove this later when we can actually write/encode to block formats
        if(isBlockCompressionFormat(inParams.format))
            return false;

        return true;
    }

    template<class ExecutionPolicy>
    static inline bool execute(ExecutionPolicy&& policy, state_type* state)
    {
        if(!validate(state))
            return false;

        auto* const inImg = state->inImage;
        const auto& inParams = inImg->getCreationParameters();
        auto respecifyRegions = [&state, &inImg, &inParams]() -> void {
            state->outImage = ICPUImage::create(IImage::SCreationParams(inParams));
            auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<IImage::SBufferCopy> >(inParams.mipLevels);
            size_t bufferSize = 0ull;
            const TexelBlockInfo info(inParams.format);
            const core::rational<size_t> bytesPerPixel = state->outImage->getBytesPerPixel();
            for(auto rit = regions->begin(); rit != regions->end(); rit++)
            {
                auto mipLevel = static_cast<uint32_t>(std::distance(regions->begin(), rit));
                auto localExtent = inImg->getMipSize(mipLevel);
                rit->bufferOffset = bufferSize;
                rit->bufferRowLength = localExtent.x;  // could round up to multiple of 8 bytes in the future
                rit->bufferImageHeight = localExtent.y;
                rit->imageSubresource.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0u);
                rit->imageSubresource.mipLevel = mipLevel;
                rit->imageSubresource.baseArrayLayer = 0u;
                rit->imageSubresource.layerCount = inParams.arrayLayers;
                rit->imageOffset = {0u, 0u, 0u};
                rit->imageExtent = {localExtent.x, localExtent.y, localExtent.z};
                auto levelSize = info.roundToBlockSize(localExtent);
                auto memsize = size_t(levelSize[0] * levelSize[1]) * size_t(levelSize[2] * inParams.arrayLayers) * bytesPerPixel;
                assert(memsize.getNumerator() % memsize.getDenominator() == 0u);
                bufferSize += memsize.getIntegerApprox();
            }
            auto buffer = core::make_smart_refctd_ptr<ICPUBuffer>(bufferSize);
            state->outImage->setBufferAndRegions(std::move(buffer), std::move(regions));
        };
        auto* outImg = state->outImage.get();
        if(outImg)
        {
            const auto& outParams = outImg->getCreationParameters();
            if(outParams.type == inParams.type &&
                outParams.samples == inParams.samples &&
                getFormatClass(outParams.format) == getFormatClass(inParams.format) &&
                (core::vectorSIMDu32(outParams.extent.width, outParams.extent.height, outParams.extent.depth, outParams.arrayLayers) ==
                    core::vectorSIMDu32(inParams.extent.width, inParams.extent.height, inParams.extent.depth, inParams.arrayLayers))
                    .all() &&
                outParams.mipLevels == inParams.mipLevels)
            {
                auto regions = outImg->getRegions();
                bool mipPresent[16u] = {false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false};
                for(auto rit = regions.begin(); rit != regions.end(); rit++)
                {
                    auto mipLevel = rit->imageSubresource.mipLevel;
                    assert(mipLevel <= 16u);
                    if(mipPresent[mipLevel])
                    {
                        respecifyRegions();
                        break;
                    }
                    mipPresent[mipLevel] = true;

                    auto localExtent = outImg->getMipSize(mipLevel);
                    if(rit->imageSubresource.baseArrayLayer ||
                        rit->imageSubresource.layerCount != inParams.arrayLayers ||
                        rit->imageOffset != VkOffset3D{0u, 0u, 0u} ||
                        rit->imageExtent != VkExtent3D{localExtent.x, localExtent.y, localExtent.z})
                    {
                        respecifyRegions();
                        break;
                    }
                }
            }
            else
                respecifyRegions();
        }
        else
            respecifyRegions();

        outImg = state->outImage.get();
        auto regions = outImg->getRegions();
        for(auto rit = regions.begin(); rit != regions.end(); rit++)
        {
            // fill
            if(state->preFill)
            {
                CFillImageFilter::state_type fill;
                fill.subresource = rit->imageSubresource;
                fill.outRange = {{0u, 0u, 0u}, rit->imageExtent};
                fill.outImage = outImg;
                fill.fillValue = state->fillValue;
                if(!CFillImageFilter::execute(&fill))
                    return false;
            }
            // copy
            CCopyImageFilter::state_type copy;
            copy.extent = rit->imageExtent;
            copy.layerCount = rit->imageSubresource.layerCount;
            copy.inOffsetBaseLayer = core::vectorSIMDu32(0, 0, 0, 0);
            copy.outOffsetBaseLayer = core::vectorSIMDu32(0, 0, 0, 0);
            copy.inMipLevel = rit->imageSubresource.mipLevel;
            copy.outMipLevel = rit->imageSubresource.mipLevel;
            copy.inImage = inImg;
            copy.outImage = outImg;
            if(!CCopyImageFilter::execute(policy, &copy))
                return false;
        }
        return true;
    }
    static inline bool execute(state_type* state)
    {
        return execute(core::execution::seq, state);
    }
};

}  // end namespace asset
}  // end namespace nbl

#endif