// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_IMAGE_ASSET_HANDLER_BASE_H_INCLUDED__
#define __NBL_ASSET_I_IMAGE_ASSET_HANDLER_BASE_H_INCLUDED__

#include "nbl/core/declarations.h"

#include "nbl/system/ILogger.h"

#include "nbl/asset/filters/CCopyImageFilter.h"
#include "nbl/asset/filters/CSwizzleAndConvertImageFilter.h"

namespace nbl
{
namespace asset
{
class IImageAssetHandlerBase : public virtual core::IReferenceCounted
{
protected:
    IImageAssetHandlerBase() {}
    virtual ~IImageAssetHandlerBase() = 0;

public:
    static const uint32_t MAX_PITCH_ALIGNMENT = 8u;

    /**
			 Returns pitch for buffer row lenght, because
			 OpenGL cannot transfer rows with arbitrary padding
		*/
    static inline uint32_t calcPitchInBlocks(uint32_t width, uint32_t blockByteSize)
    {
        auto rowByteSize = width * blockByteSize;
        for(uint32_t _alignment = MAX_PITCH_ALIGNMENT; _alignment > 1u; _alignment >>= 1u)
        {
            auto paddedSize = core::alignUp(rowByteSize, _alignment);
            if(paddedSize % blockByteSize)
                continue;
            return paddedSize / blockByteSize;
        }
        return width;
    }

    static inline core::vector3du32_SIMD calcPitchInBlocks(uint32_t width, uint32_t height, uint32_t depth, uint32_t blockByteSize)
    {
        return core::vector3du32_SIMD(calcPitchInBlocks(width, blockByteSize), height, depth);
    }

    /*
			Create a new image with only one top level region, one layer and one mip-map level.
			Handling ordinary images in asset writing process is a mess since multi-regions
			are valid. To avoid ambitious, the function will handle top level data from
			image view to save only stuff a user has choosen. You may also specify extra
			output format the top image data will be converted to, but if you leave it, there
			will be no conversion provided.

			@param imageView entry image view an image with top data will be gained thanks to it
			@param arrayLayersMax layers count, only GLI should set different array layers max values
			@param mipLevelMax layers count, only GLI should set different mip level max values
		*/

    template<asset::E_FORMAT outFormat = asset::EF_UNKNOWN>
    static inline core::smart_refctd_ptr<ICPUImage> createImageDataForCommonWriting(const ICPUImageView* imageView, const system::logger_opt_ptr logger, uint32_t arrayLayersMax = 1, uint32_t mipLevelMax = 1)
    {
        const auto& viewParams = imageView->getCreationParameters();
        const auto& subresource = viewParams.subresourceRange;

        auto finalFormat = (outFormat == asset::EF_UNKNOWN ? viewParams.format : outFormat);

        const auto referenceImage = viewParams.image;
        const auto& referenceImageParams = referenceImage->getCreationParameters();

        core::smart_refctd_ptr<ICPUImage> newImage;
        auto newArrayLayers = core::min(subresource.layerCount, arrayLayersMax);
        auto newMipCount = core::min(subresource.levelCount, mipLevelMax);
        {
            auto newImageParams = referenceImageParams;
            newImageParams.format = finalFormat;
            newImageParams.arrayLayers = newArrayLayers;
            newImageParams.mipLevels = newMipCount;
            // you don't want to change the type of the texture, will backfire
            newImage = ICPUImage::create(std::move(newImageParams));

            auto newRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(newImageParams.mipLevels);
            size_t bufferSize = 0u;
            const TexelBlockInfo info(newImageParams.format);
            const core::rational<size_t> bytesPerPixel = asset::getBytesPerPixel(newImageParams.format);
            for(auto i = 0; i < newMipCount; i++)
            {
                auto& region = newRegions->operator[](i);
                region.bufferOffset = bufferSize;
                region.imageSubresource.mipLevel = i;
                region.imageSubresource.baseArrayLayer = 0;
                region.imageSubresource.layerCount = newImageParams.arrayLayers;
                // region.imageOffset is 0,0,0 by default
                auto mipSize = newImage->getMipSize(i);
                region.imageExtent = reinterpret_cast<const VkExtent3D&>(mipSize);

                auto levelSize = info.roundToBlockSize(mipSize);
                // don't worry about alignment and stuff, the CPU code can handle it just fine, could have set the thing to 0, but you use bufferRowLength in your code, you should assrt its not 0
                region.bufferRowLength = levelSize.x;
                region.bufferImageHeight = levelSize.y;

                auto memsize = size_t(levelSize[0] * levelSize[1]) * size_t(levelSize[2] * newImageParams.arrayLayers) * bytesPerPixel;
                assert(memsize.getNumerator() % memsize.getDenominator() == 0u);
                bufferSize += memsize.getIntegerApprox();
            }

            auto texelBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(bufferSize);

            newImage->setBufferAndRegions(std::move(texelBuffer), newRegions);
        }

        using COPY_FILTER = asset::CCopyImageFilter;
        using CONVERSION_FILTER = asset::CSwizzleAndConvertImageFilter<EF_UNKNOWN, EF_UNKNOWN, DefaultSwizzle, IdentityDither /*TODO: Blue noise*/, void, true>;

        bool identityTransform = viewParams.format == finalFormat;
        for(auto i = 0; i < asset::getFormatChannelCount(outFormat); i++)
        {
            auto mapping = (&viewParams.components.r)[i];
            identityTransform = identityTransform && (mapping == (decltype(mapping)::ES_R + i) || mapping == (decltype(mapping)::ES_IDENTITY));
        }

        for(auto i = 0; i < newMipCount; i++)
        {
            auto fillCommonState = [&](auto& state) {
                state.inImage = referenceImage.get();
                state.outImage = newImage.get();
                state.inOffset = {0, 0, 0};
                state.inBaseLayer = subresource.baseArrayLayer;
                state.outOffset = {0, 0, 0};
                state.outBaseLayer = 0;
                auto extent = newImage->getMipSize(i);
                state.extent = reinterpret_cast<const VkExtent3D&>(extent);
                state.layerCount = newArrayLayers;
                state.inMipLevel = subresource.baseMipLevel + i;
                state.outMipLevel = i;
            };

            // if texel block data does not need changing, we're good
            if(identityTransform)  // TODO: why do we even copy!?
            {
                COPY_FILTER::state_type state;
                fillCommonState(state);

                if(!COPY_FILTER::execute(core::execution::par_unseq, &state))  // execute is a static method
                    logger.log("Something went wrong while copying texel block data!", system::ILogger::ELL_ERROR);
            }
            else
            {
                if(asset::isBlockCompressionFormat(finalFormat))  // execute is a static method
                {
                    logger.log("Transcoding to Block Compressed formats not supported!", system::ILogger::ELL_ERROR);
                    return newImage;
                }

                CONVERSION_FILTER::state_type state;
                fillCommonState(state);
                state.swizzle = viewParams.components;

                if(!CONVERSION_FILTER::execute(core::execution::par_unseq, &state))  // static method
                    logger.log("Something went wrong while converting the image!", system::ILogger::ELL_ERROR);
            }
        }

        return newImage;
    }

    /*
			Patch for not supported by OpenGL R8_SRGB formats.
			Input image needs to have all the regions filled 
			and texel buffer attached as well.
		*/

    static inline core::smart_refctd_ptr<ICPUImage> convertR8ToR8G8B8Image(core::smart_refctd_ptr<ICPUImage> image, const system::logger_opt_ptr logger)
    {
        constexpr auto inputFormat = EF_R8_SRGB;
        constexpr auto outputFormat = EF_R8G8B8_SRGB;

        using CONVERSION_SWIZZLE_FILTER = CSwizzleAndConvertImageFilter<inputFormat, outputFormat>;

        core::smart_refctd_ptr<ICPUImage> newConvertedImage;
        {
            auto referenceImageParams = image->getCreationParameters();
            auto referenceBuffer = image->getBuffer();
            auto referenceRegions = image->getRegions();
            auto referenceRegion = referenceRegions.begin();
            const auto newTexelOrBlockByteSize = asset::getTexelOrBlockBytesize(outputFormat);

            auto newImageParams = referenceImageParams;
            auto newCpuBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(referenceBuffer->getSize() * newTexelOrBlockByteSize);
            auto newRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(referenceRegions.size());

            for(auto newRegion = newRegions->begin(); newRegion != newRegions->end(); ++newRegion)
            {
                *newRegion = *(referenceRegion++);
                newRegion->bufferOffset = newRegion->bufferOffset * newTexelOrBlockByteSize;
            }

            newImageParams.format = outputFormat;
            newConvertedImage = ICPUImage::create(std::move(newImageParams));
            newConvertedImage->setBufferAndRegions(std::move(newCpuBuffer), newRegions);

            CONVERSION_SWIZZLE_FILTER convertFilter;
            CONVERSION_SWIZZLE_FILTER::state_type state;

            ICPUImageView::SComponentMapping mapping;
            mapping.r = ICPUImageView::SComponentMapping::ES_R;
            mapping.g = ICPUImageView::SComponentMapping::ES_R;
            mapping.b = ICPUImageView::SComponentMapping::ES_R;
            mapping.a = ICPUImageView::SComponentMapping::ES_ONE;

            state.swizzle = mapping;
            state.inImage = image.get();
            state.outImage = newConvertedImage.get();
            state.inOffset = {0, 0, 0};
            state.inBaseLayer = 0;
            state.outOffset = {0, 0, 0};
            state.outBaseLayer = 0;

            for(auto itr = 0; itr < newConvertedImage->getCreationParameters().mipLevels; ++itr)
            {
                auto regionWithMipMap = newConvertedImage->getRegions(itr).begin();

                state.extent = regionWithMipMap->getExtent();
                state.layerCount = regionWithMipMap->imageSubresource.layerCount;
                state.inMipLevel = regionWithMipMap->imageSubresource.mipLevel;
                state.outMipLevel = regionWithMipMap->imageSubresource.mipLevel;

                if(!convertFilter.execute(core::execution::par_unseq, &state))
                    logger.log("Something went wrong while converting from R8 to R8G8B8 format!", system::ILogger::ELL_WARNING);
            }
        }
        return newConvertedImage;
    };

    /*
			Performs image's texel flip. A processing image must
			be have appropriate texel buffer and regions attached.
		*/

    static inline void performImageFlip(core::smart_refctd_ptr<asset::ICPUImage> image)
    {
        bool status = image->getBuffer() && image->getRegions().begin();
        assert(status);  // , "An image doesn't have a texel buffer and regions attached!");

        auto format = image->getCreationParameters().format;
        asset::TexelBlockInfo blockInfo(format);
        core::vector3du32_SIMD trueExtent = blockInfo.convertTexelsToBlocks(image->getRegions().begin()->getTexelStrides());

        auto entry = reinterpret_cast<uint8_t*>(image->getBuffer()->getPointer());
        auto end = entry + image->getBuffer()->getSize();
        auto stride = trueExtent.X * getTexelOrBlockBytesize(format);

        performImageFlip(entry, end, trueExtent.Y, stride);
    }

    static inline void performImageFlip(uint8_t* entry, uint8_t* end, uint32_t height, uint32_t rowPitch)
    {
        for(uint32_t y = 0, yRising = 0; y < height; y += 2, ++yRising)
            core::swap_ranges(core::execution::par_unseq, entry + (yRising * rowPitch), entry + ((yRising + 1) * rowPitch), end - ((yRising + 1) * rowPitch));
    }

private:
};

}
}

#endif