#ifndef __IRR_I_IMAGE_ASSET_HANDLER_BASE_H_INCLUDED__
#define __IRR_I_IMAGE_ASSET_HANDLER_BASE_H_INCLUDED__

#include "irr/core/core.h"

namespace irr
{
namespace asset
{

class IImageAssetHandlerBase : public virtual core::IReferenceCounted
{
	protected:

		IImageAssetHandlerBase() = default;
		virtual ~IImageAssetHandlerBase() = 0;

	public:

		static const uint32_t MAX_PITCH_ALIGNMENT = 8u;										             

		/*
			 Returns pitch for buffer row lenght, because
			 OpenGL cannot transfer rows with arbitrary padding
		*/

		static inline uint32_t calcPitchInBlocks(uint32_t width, uint32_t blockByteSize)       
		{
			auto rowByteSize = width * blockByteSize;
			for (uint32_t _alignment = MAX_PITCH_ALIGNMENT; _alignment > 1u; _alignment >>= 1u)
			{
				auto paddedSize = core::alignUp(rowByteSize, _alignment);
				if (paddedSize % blockByteSize)
					continue;
				return paddedSize / blockByteSize;
			}
			return width;
		}

		static inline core::vector3du32_SIMD calcPitchInBlocks(uint32_t width, uint32_t height, uint32_t depth, uint32_t blockByteSize)
		{
			constexpr auto VALUE_FOR_ALIGNMENT = 1;
			core::vector3du32_SIMD retVal;
			retVal.X = calcPitchInBlocks(width, blockByteSize);
			retVal.Y = calcPitchInBlocks(height, VALUE_FOR_ALIGNMENT);
			retVal.Z = calcPitchInBlocks(depth, VALUE_FOR_ALIGNMENT);
			return retVal;
		}

		/*
			Create a new image with only one top level region, one layer and one mip-map level.
			Handling ordinary images in asset writing process is a mess since multi-regions
			are valid. To avoid ambitious, the function will handle top level data from
			image view to save only stuff a user has choosen. You can also specify extra
			output format, top image data will be converted to such. You can leave template
			parameter to ensure there will be no conversion.
		*/

		template<asset::E_FORMAT outFormat = EF_UNKNOWN>
		static inline core::smart_refctd_ptr<ICPUImage> getTopImageDataForCommonWriting(const ICPUImageView* imageView)
		{
			auto referenceImage = imageView->getCreationParameters().image;
			auto referenceImageParams = referenceImage->getCreationParameters();
			auto referenceRegions = referenceImage->getRegions();
			auto referenceTopRegion = referenceRegions.begin();

			core::smart_refctd_ptr<ICPUImage> newImage;
			{
				auto newImageParams = referenceImageParams;
				newImageParams.arrayLayers = 1;
				newImageParams.mipLevels = 1;
				newImageParams.type = IImage::ET_2D;

				auto newRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1);
				newRegions->front() = *referenceTopRegion;

				const auto texelOrBlockByteSize = asset::getTexelOrBlockBytesize(referenceImageParams.format);
				auto texelBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(texelOrBlockByteSize * newRegions->front().bufferRowLength * newImageParams.extent.height * newImageParams.extent.depth);

				newImage = ICPUImage::create(std::move(newImageParams));
				newImage->setBufferAndRegions(std::move(texelBuffer), newRegions);
			}

			using COPY_FILTER = CCopyImageFilter;
			COPY_FILTER copyFilter;
			COPY_FILTER::state_type state;

			auto newTopRegion = newImage->getRegions().begin();

			state.inImage = referenceImage.get();
			state.outImage = newImage.get();
			state.inOffset = { 0, 0, 0 };
			state.outOffset = { 0, 0, 0 };
			state.inBaseLayer = 0;
			state.outBaseLayer = 0;
			state.extent = newTopRegion->getExtent();
			state.layerCount = 1;
			state.inMipLevel = newTopRegion->imageSubresource.mipLevel;
			state.outMipLevel = 0;

			if(!copyFilter.execute(&state))
				os::Printer::log("Something went wrong while copying top level region texel's data to the image!", ELL_WARNING);

			if(newImage->getCreationParameters().format == outFormat || outFormat == EF_UNKNOWN)
				return newImage;
			else
			{
				using CONVERSION_FILTER = CConvertFormatImageFilter<EF_UNKNOWN, outFormat>;
				CONVERSION_FILTER convertFilter;
				CONVERSION_FILTER::state_type state;

				auto newParams = newImage->getCreationParameters();
				newParams.format = outFormat;
				auto texelOrBlockByteSize = asset::getTexelOrBlockBytesize(newParams.format);

				auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1u);
				auto& topRegion = regions->front() = *newImage->getRegions().begin();

				asset::TexelBlockInfo blockInfo(newParams.format);
				core::vector3du32_SIMD trueExtent = blockInfo.convertTexelsToBlocks(topRegion.getTexelStrides());

				auto texelBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(texelOrBlockByteSize * trueExtent.X);

				auto convertedNewImage = ICPUImage::create(std::move(newParams));
				convertedNewImage->setBufferAndRegions(std::move(texelBuffer), regions);

				state.inImage = newImage.get();
				state.outImage = convertedNewImage.get();
				state.inOffset = { 0, 0, 0 };
				state.inBaseLayer = 0;
				state.outOffset = { 0, 0, 0 };
				state.outBaseLayer = 0;
				state.extent = { newParams.extent.width, newParams.extent.height, newParams.extent.depth };
				state.layerCount = 1;
				state.inMipLevel = 0;
				state.outMipLevel = 0;

				if (!convertFilter.execute(&state))
					os::Printer::log("Something went wrong while converting the image!", ELL_WARNING);

				return convertedNewImage;
			}
		}

		/*
			Patch for not supported by OpenGL R8_SRGB formats.
			Input image needs to have all the regions filled 
			and texel buffer attached as well.
		*/

		static inline core::smart_refctd_ptr<ICPUImage> convertR8ToR8G8B8Image(core::smart_refctd_ptr<ICPUImage> image)
		{
			constexpr auto inputFormat = EF_R8_SRGB;
			constexpr auto outputFormat = EF_R8G8B8_SRGB;

			using CONVERSION_FILTER = CConvertFormatImageFilter<inputFormat, outputFormat>;

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

				for (auto newRegion = newRegions->begin(); newRegion != newRegions->end(); ++newRegion)
				{
					*newRegion = *(referenceRegion++);
					newRegion->bufferOffset = newRegion->bufferOffset * newTexelOrBlockByteSize;
				}

				newImageParams.format = outputFormat;
				newConvertedImage = ICPUImage::create(std::move(newImageParams));
				newConvertedImage->setBufferAndRegions(std::move(newCpuBuffer), newRegions);

				CONVERSION_FILTER convertFilter;
				CONVERSION_FILTER::state_type state;

				state.inImage = image.get();
				state.outImage = newConvertedImage.get();
				state.inOffset = { 0, 0, 0 };
				state.inBaseLayer = 0;
				state.outOffset = { 0, 0, 0 };
				state.outBaseLayer = 0;

				for (auto itr = 0; itr < newConvertedImage->getCreationParameters().mipLevels; ++itr)
				{
					auto regionWithMipMap = newConvertedImage->getRegions(itr).begin();

					state.extent = regionWithMipMap->getExtent();
					state.layerCount = regionWithMipMap->imageSubresource.layerCount;
					state.inMipLevel = regionWithMipMap->imageSubresource.mipLevel;
					state.outMipLevel = regionWithMipMap->imageSubresource.mipLevel;
				
					if (!convertFilter.execute(&state))
						os::Printer::log("Something went wrong while converting from R8 to R8G8B8 format!", ELL_WARNING);
				}
			}
			return newConvertedImage;
		};

	private:
};

}
}

#endif // __IRR_I_IMAGE_ASSET_HANDLER_BASE_H_INCLUDED__