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
			image view to save only stuff a user has choosen.
		*/

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

			return newImage;
		}

		/*
			Create an image containing a single row from taking an ICPUBuffer 
			as a single row and convert it to any format. Since it's
			data may not only limit to stuff being displayed on a screen,
			there is an optiomal parameter for bufferRowLength pitch that
			is helpful while dealing with specific data which needs it.
		*/

		template<E_FORMAT inputFormat, E_FORMAT outputFormat>
		static inline core::smart_refctd_ptr<ICPUImage> createSingleRowImageFromRawData(core::smart_refctd_ptr<asset::ICPUBuffer> inputBuffer, bool createWithBufferRowLengthPitch = false)
		{
			auto rowData = inputBuffer->getPointer();
			const uint32_t texelOrBlockLength = inputBuffer->getSize() / asset::getTexelOrBlockBytesize(inputFormat);

			using CONVERSION_FILTER = CConvertFormatImageFilter<inputFormat, outputFormat>;
			CONVERSION_FILTER convertFilter;
			CONVERSION_FILTER::state_type state;

			auto createImage = [&](E_FORMAT format, bool copyInputMemory = true)
			{
				const auto texelOrBlockByteSize = asset::getTexelOrBlockBytesize(format);
				const uint32_t pitchTexelOrBlockLength = createWithBufferRowLengthPitch ? calcPitchInBlocks(texelOrBlockLength, texelOrBlockByteSize) : texelOrBlockLength;

				ICPUImage::SCreationParams imgInfo;
				imgInfo.format = format;
				imgInfo.type = ICPUImage::ET_1D;
				imgInfo.extent = { texelOrBlockLength * asset::getBlockDimensions(format).X, 1, 1 };
				imgInfo.mipLevels = 1u;
				imgInfo.arrayLayers = 1u;
				imgInfo.samples = ICPUImage::ESCF_1_BIT;
				imgInfo.flags = static_cast<IImage::E_CREATE_FLAGS>(0u);

				auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1u);
				auto texelBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(texelOrBlockByteSize * pitchTexelOrBlockLength);

				if (copyInputMemory)
					texelBuffer = core::make_smart_refctd_ptr<asset::CCustomAllocatorCPUBuffer<core::null_allocator<uint8_t>>>(texelOrBlockByteSize * texelOrBlockLength, rowData, core::adopt_memory);

				ICPUImage::SBufferCopy& region = regions->front();

				region.imageSubresource.mipLevel = 0u;
				region.imageSubresource.baseArrayLayer = 0u;
				region.imageSubresource.layerCount = 1u;
				region.bufferOffset = 0u;
				region.bufferRowLength = asset::IImageAssetHandlerBase::calcPitchInBlocks(pitchTexelOrBlockLength * asset::getBlockDimensions(format).X, texelOrBlockByteSize);
				region.bufferImageHeight = 0u;
				region.imageOffset = { 0u, 0u, 0u };
				region.imageExtent = imgInfo.extent;

				auto singleRowImage = ICPUImage::create(std::move(imgInfo));
				singleRowImage->setBufferAndRegions(std::move(texelBuffer), regions);

				return singleRowImage;
			};

			core::smart_refctd_ptr<ICPUImage> inputSingleRowImage = createImage(inputFormat);
			core::smart_refctd_ptr<ICPUImage> outputSingleRowImage = createImage(outputFormat, false);

			auto attachedRegion = outputSingleRowImage->getRegions().begin();

			state.inImage = inputSingleRowImage.get();
			state.outImage = outputSingleRowImage.get();
			state.inOffset = { 0, 0, 0 };
			state.inBaseLayer = 0;
			state.outOffset = { 0, 0, 0 };
			state.outBaseLayer = 0;
			state.extent = attachedRegion->getExtent();
			state.layerCount = attachedRegion->imageSubresource.layerCount;
			state.inMipLevel = attachedRegion->imageSubresource.mipLevel;
			state.outMipLevel = attachedRegion->imageSubresource.mipLevel;

			if (!convertFilter.execute(&state))
				os::Printer::log("Something went wrong while converting the row!", ELL_WARNING);

			return outputSingleRowImage;
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