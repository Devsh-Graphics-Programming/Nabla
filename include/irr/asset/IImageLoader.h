#ifndef __IRR_I_IMAGE_LOADER_H_INCLUDED__
#define __IRR_I_IMAGE_LOADER_H_INCLUDED__

#include "irr/core/core.h"
#include "IAssetLoader.h"
#include "IImageAssetHandlerBase.h"

namespace irr
{
	namespace asset
	{

		class IImageLoader : public IAssetLoader, public IImageAssetHandlerBase
		{
			public:

			protected:

				IImageLoader() = default;
				virtual ~IImageLoader() = 0;

				//! A function converting data in EF_R8_SRGB format to EF_R8G8B8_SRGB
				/**
					\bredChannelDataLayer\b parameter is consider as bunch of texels 
					for certain single layer in region.
					Region parameter is used to provide flexibility with data texel
					convertion for various mipmap levels.

					The reson for existance of the function is that R8 format with SRGB encoding used to be available in 
					compatibility (with 2.1) OpenGL profile, but got removed from 3.3 core profile only to be added back to 
					OpenGL 4.6, so not many GPU's drivers can natively accept and display a data, so the convertion is available.
				*/

				uint8_t* convertR8SRGBdataIntoRGB8SRGBAAndGetIt(const void* redChannelDataLayer, const core::smart_refctd_ptr<ICPUImage>& image, const irr::asset::IImage::SBufferCopy& region)
				{
					const auto& imageSize = image->getCreationParameters().extent;
					irr::core::vector3d<uint32_t> imageSizeWithPitch;
					imageSizeWithPitch.X = region.bufferRowLength > 0 ? region.bufferRowLength : region.imageExtent.width;
					imageSizeWithPitch.Y = region.bufferImageHeight > 0 ? region.bufferImageHeight : region.imageExtent.height;
					imageSizeWithPitch.Z = region.imageExtent.depth;
					
					constexpr auto pixelByteSize = 3;
					const void* planarData[] = { redChannelDataLayer, nullptr, nullptr, nullptr };
					uint8_t* out = _IRR_NEW_ARRAY(uint8_t, imageSizeWithPitch.X * imageSizeWithPitch.Y * imageSizeWithPitch.Z);

					auto fillValuesOfRGBTexelsWithRValue = [&]()
					{
						for (uint64_t zPos = 0; zPos < imageSize.depth; ++zPos)
							for (uint64_t yPos = 0; yPos < imageSize.height; ++yPos)
								for (uint64_t xPos = 0; xPos < imageSize.width; ++xPos)
								{
									auto texelPtr = out + (((zPos * imageSizeWithPitch.Y + yPos) * imageSizeWithPitch.X + xPos) * pixelByteSize);
									const auto redValueOfTexel = *texelPtr;

									for (uint8_t channel = 1; channel < pixelByteSize; ++channel)
										*(texelPtr + channel) = redValueOfTexel;
								}
					};

					fillValuesOfRGBTexelsWithRValue();

					return out;
				}

			private:
		};
	}
}

#endif // __IRR_I_IMAGE_LOADER_H_INCLUDED__
