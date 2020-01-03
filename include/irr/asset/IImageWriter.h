#ifndef __IRR_I_IMAGE_WRITER_H_INCLUDED__
#define __IRR_I_IMAGE_WRITER_H_INCLUDED__

#include "IImage.h"
#include "irr/core/core.h"
#include "IAssetWriter.h"
#include "IImageAssetHandlerBase.h"

namespace irr
{
	namespace asset
	{

		class IImageWriter : public IAssetWriter, public IImageAssetHandlerBase
		{
			public:

			protected:

				IImageWriter() = default;
				virtual ~IImageWriter() = 0;
		
				template<class ConvertImageTexelToOutputFunctional> inline void flattenRegions(void* frameBuffer, uint32_t pixelStride, uint32_t rowStride, uint32_t heightStride, IImage::SBufferCopy* _begin, IImage::SBufferCopy* _end, bool doesItHandleSeperateChannelBuffers = false)
				{
					for (auto region = _begin; region != _end; ++region)
					{
						auto regionWidth = region->bufferRowLength == 0 ? region->imageExtent.width : region->bufferRowLength;
						auto regionHeight = region->bufferImageHeight == 0 ? region->imageExtent.height : region->bufferImageHeight;

						for (uint64_t yPos = region->imageOffset.y; yPos < region->imageOffset.y + regionHeight; ++yPos)
							for (uint64_t xPos = region->imageOffset.x; xPos < region->imageOffset.x + regionWidth; ++xPos)
							{
								const uint64_t offsetToPixelBeggining = ((regionHeight + yPos) * regionWidth + xPos) * pixelStride + region->bufferOffset;

								for (uint8_t channelIndex = 0; channelIndex < pixelStride; ++channelIndex)
									ConvertImageTexelToOutputFunctional().operator()(frameBuffer, offsetToPixelBeggining, channelIndex, doesItHandleSeperateChannelBuffers);
							}
					}
				}

			private:
		};
	}
}

#endif // __IRR_I_IMAGE_WRITER_H_INCLUDED__
