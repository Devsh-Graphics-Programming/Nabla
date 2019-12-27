// Copyright (C) 2009-2012 AnastaZIuk
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_I_IMAGE_LOADER_H_INCLUDED__
#define __IRR_I_IMAGE_LOADER_H_INCLUDED__

#include "irr/core/core.h"

#include "IWriteFile.h"
#include "irr/asset/IAsset.h"

namespace irr
{
	namespace asset
	{

		class IImageLoader : public virtual core::IReferenceCounted
		{
			public:

				IImageLoader();
				virtual ~IImageLoader();

			protected:

				template<class ConvertImageTexelToOutputFunctional> inline void flattenRegions(void* frameBuffer, uint32_t pixelStride, uint32_t rowStride, uint32_t heightStride, IImage::SBufferCopy* _begin, IImage::SBufferCopy* _end, bool doesItHandleSeperateChannelBuffers = false)
				{
					for (auto region = _begin; region != _end; ++region)
						for (uint64_t yPos = region->imageOffset.y; yPos < region->imageOffset.y + region->imageExtent.height; ++yPos)
							for (uint64_t xPos = region->imageOffset.x; xPos < region->imageOffset.x + region->imageExtent.width; ++xPos)
							{
								const uint64_t offsetToPixelBeggining = (yPos * rowStride * pixelStride) + (xPos * pixelStride);

								for (uint8_t channelIndex = 0; channelIndex < pixelStride; ++channelIndex)
									ConvertImageTexelToOutputFunctional().operator()(frameBuffer, offsetToPixelBeggining, channelIndex, doesItHandleSeperateChannelBuffers);
							}
				}

			private:
		};
	}
}

#endif // __IRR_I_IMAGE_LOADER_H_INCLUDED__
