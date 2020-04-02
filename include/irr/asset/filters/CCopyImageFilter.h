// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_COPY_IMAGE_FILTER_H_INCLUDED__
#define __IRR_C_COPY_IMAGE_FILTER_H_INCLUDED__

#include "irr/core/core.h"

#include <type_traits>

#include "irr/asset/filters/CMatchedSizeInOutImageFilterCommon.h"

namespace irr
{
namespace asset
{

// copy while converting format from input image to output image
class CCopyImageFilter : public CImageFilter<CCopyImageFilter>, public CMatchedSizeInOutImageFilterCommon
{
	public:
		virtual ~CCopyImageFilter() {}
		
		using state_type = CMatchedSizeInOutImageFilterCommon::state_type;

		static inline bool validate(state_type* state)
		{
			if (!CMatchedSizeInOutImageFilterCommon::validate(state))
				return false;

			return getFormatClass(state->inImage->getCreationParameters().format)==getFormatClass(state->outImage->getCreationParameters().format);
		}

		static inline bool execute(state_type* state)
		{
			if (!validate(state))
				return false;

			auto* outImg = state->outImage;
			auto* inImg = state->inImage;
			const auto& inParams = inImg->getCreationParameters();
			const auto& outParams = outImg->getCreationParameters();
			const auto outFormat = outParams.format;
			const auto blockByteSize = getTexelOrBlockBytesize(outFormat);
			assert(blockByteSize==getTexelOrBlockBytesize(inParams.format)); // if this asserts the API got broken during an update or something

			const auto* inData = reinterpret_cast<const uint8_t*>(inImg->getBuffer()->getPointer());
			auto* outData = reinterpret_cast<uint8_t*>(outImg->getBuffer()->getPointer());

			const auto outRegions = outImg->getRegions(state->outMipLevel);
			auto oit = outRegions.begin();
			core::vectorSIMDu32 offsetDifference,outByteStrides;
			auto copy = [&offsetDifference,&outByteStrides,&oit,inData,outData,blockByteSize](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos) -> void
			{
				auto localOutPos = readBlockPos+offsetDifference;
				memcpy(outData+oit->getByteOffset(localOutPos,outByteStrides),inData+readBlockArrayOffset,blockByteSize);
			};
			auto inRegions = inImg->getRegions(state->inMipLevel);
			// iterate over output regions, then input cause read cache miss is faster
			for (; oit!=outRegions.end(); oit++)
			{
				IImage::SSubresourceLayers subresource = {static_cast<IImage::E_ASPECT_FLAGS>(0u),state->inMipLevel,state->inBaseLayer,state->layerCount};
				state_type::TexelRange range = {state->inOffset,state->extent};
				CBasicImageFilterCommon::clip_region_functor_t clip(subresource,range,outFormat);
				// setup convert state
				// I know my two's complement wraparound well enough to make this work
				offsetDifference = state->outOffsetBaseLayer-(core::vectorSIMDu32(oit->imageOffset.x,oit->imageOffset.y,oit->imageOffset.z,oit->imageSubresource.baseArrayLayer)+state->inOffsetBaseLayer);
				outByteStrides = oit->getByteStrides(IImage::SBufferCopy::TexelBlockInfo(outFormat), getTexelOrBlockBytesize(outFormat));
				CBasicImageFilterCommon::executePerRegion(inImg,convert,inRegions.begin(),inRegions.end(),clip);
			}

			return true;
		}
};

} // end namespace asset
} // end namespace irr

#endif