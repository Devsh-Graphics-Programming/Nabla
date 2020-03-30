// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_CONVERT_FORMAT_IMAGE_FILTER_H_INCLUDED__
#define __IRR_C_CONVERT_FORMAT_IMAGE_FILTER_H_INCLUDED__

#include "irr/core/core.h"

#include <type_traits>

#include "irr/asset/filters/CBasicImageFilterCommon.h"

namespace irr
{
namespace asset
{

// fill a section of the image with a uniform value
class CConvertFormatImageFilter : public CImageFilter<CConvertFormatImageFilter>
{
	public:
		virtual ~CConvertFormatImageFilter() {}
#if 0 // WIP
		class CState : public CBasicOutImageFilterCommon::state_type
		{
			public:
				IImageFilter::IState::ColorValue fillValue;

				virtual ~CState() {}
		};
		using state_type = CState;

		static inline bool validate(state_type* state)
		{
			return CBasicOutImageFilterCommon::validate(state);
		}

		static inline bool execute(state_type* state)
		{
			if (!validate(state))
				return false;

			auto* img = state->outImage;
			const IImageFilter::IState::ColorValue::WriteMemoryInfo info(img->getCreationParameters().format,img->getBuffer()->getPointer());
			// do the per-pixel filling
			auto fill = [state,&info](uint32_t blockArrayOffset, uint32_t x, uint32_t y, uint32_t z, uint32_t layer) -> bool
			{
				state->fillValue.writeMemory(info,blockArrayOffset);
			};
			auto clipRegion = [state](IImage::SBufferCopy& newRegion, const IImage::SBufferCopy* referenceRegion) -> bool
			{
				if (state->subresource.mipLevel!=referenceRegion->imageSubresource.mipLevel)
					return false;
				newRegion.imageSubresource.baseArrayLayer = core::max(state->subresource.baseArrayLayer,referenceRegion->imageSubresource.baseArrayLayer);
				newRegion.imageSubresource.layerCount = core::min(	state->subresource.baseArrayLayer+state->subresource.layerCount,
																	referenceRegion->imageSubresource.baseArrayLayer+referenceRegion->imageSubresource.layerCount);
				if (newRegion.imageSubresource.layerCount <= newRegion.imageSubresource.baseArrayLayer)
					return false;
				newRegion.imageSubresource.layerCount -= newRegion.imageSubresource.baseArrayLayer;

				// handle the clipping
				newRegion.imageOffset = core::max(referenceRegion->imageOffset,state->outRange.offset);
				newRegion.imageExtent = core::min(referenceRegion->imageExtent,state->outRange.extent);

				// compute new offset
				newRegion.bufferOffset += ;
				return true;
			};
			const auto& regions = img->getRegions();
			CBasicImageFilterCommon::executePerRegion<decltype(fill),decltype(clipRegion)>(img,fill,regions.begin(),regions.end(),clipRegion);

			return true;
		}
#endif
};

} // end namespace asset
} // end namespace irr

#endif