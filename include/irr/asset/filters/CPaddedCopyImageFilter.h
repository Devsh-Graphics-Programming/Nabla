// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_PADDED_COPY_IMAGE_FILTER_H_INCLUDED__
#define __IRR_C_PADDED_COPY_IMAGE_FILTER_H_INCLUDED__

#include "irr/core/core.h"

#include <type_traits>

#include "irr/asset/ISampler.h"
#include "irr/asset/filters/CCopyImageFilter.h"

namespace irr
{
namespace asset
{

// copy while pasting a configurable border
class CPaddedCopyImageFilter : public CImageFilter<CPaddedCopyImageFilter>, public CMatchedSizeInOutImageFilterCommon
{
	public:
		virtual ~CPaddedCopyImageFilter() {}
		
		class CState : public CMatchedSizeInOutImageFilterCommon::state_type
		{
			public:
				virtual ~CState() {}
				
				VkExtent3D borderPadding = { 0u,0u,0u };
				_IRR_STATIC_INLINE_CONSTEXPR auto NumWrapAxes = 3;
				ISampler::E_TEXTURE_CLAMP axisWraps[NumWrapAxes] = {ISampler::ETC_REPEAT,ISampler::ETC_REPEAT,ISampler::ETC_REPEAT};
		};
		using state_type = CState;

		static inline bool validate(state_type* state)
		{
			if (!CMatchedSizeInOutImageFilterCommon::validate(state))
				return false;

			core::vectorSIMDu32 borderPadding(&state->borderPadding.width); borderPadding.w = 0u;
			if ((state->outOffsetBaseLayer<borderPadding).any())
				return false;
			const auto& outParams = state->outImage->getCreationParameters();
			core::vectorSIMDu32 extent(&outParams.extent.width); extent.w = outParams.arrayLayers;
			if ((state->outOffsetBaseLayer+state->extentLayerCount+borderPadding>extent).any())
				return false;

			auto const inFormat = state->inImage->getCreationParameters().format;
			auto const outFormat = outParams.format;
			// TODO: eventually remove when we can encode blocks
			for (auto i=0; i<CState::NumWrapAxes; i++)
			{
				if ((isBlockCompressionFormat(inFormat)||isBlockCompressionFormat(outFormat))&&state->axisWraps[i]!=ISampler::ETC_REPEAT)
					return false;
			}

			return getFormatClass(inFormat)==getFormatClass(outFormat);
		}

		static inline bool execute(state_type* state)
		{
			// with a valid state you are guaranteed that padding added to offsets and extent in output will not overflow the output image
			auto perOutputRegion = [](const CommonExecuteData& commonExecuteData, const CBasicImageFilterCommon::clip_region_functor_t& unusedDummy) -> bool // this lambda runs once per output region
			{
				assert(getTexelOrBlockBytesize(commonExecuteData.inFormat)==getTexelOrBlockBytesize(commonExecuteData.outFormat)); // if this asserts the API got broken during an update or something
				
				// ok now iterate through the commonExecuteData.inRegions and find out the input image regions which need to have at least 1 texel block copied to output, and call
				/// executePerBlock<padded_copy_functor_t>(inImage, region, paddedcopy);
				// on them with a copy functor (or a capture lambda instead of padded_copy_functor_t)

				// you will want to re-use or look at CBasicImageFilterCommon::clip_region_functor_t to see how we can compute corrected clipped intersection
														
				// NOTE: You want to invoke `executePerBlock` multiple times (up to 9 times) for an input region that contributes to a border
				// IMPORTANT: `offsetDifference, outByteStrides` are junk and are supposed to be filled out by YOU!

				/**
				CLEVER abuse tricks:
				FIRST OF ALL compute `offsetDifference` slightly differently to the CCopyImageFilter (set back or forward by padding size)
				- for ETC_REPEAT clip the input region appropriately
				- for ETC_CLAMP_TO_EDGE recompute the `readBlockArrayOffset` from clamped global input coordinates
				- for ETC_CLAMP_TO_BORDER ?
				- for ETC_MIRROR compute `offsetDifference` as the offset to upper right corner (MAX) of the output area, and invert values `outByteStrides` (even though its unsigned, twos complement will work here - see discord)
				- for ETC_MIRROR_CLAMP_TO_EDGE like ETC_MIRROR but ETC_CLAMP_TO_EDGE after one repetition
				- for ETC_MIRROR_CLAMP_TO_BORDER like ETC_MIRROR but ETC_CLAMP_TO_BORDER after one repetition
				**/
				assert(false);
				return false;
			};
			// this iterates through every region in output image and calls perOutputRegion
			return commonExecute(state,perOutputRegion);
		}

	private:
		struct padded_copy_functor_t
		{
			// `readBlockPos` is the global input image block position (not texel) and `readBlockArrayOffset` is the total offset of the texel block in the backing buffer
			inline void operator()(uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos)
			{
				// do stuff
				assert(false);
			}
		};
};

} // end namespace asset
} // end namespace irr

#endif