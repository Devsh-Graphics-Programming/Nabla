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
			auto perOutputRegion = [](const auto& commonExecuteData, CBasicImageFilterCommon::clip_region_functor_t& clip) -> bool
			{
				assert(getTexelOrBlockBytesize(commonExecuteData.inFormat)==getTexelOrBlockBytesize(commonExecuteData.outFormat)); // if this asserts the API got broken during an update or something
				auto copy = [&commonExecuteData](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos) -> void
				{
					auto localOutPos = readBlockPos+commonExecuteData.offsetDifference;
					memcpy(commonExecuteData.outData+commonExecuteData.oit->getByteOffset(localOutPos,commonExecuteData.outByteStrides),commonExecuteData.inData+readBlockArrayOffset,commonExecuteData.blockByteSize);
				};
				CBasicImageFilterCommon::executePerRegion(commonExecuteData.inImg,copy,commonExecuteData.inRegions.begin(),commonExecuteData.inRegions.end(),clip);
			};
			return commonExecute(state,perOutputRegion);
		}
};

} // end namespace asset
} // end namespace irr

#endif