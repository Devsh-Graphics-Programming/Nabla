// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_COPY_IMAGE_FILTER_H_INCLUDED__
#define __NBL_ASSET_C_COPY_IMAGE_FILTER_H_INCLUDED__

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

			auto perOutputRegion = [](const CommonExecuteData& commonExecuteData, CBasicImageFilterCommon::clip_region_functor_t& clip) -> bool
			{
				assert(getTexelOrBlockBytesize(commonExecuteData.inFormat)==getTexelOrBlockBytesize(commonExecuteData.outFormat)); // if this asserts the API got broken during an update or something

				const auto blockDims = asset::getBlockDimensions(commonExecuteData.inFormat);
				auto copy = [&commonExecuteData,&blockDims](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos) -> void
				{
					auto localOutPos = readBlockPos*blockDims+commonExecuteData.offsetDifference;
					memcpy(commonExecuteData.outData+commonExecuteData.oit->getByteOffset(localOutPos,commonExecuteData.outByteStrides),commonExecuteData.inData+readBlockArrayOffset,commonExecuteData.outBlockByteSize);
				};
				CBasicImageFilterCommon::executePerRegion(commonExecuteData.inImg,copy,commonExecuteData.inRegions.begin(),commonExecuteData.inRegions.end(),clip);

				return true;
			};

			return commonExecute(state,perOutputRegion);
		}
};

} // end namespace asset
} // end namespace irr

#endif