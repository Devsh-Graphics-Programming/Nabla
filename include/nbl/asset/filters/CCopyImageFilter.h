// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_COPY_IMAGE_FILTER_H_INCLUDED__
#define __NBL_ASSET_C_COPY_IMAGE_FILTER_H_INCLUDED__

#include "nbl/core/declarations.h"

#include <type_traits>

#include "nbl/asset/filters/CMatchedSizeInOutImageFilterCommon.h"

namespace nbl
{
namespace asset
{

//! Copy Filter
/*
	Copy a one input image's texel data in strictly defined way to another one output image.
	The usage is as follows:
	- create a convert filter reference by \busing YOUR_COPY_FILTER = CCopyImageFilter;\b
	- provide it's state by \bYOUR_COPY_FILTER::state_type\b and fill appropriate fields
	- launch one of \bexecute\b calls

	\attention
	{
		Take a look any overlapping regions shall be copied into the output in exactly 
		the order they were when specifying the image. So the last region copies into image 
		last, overwriting any overlapped pixels.
	}
	
	@see IImageFilter
	@see CMatchedSizeInOutImageFilterCommon
*/

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
} // end namespace nbl

#endif