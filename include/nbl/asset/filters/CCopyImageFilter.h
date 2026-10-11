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

		template<class ExecutionPolicy>
		static inline bool execute(ExecutionPolicy&& policy, state_type* state)
		{
			if (!validate(state))
				return false;

			auto perOutputRegion = [policy](const CommonExecuteData& commonExecuteData, CBasicImageFilterCommon::clip_region_functor_t& clip) -> bool
			{
				assert(getTexelOrBlockBytesize(commonExecuteData.inFormat)==getTexelOrBlockBytesize(commonExecuteData.outFormat)); // if this asserts the API got broken during an update or something

				const TexelBlockInfo blockInfo(commonExecuteData.inFormat);
				const auto outBlockStrides = commonExecuteData.oit->getBlockStrides(TexelBlockInfo(commonExecuteData.outFormat));
				for (auto region : commonExecuteData.inRegions)
				{
					if (!clip(region,&region))
						continue;
					core::vectorSIMDu32 readBlockPos(region.imageOffset.x,region.imageOffset.y,region.imageOffset.z);
					readBlockPos = blockInfo.convertTexelsToBlocks(readBlockPos);
					readBlockPos.w = region.imageSubresource.baseArrayLayer;
					core::vectorSIMDu32 extentInBlocks(region.imageExtent.width,region.imageExtent.height,region.imageExtent.depth);
					extentInBlocks = blockInfo.convertTexelsToBlocks(extentInBlocks);
					extentInBlocks.w = region.imageSubresource.layerCount;
					const auto writeOffset = commonExecuteData.oit->getByteOffset(readBlockPos+commonExecuteData.offsetDifferenceInBlocks,commonExecuteData.outByteStrides);
					CBasicImageFilterCommon::copyBlocks(policy,commonExecuteData.outData+writeOffset,outBlockStrides,commonExecuteData.inData+region.bufferOffset,region.getBlockStrides(blockInfo),blockInfo.getBlockByteSize(),extentInBlocks);
				}

				return true;
			};

			state->outImage->setContentHash(IPreHashed::INVALID_HASH);

			return commonExecute(state,perOutputRegion);
		}
		static inline bool execute(state_type* state)
		{
			return execute(core::execution::seq,state);
		}
};

} // end namespace asset
} // end namespace nbl

#endif