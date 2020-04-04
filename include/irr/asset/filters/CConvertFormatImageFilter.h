// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_CONVERT_FORMAT_IMAGE_FILTER_H_INCLUDED__
#define __IRR_C_CONVERT_FORMAT_IMAGE_FILTER_H_INCLUDED__

#include "irr/core/core.h"

#include <type_traits>

#include "irr/asset/filters/CSwizzleAndConvertImageFilter.h"

namespace irr
{
namespace asset
{

// copy while converting format from input image to output image
class CConvertFormatImageFilter : public CImageFilter<CConvertFormatImageFilter>, public CMatchedSizeInOutImageFilterCommon
{
	public:
		virtual ~CConvertFormatImageFilter() {}
		
		using state_type = CMatchedSizeInOutImageFilterCommon::state_type;

		static inline bool validate(state_type* state)
		{
			return CMatchedSizeInOutImageFilterCommon::validate(state);
		}

		static inline bool execute(state_type* state)
		{	
			auto perOutputRegion = [](const auto& commonExecuteData, CBasicImageFilterCommon::clip_region_functor_t& clip) -> bool
			{
				assert(getTexelOrBlockBytesize(commonExecuteData.inFormat)==getTexelOrBlockBytesize(commonExecuteData.outFormat)); // if this asserts the API got broken during an update or something
				auto convert = [&commonExecuteData](uint32_t readBlockArrayOffset, core::vectorSIMDu32 readBlockPos) -> void
				{
					auto localOutPos = readBlockPos*blockDims+commonExecuteData.offsetDifference;
					const void* sourcePixels[4] = {commonExecuteData.inData+readBlockArrayOffset,nullptr,nullptr,nullptr};
					assert(false);
/*
					convertColor(	commonExecuteData.inFormat,commonExecuteData.outFormat,sourcePixels,
									commonExecuteData.outData+commonExecuteData.oit->getByteOffset(localOutPos,commonExecuteData.outByteStrides),
									1u,core::vector3d<uint32_t>(1u,1u,1u));
*/
				};
				CBasicImageFilterCommon::executePerRegion(commonExecuteData.inImg,convert,commonExecuteData.inRegions.begin(),commonExecuteData.inRegions.end(),clip);
			};
			return commonExecute(state,perOutputRegion);
		}
};

} // end namespace asset
} // end namespace irr

#endif