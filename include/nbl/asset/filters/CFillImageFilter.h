// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_FILL_IMAGE_FILTER_H_INCLUDED__
#define __NBL_ASSET_C_FILL_IMAGE_FILTER_H_INCLUDED__

#include "nbl/core/declarations.h"

#include <type_traits>

#include "nbl/asset/filters/CBasicImageFilterCommon.h"

namespace nbl
{
namespace asset
{

// fill a section of the image with a uniform value
class CFillImageFilter : public CImageFilter<CFillImageFilter>
{
	public:
		virtual ~CFillImageFilter() {}

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
			const auto& params = img->getCreationParameters();
			const IImageFilter::IState::ColorValue::WriteMemoryInfo info(params.format,img->getBuffer()->getPointer());
			// do the per-pixel filling
			auto fill = [state,&info](uint32_t blockArrayOffset, core::vectorSIMDu32 unusedVariable) -> void
			{
				state->fillValue.writeMemory(info,blockArrayOffset);
			};
			CBasicImageFilterCommon::clip_region_functor_t clip(state->subresource,state->outRange,params.format);
			const auto& regions = img->getRegions(state->subresource.mipLevel);
			CBasicImageFilterCommon::executePerRegion(img,fill,regions.begin(),regions.end(),clip);

			return true;
		}
};

} // end namespace asset
} // end namespace nbl

#endif