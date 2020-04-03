// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_FILL_IMAGE_FILTER_H_INCLUDED__
#define __IRR_C_FILL_IMAGE_FILTER_H_INCLUDED__

#include "irr/core/core.h"

#include <type_traits>

#include "irr/asset/filters/CBasicImageFilterCommon.h"

namespace irr
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
			auto fill = [state,&info](uint32_t blockArrayOffset, core::vectorSIMDu32 unusedVariable) -> bool
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
} // end namespace irr

#endif