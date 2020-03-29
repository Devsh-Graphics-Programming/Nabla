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
				CImageFilter<CFillImageFilter>::ColorValue fillValue;

				virtual ~CState() {}
		};
		using state_type = CState;

		static inline bool validate(CState* state)
		{
			return CBasicOutImageFilterCommon::validate(state);
		}

		static inline bool execute(CState* state)
		{
			if (!validate(state))
				return false;

			auto* img = state->outImage();
			const CImageFilter<CFillImageFilter>::ColorValue::WriteInfo info(img->getCreationParameters().format,img->getBuffer()->getPointer());
			// do the per-pixel filling
			auto fill = [state,&info](uint32_t blockArrayOffset, uint32_t x, uint32_t y, uint32_t z) -> void
			{
				state->fillValue.writeMemory(info,blockArrayOffset);
			};
			// TODO: find regions that correspond to the subsection
			{
				CBasicImageFilterCommon::executePerBlock<std::decay<decltype(fill)>::type>(fill,region);
			}

			return true;
		}
};

} // end namespace asset
} // end namespace irr

#endif