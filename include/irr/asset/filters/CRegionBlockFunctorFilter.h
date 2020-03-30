// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_REGION_BLOCK_FUNCTOR_FILTER_H_INCLUDED__
#define __IRR_C_REGION_BLOCK_FUNCTOR_H_INCLUDED__

#include "irr/core/core.h"

#include <type_traits>

#include "irr/asset/filters/CBasicImageFilterCommon.h"

namespace irr
{
namespace asset
{

// fill a section of the image with a uniform value
template<typename Functor>
class CRegionBlockFunctorFilter : public CImageFilter<CRegionBlockFunctorFilter>
{
	public:
		virtual ~CRegionBlockFunctorFilter() {}

		class CState
		{
			public:
				ICPUImage* image;
				Functor& functor;
				IImage::SBufferCopy* regionIterator;

				virtual ~CState() {}
		};
		using state_type = CState;

		static inline bool validate(state_type* state)
		{
			if (!state->image)
				return false;

			if (!state->regionIterator)
				return false;	
			const auto& regions = state->image->getRegions();
			if (state->regionIterator<regions.begin() || state->regionIterator>=regions.end())
				return false;

			return true;
		}

		static inline bool execute(state_type* state)
		{
			if (!validate(state))
				return false;

			CBasicImageFilterCommon::executePerBlock<Functor>(state->inImage, state->region, state->functor);

			return true;
		}
};

} // end namespace asset
} // end namespace irr

#endif