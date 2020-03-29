// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_SWIZZLE_IMAGE_FILTER_H_INCLUDED__
#define __IRR_C_SWIZZLE_IMAGE_FILTER_H_INCLUDED__

#include "irr/core/core.h"

#include <type_traits>

#include "irr/asset/filters/CBasicImageFilterCommon.h"

namespace irr
{
namespace asset
{

// do a per-pixel recombination of image channels
class CSwizzleImageFilter : public CImageFilter<CSwizzleImageFilter>
{
	public:
		virtual ~CSwizzleImageFilter() {}

		class CState : public CBasicInOutImageFilterCommon::state_type
		{
			public:
				void swizzle;

				virtual ~CState() {}
		};
		using state_type = CState;

		static inline bool validate(CState* state)
		{
			return CBasicInOutImageFilterCommon::validate(state);
		}

		static inline bool execute(CState* state)
		{
			if (!validate(state))
				return false;

			// do the per-pixel filling

			return true;
		}
};

} // end namespace asset
} // end namespace irr

#endif