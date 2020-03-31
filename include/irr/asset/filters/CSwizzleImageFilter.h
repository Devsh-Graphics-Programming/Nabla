// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_SWIZZLE_IMAGE_FILTER_H_INCLUDED__
#define __IRR_C_SWIZZLE_IMAGE_FILTER_H_INCLUDED__

#include "irr/core/core.h"

#include <type_traits>

#include "irr/asset/ICPUImageView.h"
#include "irr/asset/filters/CConvertFormatImageFilter.h"

namespace irr
{
namespace asset
{

// do a per-pixel recombination of image channels
// `FlatImageInput` means that there is at most one region per mip-map level
template<bool FlatImageInput=false>
class CSwizzleImageFilter : public CImageFilter<CSwizzleImageFilter<FlatImageInput>>
{
	public:
		virtual ~CSwizzleImageFilter() {}

		class CState : public CConvertFormatImageFilter<FlatImageInput>::state_type
		{
			public:
				ICPUImageView::SComponentMapping swizzle;

				virtual ~CState() {}
		};
		using state_type = CState;

		static inline bool validate(state_type* state)
		{
			if (!CConvertFormatImageFilter<FlatImageInput>::validate(state))
				return false;

			if (state->inImage->getCreationParameters().format!=state->outImage->getCreationParameters().format)
				return false;

			return true;
		}

		static inline bool execute(state_type* state)
		{
			if (!validate(state))
				return false;

			enum FORMAT_CLASS
			{
				FC_FLOAT,
				FC_INT,
				FC_UINT
			};
#if 0
			E_FORMAT format;
			FORMAT_CLASS formatClass;
			{
				void* pixels[4] = {,nullptr,nullptr,nullptr};
				auto doSwizzle = [&pixels](auto tmp[4]) -> void
				{
					decodePixels(format,pixels,tmp,0u,0u);
					std::decay<decltype(tmp[0])>::type tmp2[4];
					tmp2[0] = tmp[swizzle.r];
					tmp2[1] = tmp[swizzle.g];
					tmp2[2] = tmp[swizzle.b];
					tmp2[3] = tmp[swizzle.a];
					encodePixels(format,pixels[0],tmp2);
				};
				switch (formatClass)
				{
					case FC_FLOAT:
					{
						double tmp[4];
						doSwizzle(tmp);
						break;
					}
					case FC_INT:
					{
						int64_t tmp[4];
						doSwizzle(tmp);
						break;
					}
					default:
					{
						uint64_t tmp[4];
						doSwizzle(tmp);
						break;
					}
				}
			}

			return true;
#endif
			return false;
		}
};

} // end namespace asset
} // end namespace irr

#endif