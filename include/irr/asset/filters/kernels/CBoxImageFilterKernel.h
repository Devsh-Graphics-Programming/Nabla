// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_BOX_IMAGE_FILTER_KERNEL_H_INCLUDED__
#define __IRR_C_BOX_IMAGE_FILTER_KERNEL_H_INCLUDED__


#include "irr/asset/filters/kernels/IImageFilterKernel.h"

#include <ratio>

namespace irr
{
namespace asset
{

// standard Box function, symmetric, value in the support is 1, integral is 1, so support must be [-1/2,1/2]
// to get box filters of different widths we can use it in composition inside `CScaledImageFilterKernel`
class CBoxImageFilterKernel : public CFloatingPointIsotropicSeparableImageFilterKernelBase<CBoxImageFilterKernel,std::ratio<1,2> >
{
		using Base = CFloatingPointIsotropicSeparableImageFilterKernelBase<CBoxImageFilterKernel,std::ratio<1,2> >;

	public:
		inline float weight(float x, int32_t channel) const
		{
			return Base::inDomain(x) ? 1.f:0.f;
		}

		_IRR_STATIC_INLINE_CONSTEXPR bool has_derivative = false;
};

} // end namespace asset
} // end namespace irr

#endif