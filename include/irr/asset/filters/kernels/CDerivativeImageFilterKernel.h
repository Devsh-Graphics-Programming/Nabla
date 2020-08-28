// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_DERIVATIVE_IMAGE_FILTER_KERNEL_H_INCLUDED__
#define __IRR_C_DERIVATIVE_IMAGE_FILTER_KERNEL_H_INCLUDED__

#include "irr/core/core.h"

#include "irr/asset/filters/kernels/CommonImageFilterKernels.h"

#include <type_traits>

namespace irr
{
namespace asset
{

// A Kernel that's a derivative of another, `Kernel` must have a `d_weight` function
template<class Kernel>
class CDerivativeImageFilterKernel : public CFloatingPointSeparableImageFilterKernelBase<CDerivativeImageFilterKernel<Kernel>>, private Kernel
{
	public:
		inline float weight(float x, int32_t channel) const
		{
			auto* scale = IImageFilterKernel::ScaleFactorUserData::cast(static_cast<const Kernel*>(this)->getUserData());
			if (scale)
				x *= scale->factor[channel];
			return Kernel::d_weight(x,channel);
		}

		_IRR_STATIC_INLINE_CONSTEXPR bool has_derivative = false;
};


} // end namespace asset
} // end namespace irr

#endif