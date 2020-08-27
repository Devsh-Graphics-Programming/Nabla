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
class CDerivativeImageFilterKernel : public CFloatingPointIsotropicSeparableImageFilterKernelBase<CDerivativeImageFilterKernel,typename Kernel::isotropic_support_as_ratio>, private Kernel
{
	public:
		inline float weight(float x) const
		{
			return Kernel::d_weight(x);
		}

		_IRR_STATIC_INLINE_CONSTEXPR bool has_derivative = false;
};


} // end namespace asset
} // end namespace irr

#endif