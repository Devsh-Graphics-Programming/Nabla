// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_TRIANGLE_IMAGE_FILTER_KERNEL_H_INCLUDED__
#define __NBL_ASSET_C_TRIANGLE_IMAGE_FILTER_KERNEL_H_INCLUDED__


#include "nbl/asset/filters/kernels/CommonImageFilterKernels.h"

namespace nbl
{
namespace asset
{

// standard Triangle function, symmetric, peak in the support is 1 and at origin, integral is 1, so support must be [-1,1]
// to get box filters of different widths we can use it in composition inside `CScaledImageFilterKernel`
class NBL_API CTriangleImageFilterKernel : public CFloatingPointIsotropicSeparableImageFilterKernelBase<CTriangleImageFilterKernel,std::ratio<1,1> >
{
		using Base = CFloatingPointIsotropicSeparableImageFilterKernelBase<CTriangleImageFilterKernel,std::ratio<1,1> >;

	public:
		inline float weight(float x, int32_t channel) const
		{
			if (Base::inDomain(x))
				return 1.f-core::abs(x);
			return 0.f;
		}

		_NBL_STATIC_INLINE_CONSTEXPR bool has_derivative = false;
		/* Derivative at 0 not defined so we cannot use
		inline float d_weight(float x) const
		{
			if (Base::inDomain(x))
				return x<0.f ? 1.f:(x>0.f ? -1.f:0.f);
			return 0.f;
		}*/
};

} // end namespace asset
} // end namespace nbl

#endif