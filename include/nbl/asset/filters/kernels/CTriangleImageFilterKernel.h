// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_TRIANGLE_IMAGE_FILTER_KERNEL_H_INCLUDED__
#define __NBL_ASSET_C_TRIANGLE_IMAGE_FILTER_KERNEL_H_INCLUDED__

#include "nbl/asset/filters/kernels/CommonImageFilterKernels.h"

namespace nbl::asset
{

// Standard Triangle function, symmetric, peak in the support is 1 and at origin, integral is 1, so support must be [-1,1)
class CTriangleImageFilterKernel : public CFloatingPointIsotropicSeparableImageFilterKernelBase<CTriangleImageFilterKernel>
{
	using Base = CFloatingPointIsotropicSeparableImageFilterKernelBase<CTriangleImageFilterKernel>;

	public:
		CTriangleImageFilterKernel() : Base(1.f) {}

		template <uint32_t derivative = 0>
		inline float weight(float x, int32_t channel) const
		{
			if (Base::inDomain(x))
			{
				if constexpr (derivative == 0)
				{
					return 1.f-core::abs(x);
				}
				else
				{
					// Derivative at 0 not defined.
					static_assert(false);
					return core::nan<float>();
				}
			}
			return 0.f;
		}

		static inline constexpr bool has_derivative = false;
};

} // end namespace nbl::asset

#endif