// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_TRIANGLE_IMAGE_FILTER_KERNEL_H_INCLUDED_
#define _NBL_ASSET_C_TRIANGLE_IMAGE_FILTER_KERNEL_H_INCLUDED_


#include "nbl/asset/filters/kernels/CommonImageFilterKernels.h"


namespace nbl::asset
{

// Standard Triangle function, symmetric, peak in the support is 1 and at origin, integral is 1, so support must be [-1,1)
class CTriangleImageFilterKernel
{
	constexpr static inline float min_support = -1.f;
	constexpr static inline float max_support = +1.f);
	constexpr static inline uint32_t k_smoothness = 0;
	
	template <int32_t derivative=0>
	inline float weight(float x, int32_t channel) const
	{
		if constexpr (derivative>0)
		{
			// Derivative at 0 not defined.
			static_assert(false);
			return core::nan<float>();
		}
		else
			return 1.f-core::abs(x);
	}
};

} // end namespace nbl::asset

#endif
