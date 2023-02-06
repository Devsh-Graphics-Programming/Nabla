// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_BOX_IMAGE_FILTER_KERNEL_H_INCLUDED__
#define __NBL_ASSET_C_BOX_IMAGE_FILTER_KERNEL_H_INCLUDED__


#include "nbl/asset/filters/kernels/IImageFilterKernel.h"

namespace nbl::asset
{

// Standard Box function, symmetric, value in the support is 1, integral is 1, so support must be [-1/2,1/2)
class NBL_API CBoxImageFilterKernel : public CFloatingPointIsotropicSeparableImageFilterKernelBase<CBoxImageFilterKernel>
{
	using Base = CFloatingPointIsotropicSeparableImageFilterKernelBase<CBoxImageFilterKernel>;

	public:
		CBoxImageFilterKernel() : Base(0.5f) {}

		template <uint32_t derivative = 0>
		inline float weight(float x, int32_t channel) const
		{
			if (Base::inDomain(x))
			{
				if constexpr (derivative == 0)
				{
					return 1.f;
				}
				else
				{
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