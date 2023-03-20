// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_GAUSSIAN_IMAGE_FILTER_KERNEL_H_INCLUDED__
#define __NBL_ASSET_C_GAUSSIAN_IMAGE_FILTER_KERNEL_H_INCLUDED__

#include "nbl/asset/filters/kernels/IImageFilterKernel.h"

namespace nbl::asset
{

// Truncated Gaussian filter, with stddev = 1.0, if you want a different stddev then you need to scale it.
class CGaussianImageFilterKernel : public CFloatingPointIsotropicSeparableImageFilterKernelBase<CGaussianImageFilterKernel>
{
	using Base = CFloatingPointIsotropicSeparableImageFilterKernelBase<CGaussianImageFilterKernel>;

	public:
		CGaussianImageFilterKernel(const float isotropicSupport = 3.f) : Base(isotropicSupport) {}

		template <uint32_t derivative = 0>
		inline float weight(float x, int32_t channel) const
		{
			if (Base::inDomain(x))
			{
				if constexpr (derivative == 0)
				{
					const float normalizationFactor = core::inversesqrt(2.f*core::PI<float>())/std::erff(core::sqrt<float>(2.f)*float(negative_support.x));
					return normalizationFactor*exp2f(-0.72134752f*x*x);
				}
				else if constexpr (derivative == 1)
				{
					return -x * CGaussianImageFilterKernel::weight(x, channel);
				}
				else if constexpr (derivative == 2)
				{
					return x * (x + 1.f) * CGaussianImageFilterKernel::weight(x, channel);
				}
				else if constexpr (derivative == 3)
				{
					return (1.f - (x - 1.f) * (x + 1.f) * (x + 1.f)) * CGaussianImageFilterKernel::weight(x, channel);
				}
				else
				{
					static_assert(false, "TODO");
					return core::nan<float>();
				}
			}
			return 0.f;
		}

		_NBL_STATIC_INLINE_CONSTEXPR bool has_derivative = true;
};

} // end namespace nbl::asset

#endif