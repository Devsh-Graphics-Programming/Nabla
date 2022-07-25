// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_GAUSSIAN_IMAGE_FILTER_KERNEL_H_INCLUDED__
#define __NBL_ASSET_C_GAUSSIAN_IMAGE_FILTER_KERNEL_H_INCLUDED__


#include "nbl/asset/filters/kernels/IImageFilterKernel.h"

#include <ratio>

namespace nbl
{
namespace asset
{

// Truncated Gaussian filter, with stddev = 1.0, if you want a different stddev then you need to scale it with `CScaledImageFilterKernel`
template<uint32_t support=3u>
class NBL_API CGaussianImageFilterKernel : public CFloatingPointIsotropicSeparableImageFilterKernelBase<CGaussianImageFilterKernel<support>,std::ratio<support,1> >
{
		using Base = CFloatingPointIsotropicSeparableImageFilterKernelBase<CGaussianImageFilterKernel<support>,std::ratio<support,1> >;

	public:
		inline float weight(float x, int32_t channel) const
		{
			if (Base::inDomain(x))
			{
				const float normalizationFactor = core::inversesqrt(2.f*core::PI<float>())/std::erff(core::sqrt<float>(2.f)*float(support));
				return normalizationFactor*exp2f(-0.72134752f*x*x);
			}
			return 0.f;
		}

		_NBL_STATIC_INLINE_CONSTEXPR bool has_derivative = true;
		inline float d_weight(float x, int32_t channel) const
		{
			if (Base::inDomain(x))
				return -x*CGaussianImageFilterKernel<support>::weight(x,channel);
			return 0.f;
		}
};

} // end namespace asset
} // end namespace nbl

#endif