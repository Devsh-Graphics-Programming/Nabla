// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_MITCHELL_IMAGE_FILTER_KERNEL_H_INCLUDED__
#define __NBL_ASSET_C_MITCHELL_IMAGE_FILTER_KERNEL_H_INCLUDED__


#include "nbl/asset/filters/kernels/IImageFilterKernel.h"

#include <ratio>

namespace nbl
{
namespace asset
{


// A standard Mitchell filter expressed as a convolution kernel, the standard has a support of [-2,2] the B and C template parameters are the same ones from the paper
template<class B=std::ratio<1,3>, class C=std::ratio<1,3> >
class NBL_API CMitchellImageFilterKernel : public CFloatingPointIsotropicSeparableImageFilterKernelBase<CMitchellImageFilterKernel<B,C>,std::ratio<2,1> >
{
		using Base = CFloatingPointIsotropicSeparableImageFilterKernelBase<CMitchellImageFilterKernel<B,C>,std::ratio<2,1> >;

	public:
		inline float weight(float x, int32_t channel) const
		{
			if (Base::inDomain(x))
			{
				x = core::abs(x);
				return core::mix(p0+x*x*(p2+x*p3),q0+x*(q1+x*(q2+x*q3)),x>=1.f);
			}
			return 0.f;
		}

		_NBL_STATIC_INLINE_CONSTEXPR bool has_derivative = true;
		inline float d_weight(float x, int32_t channel) const
		{
			if (Base::inDomain(x))
			{
				bool neg = x < 0.f;
				x = core::abs(x);
				float retval = core::mix(x * (2.f * p2 + 3.f * x * p3), q1 + x * (2.f * q2 + 3.f * x * q3), x >= 1.f);
				return neg ? (-retval) : retval;
			}
			return 0.f;
		}

	protected:
		_NBL_STATIC_INLINE_CONSTEXPR float b = float(B::num)/float(B::den);
		_NBL_STATIC_INLINE_CONSTEXPR float c = float(C::num)/float(C::den);
		_NBL_STATIC_INLINE_CONSTEXPR float p0 = (6.0f - 2.0f * b) / 6.0f;
		_NBL_STATIC_INLINE_CONSTEXPR float p2 = (-18.0f + 12.0f * b + 6.0f * c) / 6.0f;
		_NBL_STATIC_INLINE_CONSTEXPR float p3 = (12.0f - 9.0f * b - 6.0f * c) / 6.0f;
		_NBL_STATIC_INLINE_CONSTEXPR float q0 = (8.0f * b + 24.0f * c) / 6.0f;
		_NBL_STATIC_INLINE_CONSTEXPR float q1 = (-12.0f * b - 48.0f * c) / 6.0f;
		_NBL_STATIC_INLINE_CONSTEXPR float q2 = (6.0f * b + 30.0f * c) / 6.0f;
		_NBL_STATIC_INLINE_CONSTEXPR float q3 = (-b - 6.0f * c) / 6.0f;
};

} // end namespace asset
} // end namespace nbl

#endif