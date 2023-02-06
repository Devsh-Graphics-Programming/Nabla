// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_MITCHELL_IMAGE_FILTER_KERNEL_H_INCLUDED__
#define __NBL_ASSET_C_MITCHELL_IMAGE_FILTER_KERNEL_H_INCLUDED__

#include "nbl/asset/filters/kernels/IImageFilterKernel.h"

#include <ratio>

namespace nbl::asset
{

// A standard Mitchell filter expressed as a convolution kernel, the standard has a support of [-2,2] the B and C template parameters are the same ones from the paper
template<class B=std::ratio<1,3>, class C=std::ratio<1,3>>
class NBL_API CMitchellImageFilterKernel : public CFloatingPointIsotropicSeparableImageFilterKernelBase<CMitchellImageFilterKernel<B,C>>
{
	using Base = CFloatingPointIsotropicSeparableImageFilterKernelBase<CMitchellImageFilterKernel<B,C>>;

	public:
		CMitchellImageFilterKernel() : Base(2.f) {}

		template <uint32_t derivative = 0>
		inline float weight(float x, int32_t channel) const
		{
			if (Base::inDomain(x))
			{
				bool neg = x < 0.f;
				x = core::abs(x);

				float retval;
				if constexpr (derivative == 0)
				{
					return core::mix(p0 + x * x * (p2 + x * p3), q0 + x * (q1 + x * (q2 + x * q3)), x >= 1.f);
				}
				else if constexpr (derivative == 1)
				{
					retval = core::mix(x * (2.f * p2 + 3.f * x * p3), q1 + x * (2.f * q2 + 3.f * x * q3), x >= 1.f);
				}
				else if constexpr (derivative == 2)
				{
					retval = core::mix(2.f * p2 + 6.f * p3 * x, 2.f * q2 + 6.f * q3 * x, x >= 1.f);
				}
				else if constexpr (derivative == 3)
				{
					retval = core::mix(6.f * p3, 6.f * q3, x >= 1.f);
				}
				else
				{
					static_assert(false);
					return core::nan<float>();
				}

				return neg ? -retval : retval;
			}
			return 0.f;
		}

		static inline constexpr bool has_derivative = true;

	protected:
		static inline constexpr float b = float(B::num)/float(B::den);
		static inline constexpr float c = float(C::num)/float(C::den);
		static inline constexpr float p0 = (6.0f - 2.0f * b) / 6.0f;
		static inline constexpr float p2 = (-18.0f + 12.0f * b + 6.0f * c) / 6.0f;
		static inline constexpr float p3 = (12.0f - 9.0f * b - 6.0f * c) / 6.0f;
		static inline constexpr float q0 = (8.0f * b + 24.0f * c) / 6.0f;
		static inline constexpr float q1 = (-12.0f * b - 48.0f * c) / 6.0f;
		static inline constexpr float q2 = (6.0f * b + 30.0f * c) / 6.0f;
		static inline constexpr float q3 = (-b - 6.0f * c) / 6.0f;
};

} // end namespace nbl::asset

#endif