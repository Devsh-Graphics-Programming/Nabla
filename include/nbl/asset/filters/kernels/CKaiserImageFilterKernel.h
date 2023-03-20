// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_KAISER_IMAGE_FILTER_KERNEL_H_INCLUDED__
#define __NBL_ASSET_C_KAISER_IMAGE_FILTER_KERNEL_H_INCLUDED__

#include "nbl/asset/filters/kernels/IImageFilterKernel.h"

namespace nbl::asset
{

// Kaiser filter, basically a windowed sinc.
class CKaiserImageFilterKernel : public CFloatingPointIsotropicSeparableImageFilterKernelBase<CKaiserImageFilterKernel>
{
	using Base = CFloatingPointIsotropicSeparableImageFilterKernelBase<CKaiserImageFilterKernel>;

	// important constant, do not touch, do not tweak
	static inline constexpr float alpha = 3.f;

public:
	CKaiserImageFilterKernel(const float isotropicSupport = 3.f) : Base(isotropicSupport) {}

	template <uint32_t derivative = 0>
	inline float weight(float x, int32_t channel) const
	{
		if (Base::inDomain(x))
		{
			if constexpr (derivative == 0)
			{
				const auto PI = core::PI<float>();
				return core::sinc(x*PI)*core::KaiserWindow(x,alpha,negative_support.x);
			}
			else if constexpr (derivative == 1)
			{
				const auto PIx = core::PI<float>() * x;
				float f = core::sinc(PIx);
				float df = core::PI<float>() * core::d_sinc(PIx);
				float g = core::KaiserWindow(x, alpha, negative_support.x);
				float dg = core::d_KaiserWindow(x, alpha, negative_support.x);
				return df * g + f * dg;
			}
			else
			{
				static_assert(false, "TODO");
				return core::nan<float>();
			}
		}
		return 0.f;
	}

	static inline constexpr bool has_derivative = true;
};

} // end namespace nbl::asset

#endif