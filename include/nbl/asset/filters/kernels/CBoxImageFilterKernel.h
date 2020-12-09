// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_BOX_IMAGE_FILTER_KERNEL_H_INCLUDED__
#define __NBL_ASSET_C_BOX_IMAGE_FILTER_KERNEL_H_INCLUDED__


#include "nbl/asset/filters/kernels/IImageFilterKernel.h"

#include <ratio>

namespace nbl
{
namespace asset
{

// standard Box function, symmetric, value in the support is 1, integral is 1, so support must be [-1/2,1/2]
// to get box filters of different widths we can use it in composition inside `CScaledImageFilterKernel`
class CBoxImageFilterKernel : public CFloatingPointIsotropicSeparableImageFilterKernelBase<CBoxImageFilterKernel,std::ratio<1,2> >
{
		using Base = CFloatingPointIsotropicSeparableImageFilterKernelBase<CBoxImageFilterKernel,std::ratio<1,2> >;

	public:
		inline float weight(float x, int32_t channel) const
		{
			return Base::inDomain(x) ? 1.f:0.f;
		}

		_NBL_STATIC_INLINE_CONSTEXPR bool has_derivative = false;
};

} // end namespace asset
} // end namespace nbl

#endif