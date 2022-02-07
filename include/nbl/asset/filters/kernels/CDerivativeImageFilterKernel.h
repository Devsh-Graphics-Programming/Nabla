// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_DERIVATIVE_IMAGE_FILTER_KERNEL_H_INCLUDED__
#define __NBL_ASSET_C_DERIVATIVE_IMAGE_FILTER_KERNEL_H_INCLUDED__

#include "nbl/core/declarations.h"

#include "nbl/asset/filters/kernels/CommonImageFilterKernels.h"

#include <type_traits>

namespace nbl
{
namespace asset
{
// A Kernel that's a derivative of another, `Kernel` must have a `d_weight` function
template<class Kernel>
class CDerivativeImageFilterKernel : public CFloatingPointSeparableImageFilterKernelBase<CDerivativeImageFilterKernel<Kernel>>
{
    using Base = CFloatingPointSeparableImageFilterKernelBase<CDerivativeImageFilterKernel<Kernel>>;

    Kernel kernel;

public:
    using value_type = typename Base::value_type;

    CDerivativeImageFilterKernel(Kernel&& k)
        : Base(k.negative_support.x, k.positive_support.x), kernel(std::move(k)) {}

    // no special user data by default
    inline const IImageFilterKernel::UserData* getUserData() const { return nullptr; }

    inline float weight(float x, int32_t channel) const
    {
        auto* scale = IImageFilterKernel::ScaleFactorUserData::cast(kernel.getUserData());
        if(scale)
            x *= scale->factor[channel];
        return kernel.d_weight(x, channel);
    }

    _NBL_STATIC_INLINE_CONSTEXPR bool has_derivative = false;

    NBL_DECLARE_DEFINE_CIMAGEFILTER_KERNEL_PASS_THROUGHS(Base)
};

}  // end namespace asset
}  // end namespace nbl

#endif