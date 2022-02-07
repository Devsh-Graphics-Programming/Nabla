// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_CHANNEL_INDEPENDENT_IMAGE_FILTER_KERNEL_H_INCLUDED__
#define __NBL_ASSET_C_CHANNEL_INDEPENDENT_IMAGE_FILTER_KERNEL_H_INCLUDED__

#include "nbl/core/core.h"

#include "nbl/asset/filters/kernels/CommonImageFilterKernels.h"

#include <type_traits>
#include <tuple>

namespace nbl
{
namespace asset
{
namespace impl
{
template<class... Kernels>
class CChannelIndependentImageFilterKernelBase
{
protected:
    const bool haveScale;
    const IImageFilterKernel::ScaleFactorUserData scale;

    using kernels_t = std::tuple<Kernels...>;
    kernels_t kernels;

    static inline bool doesHaveScale(const Kernels&... kernels)
    {
        return (IImageFilterKernel::ScaleFactorUserData::cast(kernels.getUserData()) && ...);
    }

    static inline IImageFilterKernel::ScaleFactorUserData computeScale(bool bother, const Kernels&... kernels)
    {
        IImageFilterKernel::ScaleFactorUserData retval(1.f);
        if(bother)
        {
            std::array<const IImageFilterKernel::ScaleFactorUserData*, sizeof...(kernels)> userData = {IImageFilterKernel::ScaleFactorUserData::cast(kernels.getUserData())...};
            for(auto i = 0; i < userData.size(); i++)
            {
                retval.factor[i] = userData[i]->factor[i];
            }
        }
        return retval;
    }

public:
    explicit CChannelIndependentImageFilterKernelBase(Kernels&&... kernels)
        : haveScale(doesHaveScale(kernels...)), scale(computeScale(haveScale, kernels...)),
          kernels(std::move(kernels)...)
    {
    }
};

}

template<class... Kernels>
class CChannelIndependentImageFilterKernel : public CFloatingPointSeparableImageFilterKernelBase<CChannelIndependentImageFilterKernel<Kernels...>>, public impl::CChannelIndependentImageFilterKernelBase<Kernels...>
{
    static_assert(sizeof...(Kernels) <= 4u);
    static_assert(sizeof...(Kernels) >= 1u);

    using base_t = CFloatingPointSeparableImageFilterKernelBase<CChannelIndependentImageFilterKernel<Kernels...>>;
    using channel_indep_base_t = impl::CChannelIndependentImageFilterKernelBase<Kernels...>;

public:
    using value_type = typename base_t::value_type;

    _NBL_STATIC_INLINE_CONSTEXPR size_t MaxChannels = sizeof...(Kernels);

private:
    enum E_CHANNEL
    {
        E_R = 0,
        E_G = 1,
        E_B = 2,
        E_A = 3
    };
    template<E_CHANNEL ch>
    _NBL_STATIC_INLINE_CONSTEXPR bool has_kernel_v = ch < MaxChannels;

    struct dummy_kernel_t
    {
        _NBL_STATIC_INLINE_CONSTEXPR bool has_derivative = false;
    };
    template<E_CHANNEL ch>
    using kernel_t = std::conditional_t<has_kernel_v<ch>,
        std::tuple_element_t<std::min(static_cast<size_t>(ch), MaxChannels - 1ull), typename channel_indep_base_t::kernels_t>,
        dummy_kernel_t>;

    template<E_CHANNEL ch>
    kernel_t<ch>& getKernel() { return std::get<static_cast<size_t>(ch)>(kernels); }
    template<E_CHANNEL ch>
    const kernel_t<ch>& getKernel() const { return std::get<static_cast<size_t>(ch)>(kernels); }

    static core::vectorSIMDf maxSupport(std::initializer_list<core::vectorSIMDf> ilist)
    {
        core::vectorSIMDf m = *ilist.begin();
        for(auto it = ilist.begin() + 1; it != ilist.end(); ++it)
            m = core::max(m, *it);
        return m;
    }
    static float getMaxNegSupport(const Kernels&... kernels)
    {
        core::vectorSIMDf v = maxSupport({kernels.negative_support...});
        return *std::max_element(v.pointer, v.pointer + MaxChannels);
    }
    static float getMaxPosSupport(const Kernels&... kernels)
    {
        core::vectorSIMDf v = maxSupport({kernels.positive_support...});
        return *std::max_element(v.pointer, v.pointer + MaxChannels);
    }

public:
    CChannelIndependentImageFilterKernel(Kernels&&... kernels)
        : base_t(getMaxNegSupport(kernels...), getMaxPosSupport(kernels...)),
          channel_indep_base_t(std::move(kernels)...)
    {}

    // pass on any scale
    inline const IImageFilterKernel::UserData* getUserData() const { return haveScale ? &scale : nullptr; }

    inline float weight(float x, int32_t channel) const
    {
        switch(channel)
        {
            case 0:
                return getKernel<E_R>().weight(x, 0);
            case 1:
                if constexpr(has_kernel_v<E_G>)
                {
                    return getKernel<E_G>().weight(x, 1);
                }
                break;
            case 2:
                if constexpr(has_kernel_v<E_B>)
                {
                    return getKernel<E_B>().weight(x, 2);
                }
                break;
            case 3:
                if constexpr(has_kernel_v<E_A>)
                {
                    return getKernel<E_A>().weight(x, 3);
                }
                break;
            default:
                break;
        }
        return 0.f;
    }

    inline float d_weight(float x, int32_t channel) const
    {
        switch(channel)
        {
            case 0:
                if constexpr(kernel_t<E_R>::has_derivative)
                {
                    return getKernel<E_R>().d_weight(x, 0);
                }
                break;
            case 1:
                if constexpr(kernel_t<E_G>::has_derivative)
                {
                    return getKernel<E_G>().d_weight(x, 1);
                }
                break;
            case 2:
                if constexpr(kernel_t<E_B>::has_derivative)
                {
                    return getKernel<E_B>().d_weight(x, 2);
                }
                break;
            case 3:
                if constexpr(kernel_t<E_A>::has_derivative)
                {
                    return getKernel<E_A>().d_weight(x, 3);
                }
                break;
            default:
                break;
        }
        return 0.f;
    }

    NBL_DECLARE_DEFINE_CIMAGEFILTER_KERNEL_PASS_THROUGHS(base_t)
};

}  // end namespace asset
}  // end namespace nbl

#endif