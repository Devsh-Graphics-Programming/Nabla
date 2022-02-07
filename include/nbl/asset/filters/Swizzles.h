// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_ASSET_SWIZZLES_H_INCLUDED_
#define _NBL_ASSET_SWIZZLES_H_INCLUDED_

#include "nbl/core/declarations.h"

#include "nbl/asset/ICPUImageView.h"

#include <type_traits>

namespace nbl::asset
{
/*
	Default Swizzle for compile time cases
*/
struct DefaultSwizzle
{
    ICPUImageView::SComponentMapping swizzle;

    /*
		Performs swizzle on out compoments following
		swizzle member with all four compoments. You 
		can specify channels for custom pointers.
	*/

    template<typename InT, typename OutT>
    void operator()(const InT* in, OutT* out, uint8_t channels = SwizzleBase::MaxChannels) const
    {
        using in_t = std::conditional_t<std::is_void_v<InT>, uint64_t, InT>;
        using out_t = std::conditional_t<std::is_void_v<OutT>, uint64_t, OutT>;
        for(auto i = 0u; i < channels; i++)
        {
            in_t component;
            const auto mapping = (&swizzle.r)[i];
            switch(mapping)
            {
                case ICPUImageView::SComponentMapping::ES_IDENTITY:
                    component = reinterpret_cast<const in_t*>(in)[i];
                    break;
                case ICPUImageView::SComponentMapping::ES_ZERO:
                    component = in_t(0);
                    break;
                case ICPUImageView::SComponentMapping::ES_ONE:
                    component = in_t(1);
                    break;
                default:
                    component = reinterpret_cast<const in_t*>(in)[mapping - ICPUImageView::SComponentMapping::ES_R];
                    break;
            }
            reinterpret_cast<out_t*>(out)[i] = out_t(component);
        }
    }
};

/*
	Compile time Swizzle
*/
template<ICPUImageView::SComponentMapping::E_SWIZZLE... swizzle>
struct StaticSwizzle
{
    static_assert(sizeof...(swizzle) <= SwizzleBase::MaxChannels);

    template<typename InT, typename OutT>
    void operator()(const InT* in, OutT* out, uint8_t channels = sizeof...(swizzle)) const
    {
        assert(channels <= sizeof...(swizzle));
        DefaultSwizzle pseudoRuntime = {};
        pseudoRuntime.swizzle = {swizzle...};
        pseudoRuntime.operator()<InT, OutT>(in, out, channels);
    }
};

}  // end namespace nbl:asset

#endif