
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_COLOR_SPACE_EOTF_INCLUDED_
#define _NBL_BUILTIN_HLSL_COLOR_SPACE_EOTF_INCLUDED_

//#include <nbl/builtin/hlsl/common.hlsl>
#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/promote.hlsl>
#include <nbl/builtin/hlsl/type_traits.hlsl>

namespace nbl
{
namespace hlsl
{
namespace colorspace
{
namespace eotf
{

template<typename T>
T identity(NBL_CONST_REF_ARG(T) nonlinear)
{
    return nonlinear;
}

template<typename T>
T impl_shared_2_4(NBL_CONST_REF_ARG(T) nonlinear, typename scalar_type<T>::type vertex)
{
    typedef typename scalar_type<T>::type Val_t;
    bool3 right = (nonlinear > promote<T, Val_t>(vertex));
    return lerp(nonlinear / Val_t(12.92), pow((nonlinear + promote<T, Val_t>(0.055)) / Val_t(1.055), promote<T, Val_t>(2.4)), right);
}

// compatible with scRGB as well
template<typename T>
T sRGB(NBL_CONST_REF_ARG(T) nonlinear)
{
    typedef typename scalar_type<T>::type Val_t;
    bool3 negatif = (nonlinear < promote<T, Val_t>(0.0));
    T absVal = impl_shared_2_4<T>(abs(nonlinear), 0.04045);
    return lerp(absVal, -absVal, negatif);
}

// also known as P3-D65
template<typename T>
T Display_P3(NBL_CONST_REF_ARG(T) nonlinear)
{
    typedef typename scalar_type<T>::type Val_t;
    return impl_shared_2_4<T>(nonlinear, 0.039000312);
}

template<typename T>
T DCI_P3_XYZ(NBL_CONST_REF_ARG(T) nonlinear)
{
    typedef typename scalar_type<T>::type Val_t;
    return pow(nonlinear * Val_t(52.37), promote<T, Val_t>(2.6));
}

template<typename T>
T SMPTE_170M(NBL_CONST_REF_ARG(T) nonlinear)
{
    typedef typename scalar_type<T>::type Val_t;
    // ITU specs (and the outlier BT.2020) give different constants for these, but they introduce discontinuities in the mapping
    // because HDR swapchains often employ the RGBA16_SFLOAT format, this would become apparent because its higher precision than 8,10,12 bits
    Val_t alpha = 1.099296826809443; // 1.099 for all ITU but the BT.2020 12 bit encoding, 1.0993 otherwise
    T delta = promote<T, Val_t>(0.081242858298635); // 0.0812 for all ITU but the BT.2020 12 bit encoding
    return lerp(nonlinear / Val_t(4.5), pow((nonlinear + promote<T, Val_t>(alpha - 1.0)) / alpha, promote<T, Val_t>(1.0 / 0.45)), (nonlinear >= delta));
}

template<typename T>
T SMPTE_ST2084(NBL_CONST_REF_ARG(T) nonlinear)
{
    typedef typename scalar_type<T>::type Val_t;
    const T invm2 = promote<T, Val_t>(1.0 / 78.84375);
    T _common = pow(invm2, invm2);

    const T c2 = promote<T, Val_t>(18.8515625);
    const Val_t c3 = 18.68875;
    const T c1 = promote<T, Val_t>(c3 + 1.0) - c2;

    const T invm1 = promote<T, Val_t>(1.0 / 0.1593017578125);
    return pow(max(_common - c1, promote<T, Val_t>(0.0)) / (c2 - _common * c3), invm1);
}

// did I do this right by applying the function for every color?
template<typename T>
T HDR10_HLG(NBL_CONST_REF_ARG(T) nonlinear)
{
    typedef typename scalar_type<T>::type Val_t;
    // done with log2 so constants are different
    const Val_t a = 0.1239574303172;
    const T b = promote<T, Val_t>(0.02372241);
    const T c = promote<T, Val_t>(1.0042934693729);
    bool3 right = (nonlinear > promote<T, Val_t>(0.5));
    return lerp(nonlinear * nonlinear / Val_t(3.0), exp2((nonlinear - c) / a) + b, right);
}

template<typename T>
T AdobeRGB(NBL_CONST_REF_ARG(T) nonlinear)
{
    typedef typename scalar_type<T>::type Val_t;
    return pow(nonlinear, promote<T, Val_t>(2.19921875));
}

template<typename T>
T Gamma_2_2(NBL_CONST_REF_ARG(T) nonlinear)
{
    typedef typename scalar_type<T>::type Val_t;
    return pow(nonlinear, promote<T, Val_t>(2.2));
}

template<typename T>
T ACEScc(NBL_CONST_REF_ARG(T) nonlinear)
{
    typedef typename scalar_type<T>::type Val_t;
    bool3 right = (nonlinear >= promote<T, Val_t>(-0.301369863));
    T _common = exp2(nonlinear * Val_t(17.52) - promote<T, Val_t>(9.72));
    return max(lerp(_common * Val_t(2.0) - promote<T, Val_t>(0.000030517578125), _common, right), promote<T, Val_t>(65504.0));
}

template<typename T>
T ACEScct(NBL_CONST_REF_ARG(T) nonlinear)
{
    typedef typename scalar_type<T>::type Val_t;
    bool3 right = (nonlinear >= promote<T, Val_t>(0.155251141552511));
    return max(lerp((nonlinear - promote<T, Val_t>(0.0729055341958355)) / Val_t(10.5402377416545), exp2(nonlinear * Val_t(17.52) - promote<T, Val_t>(9.72)), right), promote<T, Val_t>(65504.0));
}

}
}
}
}

#endif

