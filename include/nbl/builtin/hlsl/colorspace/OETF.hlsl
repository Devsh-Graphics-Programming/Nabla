
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_COLOR_SPACE_OETF_INCLUDED_
#define _NBL_BUILTIN_HLSL_COLOR_SPACE_OETF_INCLUDED_

//#include <nbl/builtin/hlsl/common.hlsl>

namespace nbl
{
namespace hlsl
{
namespace colorspace
{
namespace oetf
{

template<typename T>
T identity(typename NBL_CONST_REF_ARG(T) _linear)
{
    return _linear;
}

template<typename T>
T impl_shared_2_4(typename NBL_CONST_REF_ARG(T) _linear, typename ValueType<T>::type vertex)
{
    using Val_t = ValueType<T>::type;
    bool3 right = (_linear > promote<T, Val_t>(vertex));
    return lerp(_linear * Val_t(12.92), pow(_linear, promote<T, Val_t>(1.0 / 2.4)) * Val_t(1.055) - (Val_t(0.055)), right);
}

// compatible with scRGB as well
template<typename T>
T sRGB(typename NBL_CONST_REF_ARG(T) _linear)
{
    using Val_t = ValueType<T>::type;
    bool3 negatif = (_linear < promote<T, Val_t>(0.0));
    T absVal = impl_shared_2_4<T>(abs(_linear), 0.0031308);
    return lerp(absVal, -absVal, negatif);
}

// also known as P3-D65
template<typename T>
T Display_P3(typename NBL_CONST_REF_ARG(T) _linear)
{
    return impl_shared_2_4<T>(_linear, 0.0030186);
}

template<typename T>
T DCI_P3_XYZ(typename NBL_CONST_REF_ARG(T) _linear)
{
    using Val_t = ValueType<T>::type;
    return pow(_linear / Val_t(52.37), promote<T, Val_t>(1.0 / 2.6));
}

template<typename T>
T SMPTE_170M(typename NBL_CONST_REF_ARG(T) _linear)
{
    using Val_t = ValueType<T>::type;
    // ITU specs (and the outlier BT.2020) give different constants for these, but they introduce discontinuities in the mapping
    // because HDR swapchains often employ the RGBA16_SFLOAT format, this would become apparent because its higher precision than 8,10,12 bits
    const Val_t alpha = 1.099296826809443; // 1.099 for all ITU but the BT.2020 12 bit encoding, 1.0993 otherwise
    const T beta = promote<T, Val_t>(0.018053968510808); // 0.0181 for all ITU but the BT.2020 12 bit encoding, 0.18 otherwise
    return lerp(_linear * Val_t(4.5), pow(_linear, promote<T, Val_t>(0.45)) * alpha - promote<T, Val_t>(alpha - 1.0), (_linear >= beta));
}

template<typename T>
T SMPTE_ST2084(typename NBL_CONST_REF_ARG(T) _linear)
{
    using Val_t = ValueType<T>::type;
    const T m1 = promote<T, Val_t>(0.1593017578125);
    const T m2 = promote<T, Val_t>(78.84375);
    const Val_t c2 = 18.8515625;
    const Val_t c3 = 18.68875;
    const T c1 = promote<T, Val_t>(c3 - c2 + 1.0);

    T L_m1 = pow(_linear, m1);
    return pow((c1 + L_m1 * c2) / (promote<T, Val_t>(1.0) + L_m1 * c3), m2);
}

// did I do this right by applying the function for every color?
template<typename T>
T HDR10_HLG(typename NBL_CONST_REF_ARG(T) _linear)
{
    using Val_t = ValueType<T>::type;
    
    // done with log2 so constants are different
    const Val_t a = 0.1239574303172;
    const T b = promote<T, Val_t>(0.02372241);
    const T c = promote<T, Val_t>(1.0042934693729);
    bool3 right = (_linear > promote<T, Val_t>(1.0 / 12.0));
    return lerp(sqrt(_linear * Val_t(3.0)), log2(_linear - b) * a + c, right);
}

template<typename T>
T AdobeRGB(typename NBL_CONST_REF_ARG(T) _linear)
{
    using Val_t = ValueType<T>::type;
    return pow(_linear, promote<T, Val_t>(1.0 / 2.19921875));
}

template<typename T>
T Gamma_2_2(typename NBL_CONST_REF_ARG(T) _linear)
{
    using Val_t = ValueType<T>::type;
    return pow(_linear, promote<T, Val_t>(1.0 / 2.2));
}

template<typename T>
T ACEScc(typename NBL_CONST_REF_ARG(T) _linear)
{
    using Val_t = ValueType<T>::type;
    bool3 mid = (_linear >= promote<T, Val_t>(0.0));
    bool3 right = (_linear >= promote<T, Val_t>(0.000030517578125));
    return (log2(lerp(promote<T, Val_t>(0.0000152587890625), promote<T, Val_t>(0.0), right) + _linear * lerp(promote<T, Val_t>(0.0), lerp(promote<T, Val_t>(0.5), promote<T, Val_t>(1.0), right), mid)) + promote<T, Val_t>(9.72)) / Val_t(17.52);
}

template<typename T>
T ACEScct(typename NBL_CONST_REF_ARG(T) _linear)
{
    using Val_t = ValueType<T>::type;
    bool3 right = (_linear > promote<T, Val_t>(0.0078125));
    return lerp(Val_t(10.5402377416545) * _linear + Val_t(0.0729055341958355), (log2(_linear) + promote<T, Val_t>(9.72)) / Val_t(17.52), right);
}

}
}
}
}

#endif