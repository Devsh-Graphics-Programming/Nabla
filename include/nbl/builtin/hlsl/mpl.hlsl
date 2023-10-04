// Copyright (C) 2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_META_MATH_INCLUDED_
#define _NBL_BUILTIN_HLSL_META_MATH_INCLUDED_

#include <nbl/builtin/hlsl/type_traits.hlsl>


namespace nbl
{
namespace hlsl
{
namespace mpl
{

#ifdef __HLSL_VERSION
namespace impl
{

template<uint16_t bits_log2>
struct clz_masks
{
    static const uint16_t SHIFT = uint16_t(1)<<(bits_log2-1);
    static const uint64_t LO_MASK = (1ull<<SHIFT)-1;
};

template<>
struct clz_masks<0>
{
    static const uint16_t SHIFT = 0;
    static const uint64_t LO_MASK = 0;
};

template<uint64_t N, uint16_t bits_log2>
struct clz
{
    static const bool CHOOSE_HIGH = N&(clz_masks<bits_log2>::LO_MASK<<clz_masks<bits_log2>::SHIFT);
    static const uint64_t NEXT_N = (CHOOSE_HIGH ? (N>>clz_masks<bits_log2>::SHIFT):N)&clz_masks<bits_log2>::LO_MASK;
    static const uint16_t value   = type_traits::conditional<bits_log2,clz<NEXT_N,bits_log2-1>,type_traits::integral_constant<uint16_t,0> >::type::value + (CHOOSE_HIGH ? 0ull:clz_masks<bits_log2>::SHIFT);
};

}

template<uint64_t N>
struct clz
{
    static const uint16_t value = impl::clz<N, 6>::value;
};

template<uint64_t X>
struct consteval_log2
{
    static const uint16_t value = X ? (1ull<<6)-clz<X>::value-1 : -1ull;
};

#endif

}
}
}

#endif
