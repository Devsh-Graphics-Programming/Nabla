// Copyright (C) 2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MPL_INCLUDED_
#define _NBL_BUILTIN_HLSL_MPL_INCLUDED_

#ifdef __HLSL_VERSION
#include <nbl/builtin/hlsl/type_traits.hlsl>
#else
#include <bit>

#endif

namespace nbl
{
namespace hlsl
{
namespace mpl
{

#ifdef __HLSL_VERSION
namespace impl
{

// TODO: uncomment when fixed https://github.com/microsoft/DirectXShaderCompiler/issues/5859
// template<uint16_t bits_log2>
// struct countl_zero_masks
// {
//     static const uint16_t SHIFT = uint16_t(1)<<(bits_log2-1);
//     static const uint64_t LO_MASK = (1ull<<SHIFT)-1;
// };

// template<>
// struct countl_zero_masks<0>
// {
//     static const uint16_t SHIFT = 0;
//     static const uint64_t LO_MASK = 0;
// };

// TODO: this is temporary workaround, delete when fixed: https://github.com/microsoft/DirectXShaderCompiler/issues/5859
template<uint16_t bits_log2>
struct countl_zero_masks
{
    static const uint16_t SHIFT   = bits_log2 ? uint16_t(1)<<(bits_log2-1) : 0;
    static const uint64_t LO_MASK = bits_log2 ? (1ull<<SHIFT)-1 : 0;
};

template<uint64_t N, uint16_t bits_log2>
struct countl_zero
{
    static const bool CHOOSE_HIGH = N&(countl_zero_masks<bits_log2>::LO_MASK<<countl_zero_masks<bits_log2>::SHIFT);
    static const uint64_t NEXT_N = (CHOOSE_HIGH ? (N>>countl_zero_masks<bits_log2>::SHIFT):N)&countl_zero_masks<bits_log2>::LO_MASK;
    static const uint16_t value   = type_traits::conditional<bits_log2,countl_zero<NEXT_N,bits_log2-1>,type_traits::integral_constant<uint16_t,0> >::type::value + (CHOOSE_HIGH ? 0ull:countl_zero_masks<bits_log2>::SHIFT);
};

}
#endif

template<uint64_t N>
struct countl_zero
{
    NBL_CONSTEXPR_STATIC_INLINE uint16_t value =
#ifdef __HLSL_VERSION
      impl::countl_zero<N, 6>::value;
#else
      std::countl_zero(N);
#endif
};

template<uint64_t X>
struct log2
{
    NBL_CONSTEXPR_STATIC_INLINE uint16_t value = X ? (1ull<<6)-countl_zero<X>::value-1 : -1ull;
};

}
}
}

#endif
