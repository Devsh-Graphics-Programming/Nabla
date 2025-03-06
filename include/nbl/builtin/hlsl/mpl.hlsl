// Copyright (C) 2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MPL_INCLUDED_
#define _NBL_BUILTIN_HLSL_MPL_INCLUDED_


#include <nbl/builtin/hlsl/type_traits.hlsl>
#include <nbl/builtin/hlsl/bit.hlsl>


namespace nbl
{
namespace hlsl
{
namespace mpl
{

namespace impl
{

template<uint64_t N, uint16_t bits>
struct countl_zero
{
    NBL_CONSTEXPR_STATIC_INLINE uint64_t SHIFT = bits >> 1;
    NBL_CONSTEXPR_STATIC_INLINE uint64_t LO_MASK = (1ull << SHIFT) - 1;
    NBL_CONSTEXPR_STATIC_INLINE bool CHOOSE_HIGH = (N & (LO_MASK << SHIFT))!=0;
    NBL_CONSTEXPR_STATIC_INLINE uint64_t NEXT = (CHOOSE_HIGH ? (N >> SHIFT) : N) & LO_MASK;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t value = countl_zero<NEXT, SHIFT>::value + (CHOOSE_HIGH ? 0ull : SHIFT);
};

template<uint64_t N>
struct countl_zero<N, 1> : integral_constant<uint16_t, uint16_t(1u - (N & 1))>
{};

}

template<class T, T N>
struct countl_zero : impl::countl_zero<uint64_t(N), (sizeof(T) * 8)>
{
    static_assert(is_integral<T>::value, "countl_zero type parameter must be an integral type");
};
template<class T, T N>
NBL_CONSTEXPR T countl_zero_v = countl_zero<T,N>::value;

template<uint64_t X>
struct log2
{
    NBL_CONSTEXPR_STATIC_INLINE uint16_t value = X ? (1ull<<6)-countl_zero<uint64_t, X>::value-1 : -1ull;
};
template<uint64_t X>
NBL_CONSTEXPR uint64_t log2_v = log2<X>::value;

template<typename T, T X, int32_t S>
struct rotl
{
    NBL_CONSTEXPR_STATIC_INLINE uint32_t N = 32u;
    NBL_CONSTEXPR_STATIC_INLINE int32_t r = S % N;
    NBL_CONSTEXPR_STATIC_INLINE T value = (S >= 0) ? ((X << r) | (X >> (N - r))) : (X >> (-r)) | (X << (N - (-r)));
};
template<typename T, T X, int32_t S>
NBL_CONSTEXPR T rotl_v = rotl<T,X,S>::value;

template<typename T, T X, int32_t S>
struct rotr
{
    NBL_CONSTEXPR_STATIC_INLINE uint32_t N = 32u;
    NBL_CONSTEXPR_STATIC_INLINE int32_t r = S % N;
    NBL_CONSTEXPR_STATIC_INLINE T value = (S >= 0) ? ((X >> r) | (X << (N - r))) : (X << (-r)) | (X >> (N - (-r)));
};
template<typename T, T X, int32_t S>
NBL_CONSTEXPR T rotr_v = rotr<T,X,S>::value;

template<uint64_t X, uint64_t M>
struct align_up
{
    NBL_CONSTEXPR_STATIC_INLINE uint64_t value = (X+M-1)/M;
};
template<uint64_t X, uint64_t M>
NBL_CONSTEXPR uint64_t align_up_v = align_up<X,M>::value;

template<uint64_t N>
struct is_pot : bool_constant< (N > 0 && !(N & (N - 1))) > {};
template<uint64_t N>
NBL_CONSTEXPR bool is_pot_v = is_pot<N>::value;

template<typename T, T X, T Y>
struct max
{
    NBL_CONSTEXPR_STATIC_INLINE T value = X<Y ? Y:X;
};
template<typename T, T X, T Y>
NBL_CONSTEXPR T max_v = max<T,X,Y>::value;
}
}
}
#endif
