// Copyright (C) 2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MPL_INCLUDED_
#define _NBL_BUILTIN_HLSL_MPL_INCLUDED_

#ifdef __HLSL_VERSION
#include <nbl/builtin/hlsl/type_traits.hlsl>
#endif

namespace nbl
{
namespace hlsl
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
    static const uint16_t value   = conditional<bits_log2,clz<NEXT_N,bits_log2-1>,integral_constant<uint16_t,0> >::type::value + (CHOOSE_HIGH ? 0ull:clz_masks<bits_log2>::SHIFT);
};

template<class T> float16_t asfloat16_t(T val) { return asfloat16(val); }
template<class T> float32_t asfloat32_t(T val) { return asfloat(val); }
template<class T> int16_t   asint16_t(T val) { return asint16(val); }
template<class T> int32_t   asint32_t(T val) { return asint(val); }
template<class T> uint16_t  asuint16_t(T val) { return asuint16(val); }
template<class T> uint32_t  asuint32_t(T val) { return asuint(val); }

template<class T>
float64_t asfloat64_t(T val) 
{ 
    uint64_t us = uint64_t(val);
    return asdouble(uint32_t(val & ~0u), uint32_t((val >> 32u) & ~0u)); 
}

template<>
float64_t asfloat64_t<float64_t>(float64_t val) { return val; }

template<class T> uint64_t asuint64_t(T val) {  return val; }
template<class T> int64_t asint64_t(T val) {  return asuint64_t(val); }

template<>
uint64_t asuint64_t<float64_t>(float64_t val) 
{ 
    uint32_t lo, hi;
    asuint(val, lo, hi);
    return (uint64_t(hi) << 32u) | uint64_t(lo);
}

}

template<uint64_t N>
struct clz
{
    static const uint16_t value = impl::clz<N, 6>::value;
};

template<uint64_t X>
struct log2
{
    static const uint16_t value = X ? (1ull<<6)-clz<X>::value-1 : -1ull;
};


template<class T, class U = T>
T bit_cast(U val);

#define NBL_DECLARE_BIT_CAST(FROM, TO) template<> TO bit_cast<TO, FROM>(FROM val) { return impl::as##TO (val); }

#define NBL_DECLARE_BIT_CAST_TYPES(BASE, BITS) \
    NBL_DECLARE_BIT_CAST(BASE ## BITS, float ## BITS) \
    NBL_DECLARE_BIT_CAST(BASE ## BITS, uint ## BITS) \
    NBL_DECLARE_BIT_CAST(BASE ## BITS, int ## BITS)

#define NBL_DECLARE_BIT_CAST_BITS(BASE) \
    NBL_DECLARE_BIT_CAST_TYPES(BASE, 16_t) \
    NBL_DECLARE_BIT_CAST_TYPES(BASE, 32_t) \
    NBL_DECLARE_BIT_CAST_TYPES(BASE, 64_t)

NBL_DECLARE_BIT_CAST_BITS(float)
NBL_DECLARE_BIT_CAST_BITS(uint)
NBL_DECLARE_BIT_CAST_BITS(int)

#undef NBL_DECLARE_BIT_CAST
#undef NBL_DECLARE_BIT_CAST_TYPES
#undef NBL_DECLARE_BIT_CAST_BITS

#else

template<class T, class U>
constexpr T bit_cast(const U& val)
{
    return std::bit_cast<T, U>(val);
}

#endif

}
}

#endif
