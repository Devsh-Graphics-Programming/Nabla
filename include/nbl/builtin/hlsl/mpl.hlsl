// Copyright (C) 2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_META_MATH_INCLUDED_
#define _NBL_BUILTIN_HLSL_META_MATH_INCLUDED_

namespace nbl
{
namespace hlsl
{
namespace mpl
{

#ifdef __HLSL_VERSION
namespace impl
{
template<uint64_t N, uint8_t bits_log2>
struct clz;

template<uint64_t N>
struct clz<N, 0>
{
    static const uint64_t value = 1 - (N & 1);
};

template<uint64_t N, uint8_t bits_log2>
struct clz
{
    static const uint64_t SHIFT = 1ull<<(bits_log2-1);
    static const uint64_t LO_MASK = (1ull<<SHIFT)-1;
    static const bool CHOOSE_HIGH = N&(LO_MASK<<SHIFT);
    static const uint64_t value   = clz<(CHOOSE_HIGH ? (N>>SHIFT):N)&LO_MASK,bits_log2-1>::value + (CHOOSE_HIGH ? 0ull:SHIFT);
};

}

template<uint64_t X>
struct clz
{
    static const uint64_t value = impl::clz<X, 6>::value;
};

template<uint64_t X>
struct consteval_log2
{
    static const uint64_t value = X ? (1ull<<6)-impl::clz<X,6>::value-1 : -1ull;
};

}
#endif


}
}

#endif
