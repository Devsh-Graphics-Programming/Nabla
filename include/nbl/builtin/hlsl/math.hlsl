// Copyright (C) 2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MATH_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat/type_traits.h"

// REVIEW-519: Review this whole header and content (whether it should be here or somewhere else)
namespace nbl
{
namespace hlsl
{
namespace math
{
template<uint32_t N, uint32_t K>
struct ceil_div : std::integral_constant<uint32_t, ((N-1) / K) + 1u> {};

template<uint32_t N, uint32_t K, uint32_t W=K, bool finish=N<=W >
struct trunc_geom_series;

template<uint32_t N, uint32_t K, uint32_t W>
struct trunc_geom_series<N,K,W,false> : std::integral_constant<uint32_t,ceil_div<N,W>::value+trunc_geom_series<N,K,W*K>::value> {};

template<uint32_t N, uint32_t K, uint32_t W>
struct trunc_geom_series<N,K,W,true> : std::integral_constant<uint32_t,0> {};
}
}
}
#endif