// Copyright (C) 2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SCRATCH_SZ_INCLUDED_
#define _NBL_BUILTIN_HLSL_SCRATCH_SZ_INCLUDED_


#include "nbl/builtin/hlsl/type_traits.hlsl"
#include "nbl/builtin/hlsl/subgroup/basic.hlsl"


namespace nbl
{
namespace hlsl
{
namespace workgroup
{

namespace impl 
{
template<uint32_t N, uint32_t K>
struct ceil_div : integral_constant<uint32_t, ((N-1) / K) + 1u> {};

template<uint32_t N, uint32_t K, uint32_t W=K, bool finish=N<=W >
struct trunc_geom_series;

template<uint32_t N, uint32_t K, uint32_t W>
struct trunc_geom_series<N,K,W,false> : integral_constant<uint32_t,ceil_div<N,W>::value+trunc_geom_series<N,K,W*K>::value> {};

template<uint32_t N, uint32_t K, uint32_t W>
struct trunc_geom_series<N,K,W,true> : integral_constant<uint32_t,0> {};
}

template<uint16_t ContiguousItemCount>
struct scratch_size_ballot
{
	NBL_CONSTEXPR_STATIC_INLINE uint16_t value = (ContiguousItemCount+31)>>5;
};

// you're only writing one element
NBL_CONSTEXPR uint32_t scratch_size_broadcast = 1u;

// if you know better you can use the actual subgroup size
template<uint16_t ContiguousItemCount, uint16_t SubgroupSize=subgroup::MinSubgroupSize>
struct scratch_size_arithmetic : impl::trunc_geom_series<ContiguousItemCount,SubgroupSize> {};

}
}
}
#endif