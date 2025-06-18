// Copyright (C) 2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP2_ARITHMETIC_PORTABILITY_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP2_ARITHMETIC_PORTABILITY_INCLUDED_


#include "nbl/builtin/hlsl/device_capabilities_traits.hlsl"

#include "nbl/builtin/hlsl/subgroup2/ballot.hlsl"
#include "nbl/builtin/hlsl/subgroup2/arithmetic_portability_impl.hlsl"

namespace nbl
{
namespace hlsl
{
namespace subgroup2
{

template<typename Params>
struct reduction : impl::reduction<Params,typename Params::binop_t,Params::ItemsPerInvocation,Params::UseNativeIntrinsics> {};
template<typename Params>
struct inclusive_scan : impl::inclusive_scan<Params,typename Params::binop_t,Params::ItemsPerInvocation,Params::UseNativeIntrinsics> {};
template<typename Params>
struct exclusive_scan : impl::exclusive_scan<Params,typename Params::binop_t,Params::ItemsPerInvocation,Params::UseNativeIntrinsics> {};

}
}
}

#endif
