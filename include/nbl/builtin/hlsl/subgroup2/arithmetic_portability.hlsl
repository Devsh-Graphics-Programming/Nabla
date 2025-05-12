// Copyright (C) 2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP2_ARITHMETIC_PORTABILITY_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP2_ARITHMETIC_PORTABILITY_INCLUDED_


#include "nbl/builtin/hlsl/device_capabilities_traits.hlsl"

#include "nbl/builtin/hlsl/subgroup2/ballot.hlsl"
#include "nbl/builtin/hlsl/subgroup2/arithmetic_portability_impl.hlsl"
#include "nbl/builtin/hlsl/concepts.hlsl"


namespace nbl
{
namespace hlsl
{
namespace subgroup2
{

template<typename Config, class BinOp, int32_t _ItemsPerInvocation=1, class device_capabilities=void NBL_PRIMARY_REQUIRES(is_configuration_v<Config> && is_scalar_v<typename BinOp::type_t>)
struct ArithmeticParams
{
    using config_t = Config;
    using binop_t = BinOp;
    using scalar_t = typename BinOp::type_t;
    using type_t = vector<scalar_t, _ItemsPerInvocation>;
    using device_traits = device_capabilities_traits<device_capabilities>;

    NBL_CONSTEXPR_STATIC_INLINE int32_t ItemsPerInvocation = _ItemsPerInvocation;
    NBL_CONSTEXPR_STATIC_INLINE bool UseNativeIntrinsics = device_capabilities_traits<device_capabilities>::shaderSubgroupArithmetic /*&& /*some heuristic for when its faster*/;
    // TODO add a IHV enum to device_capabilities_traits to check !is_nvidia
};

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
