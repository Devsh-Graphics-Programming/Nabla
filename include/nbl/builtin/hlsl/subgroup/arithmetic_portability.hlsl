// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_ARITHMETIC_PORTABILITY_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_ARITHMETIC_PORTABILITY_INCLUDED_


#include "nbl/builtin/hlsl/device_capabilities_traits.hlsl"

#include "nbl/builtin/hlsl/subgroup/basic.hlsl"
#include "nbl/builtin/hlsl/subgroup/arithmetic_portability_impl.hlsl"
#include "nbl/builtin/hlsl/concepts.hlsl"


namespace nbl
{
namespace hlsl
{
namespace subgroup
{

template<class Binop, class device_capabilities=void>
struct reduction : impl::reduction<Binop,device_capabilities_traits<device_capabilities>::shaderSubgroupArithmetic> {};
template<class Binop, class device_capabilities=void>
struct inclusive_scan : impl::inclusive_scan<Binop,device_capabilities_traits<device_capabilities>::shaderSubgroupArithmetic> {};
template<class Binop, class device_capabilities=void>
struct exclusive_scan : impl::exclusive_scan<Binop,device_capabilities_traits<device_capabilities>::shaderSubgroupArithmetic> {};

}

namespace subgroup2
{

template<typename Config, class BinOp, int32_t ItemsPerInvocation=1, class device_capabilities=void NBL_PRIMARY_REQUIRES(is_configuration_v<Config>)
struct ArithmeticParams
{
    using config_t = Config;
    using binop_t = BinOp;
    using type_t = typename BinOp::type_t;

    NBL_CONSTEXPR_STATIC_INLINE int32_t itemsPerInvocation = ItemsPerInvocation;
    NBL_CONSTEXPR_STATIC_INLINE bool UseNativeIntrinsics = device_capabilities_traits<device_capabilities>::shaderSubgroupArithmetic /*&& /*some heuristic for when its faster*/;
};

template<typename Params>
struct reduction : impl::reduction<typename Params::binop_t,Params::UseNativeIntrinsics> {};
template<typename Params>
struct inclusive_scan : impl::inclusive_scan<typename Params::binop_t,Params::UseNativeIntrinsics> {};
template<typename Params>
struct exclusive_scan : impl::exclusive_scan<typename Params::binop_t,Params::UseNativeIntrinsics> {};

}
}
}

#endif