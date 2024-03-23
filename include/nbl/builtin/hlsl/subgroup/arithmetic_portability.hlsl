// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_ARITHMETIC_PORTABILITY_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_ARITHMETIC_PORTABILITY_INCLUDED_


#include "nbl/builtin/hlsl/device_capabilities_traits.hlsl"

#include "nbl/builtin/hlsl/subgroup/basic.hlsl"
#include "nbl/builtin/hlsl/subgroup/arithmetic_portability_impl.hlsl"


namespace nbl
{
namespace hlsl
{
namespace subgroup
{

template<class Binop, class device_capabilities=void>
struct reduction : impl::reduction<Binop,device_capabilities_traits<device_capabilities>::subgroupArithmetic> {};
template<class Binop, class device_capabilities=void>
struct inclusive_scan : impl::inclusive_scan<Binop,device_capabilities_traits<device_capabilities>::subgroupArithmetic> {};
template<class Binop, class device_capabilities=void>
struct exclusive_scan : impl::exclusive_scan<Binop,device_capabilities_traits<device_capabilities>::subgroupArithmetic> {};

}
}
}

#endif