// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_ARITHMETIC_PORTABILITY_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_ARITHMETIC_PORTABILITY_INCLUDED_


#include "nbl/builtin/hlsl/subgroup/basic.hlsl"

#include "nbl/builtin/hlsl/subgroup/arithmetic_portability_impl.hlsl"
#include "nbl/builtin/hlsl/jit/device_capabilities.hlsl"


namespace nbl
{
namespace hlsl
{
namespace subgroup
{

// TODO: change to alias using template when DXC finally fixes SPIR-V codegen impartity
//template<class Binop, class device_capability_traits/*=TODO: Atil give us the traits so we can default with `void`!*/>
//struct reduction : impl::reduction<Binop,device_capability_traits::subgroupArithmetic> {};

template<class Binop>
struct reduction : impl::reduction<Binop,nbl::hlsl::jit::device_capabilities::subgroupArithmetic> {};
template<class Binop>
struct inclusive_scan : impl::inclusive_scan<Binop,nbl::hlsl::jit::device_capabilities::subgroupArithmetic> {};
template<class Binop>
struct exclusive_scan : impl::exclusive_scan<Binop,nbl::hlsl::jit::device_capabilities::subgroupArithmetic> {};

}
}
}

#endif