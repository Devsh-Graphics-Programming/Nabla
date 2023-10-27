// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_ARITHMETIC_PORTABILITY_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_ARITHMETIC_PORTABILITY_INCLUDED_


#include "nbl/builtin/hlsl/subgroup/basic.hlsl"

#include "nbl/builtin/hlsl/subgroup/arithmetic_portability_impl.hlsl"


namespace nbl
{
namespace hlsl
{
namespace subgroup
{

#ifdef NBL_GLSL_LIMIT_SHADER_SUBGROUP_ARITHMETIC
#define IMPL native
#else
#define IMPL portability
#endif

template<class Binop>
struct reduction : IMPL::reduction<Binop> {};
template<class Binop>
struct inclusive_scan : IMPL::inclusive_scan<Binop> {};
template<class Binop>
struct exclusive_scan : IMPL::exclusive_scan<Binop> {};

#undef IMPL

}
}
}

#endif