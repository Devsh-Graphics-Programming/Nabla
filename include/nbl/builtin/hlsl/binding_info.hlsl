// Copyright (C) 2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BINDING_INFO_INCLUDED_
#define _NBL_BUILTIN_HLSL_BINDING_INFO_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

namespace nbl
{
namespace hlsl
{

template<uint32_t set, uint32_t ix, uint32_t count=1>
struct ConstevalBindingInfo
{
	NBL_CONSTEXPR_STATIC_INLINE uint32_t Set = set;
	NBL_CONSTEXPR_STATIC_INLINE uint32_t Index = ix;
	NBL_CONSTEXPR_STATIC_INLINE uint32_t Count = count;
};

// used for descriptor set layout lookups
struct SBindingInfo
{
	//! binding index for a given resource
	uint32_t binding : 29;
	//! descriptor set index for a resource
	uint32_t set : 3;
};

}
}
#endif
