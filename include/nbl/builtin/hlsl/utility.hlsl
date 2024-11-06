// Copyright (C) 2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_UTILITY_INCLUDED_
#define _NBL_BUILTIN_HLSL_UTILITY_INCLUDED_


#include <nbl/builtin/hlsl/type_traits.hlsl>


// for now we only implement declval
namespace nbl
{
namespace hlsl
{
#ifndef __HLSL_VERSION

template<class T>
std::add_rvalue_reference_t<T> declval() noexcept
{
	static_assert(false,"Actually calling declval is ill-formed.");
}

#else

namespace experimental
{

template<class T>
T declval() {}

}

#endif
}
}

#endif
