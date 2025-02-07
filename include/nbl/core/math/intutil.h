// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

// TODO: kill this file
#ifndef __NBL_CORE_MATH_INTUTIL_H_INCLUDED__
#define __NBL_CORE_MATH_INTUTIL_H_INCLUDED__

#include "nbl/builtin/hlsl/math/intutil.hlsl"

#include "nbl/builtin/hlsl/cpp_compat/intrinsics.h"
#include "nbl/macros.h"
#include "nbl/core/math/glslFunctions.h"

#include <cstdint>
#include <limits.h> // For INT_MAX / UINT_MAX
#include <initializer_list>
#include <type_traits>
#ifdef _MSC_VER
    #include <intrin.h>
#endif

namespace nbl
{
namespace core
{

template<typename INT_TYPE>
[[deprecated("Use the nbl::hlsl version in builtin/hlsl/math/intutil.hlsl")]] NBL_FORCE_INLINE constexpr bool isNPoT(INT_TYPE value)
{
    return hlsl::isNPoT<INT_TYPE>(value);
}

template<typename INT_TYPE>
[[deprecated("Use the nbl::hlsl version in builtin/hlsl/math/intutil.hlsl")]] 
NBL_FORCE_INLINE constexpr bool isPoT(INT_TYPE value)
{
    return hlsl::isPoT<INT_TYPE>(value);
}


template<typename INT_TYPE>
[[deprecated("Use the nbl::hlsl version in builtin/hlsl/math/intutil.hlsl")]] 
NBL_FORCE_INLINE constexpr INT_TYPE roundUpToPoT(INT_TYPE value)
{
    return hlsl::roundUpToPoT<INT_TYPE>(value); // this wont result in constexpr because findMSB is not one
}

template<typename INT_TYPE>
[[deprecated("Use the nbl::hlsl version in builtin/hlsl/math/intutil.hlsl")]] 
NBL_FORCE_INLINE constexpr INT_TYPE roundDownToPoT(INT_TYPE value)
{
    return hlsl::roundDownToPoT<INT_TYPE>(value);
}

template<typename INT_TYPE>
[[deprecated("Use the nbl::hlsl version in builtin/hlsl/math/intutil.hlsl")]] 
NBL_FORCE_INLINE constexpr INT_TYPE roundUp(INT_TYPE value, INT_TYPE multiple)
{
    return hlsl::roundUp<INT_TYPE>(value, multiple);
}

template<typename INT_TYPE>
[[deprecated("Use the nbl::hlsl version in builtin/hlsl/math/intutil.hlsl")]] 
NBL_FORCE_INLINE constexpr INT_TYPE align(INT_TYPE alignment, INT_TYPE size, INT_TYPE& address, INT_TYPE& space)
{
    return hlsl::align<INT_TYPE>(alignment, size, address, space);
}

//! Get bitmask from variadic arguments passed. 
/*
    For example if you were to create bitmask for vertex attributes
    having positions inteeger set as 0, colors as 1 and normals
    as 3, just pass them to it and use the value returned.
*/

template<typename BITMASK_TYPE>
[[deprecated("Use the nbl::hlsl version in builtin/hlsl/math/intutil.hlsl")]] 
NBL_FORCE_INLINE constexpr uint64_t createBitmask(std::initializer_list<BITMASK_TYPE> initializer)
{
    return hlsl::createBitmask<BITMASK_TYPE>(initializer);
}

} // end namespace core
} // end namespace nbl

#endif

