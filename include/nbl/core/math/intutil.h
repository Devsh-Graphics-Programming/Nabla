// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

// TODO: kill this file
#ifndef __NBL_MATH_H_INCLUDED__
#define __NBL_MATH_H_INCLUDED__

#include "BuildConfigOptions.h"

#include "nbl/macros.h"
#include "nbl/core/math/glslFunctions.h"
#include "nbl/builtin/hlsl/cpp_compat.hlsl"

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
NBL_FORCE_INLINE constexpr bool isNPoT(INT_TYPE value)
{
    static_assert(std::is_integral<INT_TYPE>::value, "Integral required.");
    return value & (value - static_cast<INT_TYPE>(1));
}

template<typename INT_TYPE>
NBL_FORCE_INLINE constexpr bool isPoT(INT_TYPE value)
{
    return !isNPoT<INT_TYPE>(value);
}


template<typename INT_TYPE>
NBL_FORCE_INLINE constexpr INT_TYPE roundUpToPoT(INT_TYPE value)
{
        return INT_TYPE(0x1u)<<INT_TYPE(1+hlsl::findMSB<INT_TYPE>(value-INT_TYPE(1)));
}

template<typename INT_TYPE>
NBL_FORCE_INLINE constexpr INT_TYPE roundDownToPoT(INT_TYPE value)
{
    return INT_TYPE(0x1u)<<hlsl::findMSB<INT_TYPE>(value);
}

template<typename INT_TYPE>
NBL_FORCE_INLINE constexpr INT_TYPE roundUp(INT_TYPE value, INT_TYPE multiple)
{
    INT_TYPE tmp = (value+multiple-1u)/multiple;
    return tmp*multiple;
}

template<typename INT_TYPE>
NBL_FORCE_INLINE constexpr INT_TYPE align(INT_TYPE alignment, INT_TYPE size, INT_TYPE& address, INT_TYPE& space)
{
    INT_TYPE nextAlignedAddr = roundUp<INT_TYPE>(address,alignment);

    INT_TYPE spaceDecrement = nextAlignedAddr-address;
    if (spaceDecrement>space)
        return 0u;

    INT_TYPE newSpace = space-spaceDecrement;
    if (size>newSpace)
        return 0u;

    space = newSpace;
    return address = nextAlignedAddr;
}

//! Get bitmask from variadic arguments passed. 
/*
    For example if you were to create bitmask for vertex attributes
    having positions inteeger set as 0, colors as 1 and normals
    as 3, just pass them to it and use the value returned.
*/

template<typename BITMASK_TYPE>
NBL_FORCE_INLINE constexpr uint64_t createBitmask(std::initializer_list<BITMASK_TYPE> initializer)
{
    static_assert(std::is_integral<BITMASK_TYPE>::value || std::is_enum<BITMASK_TYPE>::value, "Integral or enum required.");
    uint64_t retval {};
    for (const auto& it : initializer)
        retval |= (1ull << it);
    return retval;
}

} // end namespace core
} // end namespace nbl

#endif

