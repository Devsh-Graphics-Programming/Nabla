// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_MATH_H_INCLUDED__
#define __IRR_MATH_H_INCLUDED__

#include "IrrCompileConfig.h"

#include <limits.h> // For INT_MAX / UINT_MAX
#include <type_traits>
#ifdef _MSC_VER
    #include <intrin.h>
#endif

#include "irr/macros.h"
#include "irr/core/math/glslFunctions.h"

namespace irr
{
namespace core
{


template<typename INT_TYPE>
IRR_FORCE_INLINE constexpr bool isNPoT(INT_TYPE value)
{
    static_assert(std::is_integral<INT_TYPE>::value, "Integral required.");
    return value & (value - static_cast<INT_TYPE>(1));
}

template<typename INT_TYPE>
IRR_FORCE_INLINE constexpr bool isPoT(INT_TYPE value)
{
    return !isNPoT<INT_TYPE>(value);
}


template<typename INT_TYPE>
IRR_FORCE_INLINE constexpr INT_TYPE roundUpToPoT(INT_TYPE value)
{
        return INT_TYPE(0x1u)<<INT_TYPE(1+core::findMSB<INT_TYPE>(value-INT_TYPE(1)));
}

template<typename INT_TYPE>
IRR_FORCE_INLINE constexpr INT_TYPE roundDownToPoT(INT_TYPE value)
{
    return INT_TYPE(0x1u)<<core::findLSB<INT_TYPE>(value);
}

template<typename INT_TYPE>
IRR_FORCE_INLINE constexpr INT_TYPE roundUp(INT_TYPE value, INT_TYPE multiple)
{
    INT_TYPE tmp = (value+multiple-1u)/multiple;
    return tmp*multiple;
}

template<typename INT_TYPE>
IRR_FORCE_INLINE constexpr INT_TYPE align(INT_TYPE alignment, INT_TYPE size, INT_TYPE& address, INT_TYPE& space)
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
IRR_FORCE_INLINE constexpr uint8_t createBitmask(std::initializer_list<BITMASK_TYPE> initializer)
{
    static_assert(std::is_integral<BITMASK_TYPE>::value || std::is_enum<BITMASK_TYPE>::value, "Integral or enum required.");
    uint8_t retval {};
    for (const auto& it : initializer)
        retval |= (1 << it);
    return retval;
}

} // end namespace core
} // end namespace irr

#endif

