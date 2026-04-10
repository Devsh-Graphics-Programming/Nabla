#ifndef _NBL_BUILTIN_HLSL_MATH_INTUTIL_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_INTUTIL_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/type_traits.hlsl"

namespace nbl
{
namespace hlsl
{

template<typename Integer NBL_FUNC_REQUIRES(is_integral_v<Integer>)
NBL_CONSTEXPR_FORCED_INLINE_FUNC bool isNPoT(Integer value)
{
    return value & (value - Integer(1));
}

template<typename Integer NBL_FUNC_REQUIRES(is_integral_v<Integer>)
NBL_CONSTEXPR_FORCED_INLINE_FUNC bool isPoT(Integer value)
{
    return !isNPoT<Integer>(value);
}

template<typename Integer NBL_FUNC_REQUIRES(is_integral_v<Integer>)
// Returns ceiled log2
NBL_CONSTEXPR_FORCED_INLINE_FUNC Integer log2ceil(Integer value)
{
    return Integer(1 + hlsl::findMSB<Integer>(value - Integer(1))); 
}

template<typename Integer NBL_FUNC_REQUIRES(is_integral_v<Integer>)
NBL_CONSTEXPR_FORCED_INLINE_FUNC Integer roundUpToPoT(Integer value)
{
    return Integer(0x1u) << log2ceil(value); 
}

template<typename Integer NBL_FUNC_REQUIRES(is_integral_v<Integer>)
NBL_CONSTEXPR_FORCED_INLINE_FUNC Integer roundDownToPoT(Integer value)
{
    return Integer(0x1u) << hlsl::findMSB<Integer>(value);
}

template<typename Integer NBL_FUNC_REQUIRES(is_integral_v<Integer>)
NBL_CONSTEXPR_FORCED_INLINE_FUNC Integer ceilDiv(Integer dividend, Integer divisor)
{
    return (dividend + divisor - 1) / divisor;
}

template<typename Integer NBL_FUNC_REQUIRES(is_integral_v<Integer>)
NBL_CONSTEXPR_FORCED_INLINE_FUNC Integer roundUp(Integer value, Integer multiple)
{
    return ceilDiv(value, multiple) * multiple;
}

template<typename Integer NBL_FUNC_REQUIRES(is_integral_v<Integer>)
NBL_CONSTEXPR_FORCED_INLINE_FUNC Integer align(Integer alignment, Integer size, NBL_REF_ARG(Integer) address, NBL_REF_ARG(Integer) space)
{
    Integer nextAlignedAddr = roundUp<Integer>(address, alignment);

    Integer spaceDecrement = nextAlignedAddr - address;
    if (spaceDecrement > space)
        return 0u;

    Integer newSpace = space - spaceDecrement;
    if (size > newSpace)
        return 0u;

    space = newSpace;
    return address = nextAlignedAddr;
}

// Bitshift utils
// TODO: These can be expanded to shift by more than just one position at a time
// TODO: Can be made to wrok on uint64_t

// Given an N-bit number stored as uint32_t, performs a circular bit shift right on the upper H bits
template<uint16_t N, uint16_t H>
enable_if_t<(1 < H) && (H <= N) && (N <= 32), uint32_t> circularBitShiftRightHigher(uint32_t i)
{
    // Highest H bits are numbered N-1 through N - H
    // N - H is then the middle bit
    // Lowest bits numbered from 0 through N - H - 1
    NBL_CONSTEXPR_FUNC_SCOPE_VAR uint32_t lowMask = (1 << (N - H)) - 1;
    NBL_CONSTEXPR_FUNC_SCOPE_VAR uint32_t midMask = 1 << (N - H);
    NBL_CONSTEXPR_FUNC_SCOPE_VAR uint32_t highMask = ~(lowMask | midMask);

    uint32_t low = i & lowMask;
    uint32_t mid = i & midMask;
    uint32_t high = i & highMask;

    high >>= 1;
    mid <<= H - 1;

    return mid | high | low;
}

// Given an N-bit number stored as uint32_t, performs a circular bit shift left on the upper H bits
template<uint16_t N, uint16_t H>
enable_if_t<(1 < H) && (H <= N) && (N < 32), uint32_t> circularBitShiftLeftHigher(uint32_t i)
{
    // Highest H bits are numbered N-1 through N - H
    // N - 1 is then the highest bit, and N - 2 through N - H are the middle bits
    // Lowest bits numbered from 0 through N - H - 1
    NBL_CONSTEXPR_FUNC_SCOPE_VAR uint32_t lowMask = (1 << (N - H)) - 1;
    NBL_CONSTEXPR_FUNC_SCOPE_VAR uint32_t highMask = 1 << (N - 1);
    NBL_CONSTEXPR_FUNC_SCOPE_VAR uint32_t midMask = ~(lowMask | highMask);

    uint32_t low = i & lowMask;
    uint32_t mid = i & midMask;
    uint32_t high = i & highMask;

    mid <<= 1;
    high >>= H - 1;

    return mid | high | low;
}
// Perform a circular bit shift right on the lower L bits of a number
template<uint16_t L>
enable_if_t<(1 < L), uint32_t> circularBitShiftRightLower(uint32_t i)
{
    // Lowest bit is indexed 0
    // Middle bits numbered 1 to L-1
    // Highest bits numbered from L through N-1
    NBL_CONSTEXPR_FUNC_SCOPE_VAR uint32_t lowMask = 1;
    NBL_CONSTEXPR_FUNC_SCOPE_VAR uint32_t midMask = ((1 << L) - 1) ^ 1;
    NBL_CONSTEXPR_FUNC_SCOPE_VAR uint32_t highMask = ~(lowMask | midMask);

    uint32_t low = i & lowMask;
    uint32_t mid = i & midMask;
    uint32_t high = i & highMask;

    low <<= L - 1;
    mid >>= 1;

    return high | low | mid;
}

// Perform a circular bit shift left on the lower L bits of a number
template<uint16_t L>
enable_if_t<(1 < L), uint32_t> circularBitShiftLeftLower(uint32_t i)
{
    // Lowest L - 1 bits numbered 0 through L - 2
    // L - 1 is then the middle bit
    // L through N-1 the higher bits
    NBL_CONSTEXPR_FUNC_SCOPE_VAR uint32_t lowMask = (1 << (L - 1)) - 1;
    NBL_CONSTEXPR_FUNC_SCOPE_VAR uint32_t midMask = 1 << (L - 1);
    NBL_CONSTEXPR_FUNC_SCOPE_VAR uint32_t highMask = ~(lowMask | midMask);

    uint32_t low = i & lowMask;
    uint32_t mid = i & midMask;
    uint32_t high = i & highMask;

    low <<= 1;
    mid >>= L - 1;

    return high | low | mid;
}

// ------------------------------------- CPP ONLY ----------------------------------------------------------
#ifndef __HLSL_VERSION

//! Get bitmask from variadic arguments passed. 
/*
    For example if you were to create bitmask for vertex attributes
    having positions inteeger set as 0, colors as 1 and normals
    as 3, just pass them to it and use the value returned.
*/

template<typename BitmaskType NBL_FUNC_REQUIRES(is_integral_v<BitmaskType> || std::is_enum_v<BitmaskType>)
NBL_CONSTEXPR_FORCED_INLINE_FUNC uint64_t createBitmask(std::initializer_list<BitmaskType> initializer)
{
    uint64_t retval{};
    for (const auto& it : initializer)
        retval |= (1ull << it);
    return retval;
}

#endif

} // end namespace hlsl
} // end namespace nbl


#endif

