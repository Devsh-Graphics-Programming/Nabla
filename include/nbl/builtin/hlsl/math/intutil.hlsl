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
NBL_CONSTEXPR_FORCED_INLINE_FUNC Integer roundUpToPoT(Integer value)
{
    return Integer(0x1u) << Integer(1 + hlsl::findMSB<Integer>(value - Integer(1))); // this wont result in constexpr because findMSB is not one
}

template<typename Integer NBL_FUNC_REQUIRES(is_integral_v<Integer>)
NBL_CONSTEXPR_FORCED_INLINE_FUNC Integer roundDownToPoT(Integer value)
{
    return Integer(0x1u) << hlsl::findMSB<Integer>(value);
}

template<typename Integer NBL_FUNC_REQUIRES(is_integral_v<Integer>)
NBL_CONSTEXPR_FORCED_INLINE_FUNC Integer roundUp(Integer value, Integer multiple)
{
    Integer tmp = (value + multiple - 1u) / multiple;
    return tmp * multiple;
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

#ifndef __HLSL_VERSION

// Have to wait for the HLSL patch for `is_enum`. Would also have to figure out how to do it without initializer lists for HLSL use. 

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

