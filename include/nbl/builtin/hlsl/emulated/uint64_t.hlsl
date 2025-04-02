#ifndef _NBL_BUILTIN_HLSL_EMULATED_UINT64_T_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_EMULATED_UINT64_T_HLSL_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/functional.hlsl"

namespace nbl 
{
namespace hlsl
{

struct emulated_uint64_t
{
	using storage_t = vector<uint32_t, 2>;
	using this_t = emulated_uint64_t;

	storage_t data;

    // ---------------------------------------------------- CONSTRUCTORS ---------------------------------------------------------------

    
    #ifndef __HLSL_VERSION

    emulated_uint64_t() = default;

    // To immediately get compound operators and functional structs in CPP side
    explicit emulated_uint64_t(const storage_t _data) : data(_data) {}

    #endif

    /**
    * @brief Creates an `emulated_uint64_t` from a vector of two `uint32_t`s representing its bitpattern
    *
    * @param [in] _data Vector of `uint32_t` encoding the `uint64_t` being emulated
    */
    NBL_CONSTEXPR_STATIC_FUNC this_t create(NBL_CONST_REF_ARG(storage_t) _data)
    {
        this_t retVal;
        retVal.data = _data;
        return retVal;
    }

    /**
    * @brief Creates an `emulated_uint64_t` from two `uint32_t`s representing its bitpattern
    *
    * @param [in] hi Highest 32 bits of the `uint64` being emulated
    * @param [in] lo Lowest 32 bits of the `uint64` being emulated
    */
    NBL_CONSTEXPR_STATIC_FUNC this_t create(NBL_CONST_REF_ARG(uint32_t) hi, NBL_CONST_REF_ARG(uint32_t) lo)
    {
        return create(storage_t(hi, lo));
    }

    // ------------------------------------------------------- BITWISE OPERATORS -------------------------------------------------

    NBL_CONSTEXPR_INLINE_FUNC this_t operator&(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        this_t retVal = create(data & rhs.data);
        return retVal;
    }

    NBL_CONSTEXPR_INLINE_FUNC this_t operator|(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        this_t retVal = create(data | rhs.data);
        return retVal;
    }

    NBL_CONSTEXPR_INLINE_FUNC this_t operator^(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        this_t retVal = create(data ^ rhs.data);
        return retVal;
    }

    NBL_CONSTEXPR_INLINE_FUNC this_t operator~() NBL_CONST_MEMBER_FUNC
    {
        this_t retVal = create(~data);
        return retVal;
    }

    // Only valid in CPP
    #ifndef __HLSL_VERSION

    constexpr inline this_t operator<<(uint16_t bits) const;

    constexpr inline this_t operator>>(uint16_t bits) const;

    #endif

    // ------------------------------------------------------- ARITHMETIC OPERATORS -------------------------------------------------

    NBL_CONSTEXPR_INLINE_FUNC this_t operator+(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        const spirv::AddCarryOutput<uint32_t> lowerAddResult = addCarry(data.y, rhs.data.y);
        const storage_t addResult = { data.x + rhs.data.x + lowerAddResult.carry, lowerAddResult.result };
        const this_t retVal = create(addResult);
        return retVal;
    }

    NBL_CONSTEXPR_INLINE_FUNC this_t operator-(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        const spirv::SubBorrowOutput<uint32_t> lowerSubResult = subBorrow(data.y, rhs.data.y);
        const storage_t subResult = { data.x - rhs.data.x - lowerSubResult.borrow, lowerSubResult.result };
        const this_t retVal = create(subResult);
        return retVal;
    }

};

template<>
struct left_shift_operator<emulated_uint64_t> 
{
    using type_t = emulated_uint64_t;
    NBL_CONSTEXPR_STATIC uint32_t ComponentBitWidth = uint32_t(8 * sizeof(uint32_t));

    NBL_CONSTEXPR_INLINE_FUNC type_t operator()(NBL_CONST_REF_ARG(type_t) operand, uint16_t bits)
    {
        const uint32_t _bits = uint32_t(bits);
        const uint32_t shift = ComponentBitWidth - _bits;
        const uint32_t higherBitsMask = ~uint32_t(0) << shift;
        // We need the `x` component of the vector (which represents the higher bits of the emulated uint64) to get the `bits` higher bits of the `y` component
        const vector<uint32_t, 2> retValData = { (operand.data.x << _bits) | ((operand.data.y & higherBitsMask) >> shift), operand.data.y << _bits };
        return emulated_uint64_t::create(retValData);
    }
};

template<>
struct arithmetic_right_shift_operator<emulated_uint64_t>
{
    using type_t = emulated_uint64_t;
    NBL_CONSTEXPR_STATIC uint32_t ComponentBitWidth = uint32_t(8 * sizeof(uint32_t));

    NBL_CONSTEXPR_INLINE_FUNC type_t operator()(NBL_CONST_REF_ARG(type_t) operand, uint16_t bits)
    {
        const uint32_t _bits = uint32_t(bits);
        const uint32_t shift = ComponentBitWidth - _bits;
        const uint32_t lowerBitsMask = ~uint32_t(0) >> shift;
        // We need the `y` component of the vector (which represents the lower bits of the emulated uint64) to get the `bits` lower bits of the `x` component
        const vector<uint32_t, 2> retValData = { operand.data.x >> _bits, ((operand.data.x & lowerBitsMask) << shift) | (operand.data.y >> _bits) };
        return emulated_uint64_t::create(retValData);
    }
};

#ifndef __HLSL_VERSION

constexpr inline emulated_uint64_t emulated_uint64_t::operator<<(uint16_t bits) const
{
    left_shift_operator<emulated_uint64_t> leftShift;
    return leftShift(*this, bits);
}

constexpr inline emulated_uint64_t emulated_uint64_t::operator>>(uint16_t bits) const
{
    arithmetic_right_shift_operator<emulated_uint64_t> rightShift;
    return rightShift(*this, bits);
}

#endif

} //namespace nbl
} //namespace hlsl



#endif
