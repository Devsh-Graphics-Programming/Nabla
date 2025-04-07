#ifndef _NBL_BUILTIN_HLSL_EMULATED_INT64_T_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_EMULATED_INT64_T_HLSL_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/functional.hlsl"
#include "nbl/builtin/hlsl/concepts/core.hlsl"

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

    /**
    * @brief Creates an `emulated_uint64_t` from a `uint64_t`. Useful for compile-time encoding.
    *
    * @param [in] _data `uint64_t` to be unpacked into high and low bits
    */
    NBL_CONSTEXPR_STATIC_FUNC this_t create(NBL_CONST_REF_ARG(uint64_t) u)
    {
        return create(_static_cast<uint32_t>(u >> 32), _static_cast<uint32_t>(u));
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

    // ------------------------------------------------------- COMPARISON OPERATORS -------------------------------------------------
    NBL_CONSTEXPR_INLINE_FUNC bool operator==(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        return data.x == rhs.data.x && data.y == rhs.data.y;
    }

    NBL_CONSTEXPR_INLINE_FUNC bool operator!=(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        return data.x != rhs.data.x || data.y != rhs.data.y;
    }

    NBL_CONSTEXPR_INLINE_FUNC bool operator<(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        if (data.x != rhs.data.x)
            return data.x < rhs.data.x;
        else
            return data.y < rhs.data.y;
    }

    NBL_CONSTEXPR_INLINE_FUNC bool operator>(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        if (data.x != rhs.data.x)
            return data.x > rhs.data.x;
        else
            return data.y > rhs.data.y;
    }

    NBL_CONSTEXPR_INLINE_FUNC bool operator<=(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        return !operator>(rhs);
    }

    NBL_CONSTEXPR_INLINE_FUNC bool operator>=(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        return !operator<(rhs);
    }
};

struct emulated_int64_t : emulated_uint64_t
{
    using base_t = emulated_uint64_t;
    using base_t::storage_t;
    using this_t = emulated_int64_t;

    // ---------------------------------------------------- CONSTRUCTORS ---------------------------------------------------------------


    #ifndef __HLSL_VERSION

    emulated_int64_t() = default;

    #endif

    /**
    * @brief Creates an `emulated_int64_t` from a vector of two `uint32_t`s representing its bitpattern
    *
    * @param [in] _data Vector of `uint32_t` encoding the `int64_t` being emulated
    */
    NBL_CONSTEXPR_STATIC_FUNC this_t create(NBL_CONST_REF_ARG(storage_t) _data)
    {
        return _static_cast<this_t>(base_t::create(_data));
    }

    /**
    * @brief Creates an `emulated_int64_t` from two `uint32_t`s representing its bitpattern
    *
    * @param [in] hi Highest 32 bits of the `int64` being emulated
    * @param [in] lo Lowest 32 bits of the `int64` being emulated
    */
    NBL_CONSTEXPR_STATIC_FUNC this_t create(NBL_CONST_REF_ARG(uint32_t) hi, NBL_CONST_REF_ARG(uint32_t) lo)
    {
        return  _static_cast<this_t>(base_t::create(hi, lo));
    }

    /**
    * @brief Creates an `emulated_int64_t` from a `int64_t`. Useful for compile-time encoding.
    *
    * @param [in] _data `int64_t` to be unpacked into high and low bits
    */
    NBL_CONSTEXPR_STATIC_FUNC this_t create(NBL_CONST_REF_ARG(int64_t) i)
    {
        return _static_cast<this_t>(base_t::create(_static_cast<uint64_t>(i)));
    }

    // Only valid in CPP
    #ifndef __HLSL_VERSION

    // Only this one needs to be redefined since it's arithmetic
    constexpr inline this_t operator>>(uint16_t bits) const;

    #endif

    // ------------------------------------------------------- COMPARISON OPERATORS -------------------------------------------------

    // Same as unsigned but the topmost bits are compared as signed
    NBL_CONSTEXPR_INLINE_FUNC bool operator<(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        if (data.x != rhs.data.x)
            return _static_cast<int32_t>(data.x) < _static_cast<int32_t>(rhs.data.x);
        else
            return data.y < rhs.data.y;
    }

    NBL_CONSTEXPR_INLINE_FUNC bool operator>(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        if (data.x != rhs.data.x)
            return _static_cast<int32_t>(data.x) > _static_cast<int32_t>(rhs.data.x);
        else
            return data.y > rhs.data.y;
    }

    NBL_CONSTEXPR_INLINE_FUNC bool operator<=(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        return !operator>(rhs);
    }

    NBL_CONSTEXPR_INLINE_FUNC bool operator>=(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        return !operator<(rhs);
    }
};

template<>
struct left_shift_operator<emulated_uint64_t> 
{
    using type_t = emulated_uint64_t;
    NBL_CONSTEXPR_STATIC uint32_t ComponentBitWidth = uint32_t(8 * sizeof(uint32_t));

    NBL_CONSTEXPR_INLINE_FUNC type_t operator()(NBL_CONST_REF_ARG(type_t) operand, uint16_t bits)
    {
        if (!bits)
            return operand;
        const uint32_t _bits = uint32_t(bits);
        const uint32_t shift = ComponentBitWidth - _bits;
        // We need the `x` component of the vector (which represents the higher bits of the emulated uint64) to get the `bits` higher bits of the `y` component
        const vector<uint32_t, 2> retValData = { (operand.data.x << _bits) | (operand.data.y  >> shift), operand.data.y << _bits };
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
        if (!bits)
            return operand;
        const uint32_t _bits = uint32_t(bits);
        const uint32_t shift = ComponentBitWidth - _bits;
        // We need the `y` component of the vector (which represents the lower bits of the emulated uint64) to get the `bits` lower bits of the `x` component
        const vector<uint32_t, 2> retValData = { operand.data.x >> _bits, (operand.data.x << shift) | (operand.data.y >> _bits) };
        return emulated_uint64_t::create(retValData);
    }
};

template<>
struct left_shift_operator<emulated_int64_t>
{
    using type_t = emulated_int64_t;

    NBL_CONSTEXPR_INLINE_FUNC type_t operator()(NBL_CONST_REF_ARG(type_t) operand, uint16_t bits)
    {
        left_shift_operator<emulated_uint64_t> leftShift;
        return _static_cast<type_t>(leftShift(_static_cast<emulated_uint64_t>(operand), bits));
    }
};

template<>
struct arithmetic_right_shift_operator<emulated_int64_t>
{
    using type_t = emulated_int64_t;
    NBL_CONSTEXPR_STATIC uint32_t ComponentBitWidth = uint32_t(8 * sizeof(uint32_t));

    NBL_CONSTEXPR_INLINE_FUNC type_t operator()(NBL_CONST_REF_ARG(type_t) operand, uint16_t bits)
    {
        if (!bits)
            return operand;
        const uint32_t _bits = uint32_t(bits);
        const uint32_t shift = ComponentBitWidth - _bits;
        // We need the `y` component of the vector (which represents the lower bits of the emulated uint64) to get the `bits` lower bits of the `x` component
        // Also the right shift *only* in the top bits happens as a signed arithmetic right shift
        const vector<uint32_t, 2> retValData = { _static_cast<uint32_t>(_static_cast<int32_t>(operand.data.x)) >> _bits, (operand.data.x << shift) | (operand.data.y >> _bits) };
        return emulated_int64_t::create(retValData);
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

constexpr inline emulated_int64_t emulated_int64_t::operator>>(uint16_t bits) const
{
    arithmetic_right_shift_operator<emulated_int64_t> rightShift;
    return rightShift(*this, bits);
}

#endif

namespace impl
{

template<>
struct static_cast_helper<emulated_uint64_t, emulated_int64_t>
{
    using To = emulated_uint64_t;
    using From = emulated_int64_t;

    // Return only the lowest bits
    NBL_CONSTEXPR_STATIC_INLINE_FUNC To cast(From i)
    {
        To retVal;
        retVal.data = i.data;
        return retVal;
    }
};

template<>
struct static_cast_helper<emulated_int64_t, emulated_uint64_t>
{
    using To = emulated_int64_t;
    using From = emulated_uint64_t;

    // Return only the lowest bits
    NBL_CONSTEXPR_STATIC_INLINE_FUNC To cast(From u)
    {
        To retVal;
        retVal.data = u.data;
        return retVal;
    }
};

template<typename Unsigned> NBL_PARTIAL_REQ_TOP(concepts::UnsignedIntegralScalar<Unsigned> && (sizeof(Unsigned) <= sizeof(uint32_t)))
struct static_cast_helper<Unsigned, emulated_uint64_t NBL_PARTIAL_REQ_BOT(concepts::UnsignedIntegralScalar<Unsigned> && (sizeof(Unsigned) <= sizeof(uint32_t))) >
{
    using To = Unsigned;
    using From = emulated_uint64_t;

    // Return only the lowest bits
    NBL_CONSTEXPR_STATIC_INLINE_FUNC To cast(From u)
    {
        return _static_cast<To>(u.data.y);
    }
};

template<>
struct static_cast_helper<uint64_t, emulated_uint64_t>
{
    using To = uint64_t;
    using From = emulated_uint64_t;

    NBL_CONSTEXPR_STATIC_INLINE_FUNC To cast(From u)
    {
        const To highBits = _static_cast<To>(u.data.x) << To(32);
        return highBits | _static_cast<To>(u.data.y);
    }
};

template<typename Unsigned> NBL_PARTIAL_REQ_TOP(concepts::UnsignedIntegralScalar<Unsigned> && (sizeof(Unsigned) <= sizeof(uint32_t)))
struct static_cast_helper<emulated_uint64_t, Unsigned NBL_PARTIAL_REQ_BOT(concepts::UnsignedIntegralScalar<Unsigned> && (sizeof(Unsigned) <= sizeof(uint32_t))) >
{
    using To = emulated_uint64_t;
    using From = Unsigned;

    // Set only lower bits
    NBL_CONSTEXPR_STATIC_INLINE_FUNC To cast(From u)
    {
        return emulated_uint64_t::create(uint32_t(0), _static_cast<uint32_t>(u));
    }
};

template<>
struct static_cast_helper<emulated_uint64_t, uint64_t>
{
    using To = emulated_uint64_t;
    using From = uint64_t;

    NBL_CONSTEXPR_STATIC_INLINE_FUNC To cast(From u)
    {
        return emulated_uint64_t::create(u);
    }
};

template<typename Signed> NBL_PARTIAL_REQ_TOP(concepts::SignedIntegralScalar<Signed> && (sizeof(Signed) <= sizeof(uint32_t)))
struct static_cast_helper<Signed, emulated_uint64_t NBL_PARTIAL_REQ_BOT(concepts::SignedIntegralScalar<Signed> && (sizeof(Signed) <= sizeof(uint32_t))) >
{
    using To = Signed;
    using From = emulated_int64_t;

    // Return only the lowest bits
    NBL_CONSTEXPR_STATIC_INLINE_FUNC To cast(From i)
    {
        return _static_cast<To>(i.data.y);
    }
};

template<>
struct static_cast_helper<int64_t, emulated_int64_t>
{
    using To = int64_t;
    using From = emulated_int64_t;

    NBL_CONSTEXPR_STATIC_INLINE_FUNC To cast(From i)
    {
        const To highBits = _static_cast<To>(i.data.x) << To(32);
        return highBits | _static_cast<To>(i.data.y);
    }
};

template<typename Signed> NBL_PARTIAL_REQ_TOP(concepts::SignedIntegralScalar<Signed> && (sizeof(Signed) <= sizeof(uint32_t)))
struct static_cast_helper<emulated_uint64_t, Signed NBL_PARTIAL_REQ_BOT(concepts::SignedIntegralScalar<Signed> && (sizeof(Signed) <= sizeof(uint32_t))) >
{
    using To = emulated_int64_t;
    using From = Signed;

    // Set only lower bits
    NBL_CONSTEXPR_STATIC_INLINE_FUNC To cast(From i)
    {
        return emulated_int64_t::create(uint32_t(0), _static_cast<uint32_t>(i));
    }
};

template<>
struct static_cast_helper<emulated_int64_t, int64_t>
{
    using To = emulated_int64_t;
    using From = int64_t;

    NBL_CONSTEXPR_STATIC_INLINE_FUNC To cast(From i)
    {
        return emulated_int64_t::create(i);
    }
};

} //namespace impl

} //namespace nbl
} //namespace hlsl

#ifndef __HLSL_VERSION
#define NBL_ADD_STD std::
#else 
#define NBL_ADD_STD nbl::hlsl:: 
#endif

template<>
struct NBL_ADD_STD make_unsigned<nbl::hlsl::emulated_uint64_t> : type_identity<nbl::hlsl::emulated_uint64_t> {};

template<>
struct NBL_ADD_STD make_unsigned<nbl::hlsl::emulated_int64_t> : type_identity<nbl::hlsl::emulated_uint64_t> {};

template<>
struct NBL_ADD_STD make_signed<nbl::hlsl::emulated_uint64_t> : type_identity<nbl::hlsl::emulated_int64_t> {};

template<>
struct NBL_ADD_STD make_signed<nbl::hlsl::emulated_int64_t> : type_identity<nbl::hlsl::emulated_int64_t> {};

#undef NBL_ADD_STD



#endif
