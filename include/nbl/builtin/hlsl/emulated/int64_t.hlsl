#ifndef _NBL_BUILTIN_HLSL_EMULATED_INT64_T_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_EMULATED_INT64_T_HLSL_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/functional.hlsl"
#include "nbl/builtin/hlsl/concepts/core.hlsl"

// Didn't bother with operator*, operator/, implement if you need them. Multiplication is pretty straightforward, division requires switching on signs 
// and whether the topmost bits of the divisor are equal to 0
// - Francisco

namespace nbl 
{
namespace hlsl
{

template<bool Signed>
struct emulated_int64_base
{
    using storage_t = vector<uint32_t, 2>;
    using this_t = emulated_int64_base<Signed>;

    storage_t data;

    // ---------------------------------------------------- CONSTRUCTORS ---------------------------------------------------------------

    #ifndef __HLSL_VERSION

    emulated_int64_base() = default;

    #endif

    /**
    * @brief Creates an `emulated_int64` from a vector of two `uint32_t`s representing its bitpattern
    *
    * @param [in] _data Vector of `uint32_t` encoding the `uint64_t/int64_t` being emulated
    */
    NBL_CONSTEXPR_STATIC_FUNC this_t create(NBL_CONST_REF_ARG(storage_t) _data)
    {
        this_t retVal;
        retVal.data = _data;
        return retVal;
    }

    /**
    * @brief Creates an `emulated_int64` from two `uint32_t`s representing its bitpattern
    *
    * @param [in] hi Highest 32 bits of the `uint64_t/int64_t` being emulated
    * @param [in] lo Lowest 32 bits of the `uint64_t/int64_t` being emulated
    */
    NBL_CONSTEXPR_STATIC_FUNC this_t create(NBL_CONST_REF_ARG(uint32_t) hi, NBL_CONST_REF_ARG(uint32_t) lo)
    {
        return create(storage_t(hi, lo));
    }

    /**
    * @brief Creates an `emulated_int64_base` from a `uint64_t` with its bitpattern. Useful for compile-time encoding.
    *
    * @param [in] u `uint64_t` to be unpacked into high and low bits
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
        {
            // If signed, compare topmost bits as signed
            NBL_IF_CONSTEXPR(Signed)
                return _static_cast<int32_t>(data.x) < _static_cast<int32_t>(rhs.data.x);
            // If unsigned, compare them as-is
            else
                return data.x < rhs.data.x;
        }
        // Lower bits are positive in both signed and unsigned
        else
            return data.y < rhs.data.y;
    }

    NBL_CONSTEXPR_INLINE_FUNC bool operator>(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        if (data.x != rhs.data.x)
        {
            // If signed, compare topmost bits as signed
            NBL_IF_CONSTEXPR(Signed)
                return _static_cast<int32_t>(data.x) > _static_cast<int32_t>(rhs.data.x);
            // If unsigned, compare them as-is
            else
                return data.x > rhs.data.x;
        }
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

using emulated_uint64_t = emulated_int64_base<false>;
using emulated_int64_t = emulated_int64_base<true>;

// ---------------------- Functional operatos ------------------------

template<bool Signed>
struct left_shift_operator<emulated_int64_base<Signed> > 
{
    using type_t = emulated_int64_base<Signed>;
    NBL_CONSTEXPR_STATIC uint32_t ComponentBitWidth = uint32_t(8 * sizeof(uint32_t));

    NBL_CONSTEXPR_INLINE_FUNC type_t operator()(NBL_CONST_REF_ARG(type_t) operand, uint16_t bits)
    {
        if (!bits)
            return operand;
        const uint32_t _bits = uint32_t(bits);
        const uint32_t shift = ComponentBitWidth - _bits;
        // We need the `x` component of the vector (which represents the higher bits of the emulated uint64) to get the `bits` higher bits of the `y` component
        const vector<uint32_t, 2> retValData = { (operand.data.x << _bits) | (operand.data.y  >> shift), operand.data.y << _bits };
        return type_t::create(retValData);
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

template<bool Signed>
constexpr inline emulated_int64_base<Signed> emulated_int64_base<Signed>::operator<<(uint16_t bits) const
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

template<typename I, bool Signed> NBL_PARTIAL_REQ_TOP(concepts::IntegralScalar<I> && (sizeof(I) <= sizeof(uint32_t)) && (is_signed_v<I> == Signed))
struct static_cast_helper<I, emulated_int64_base<Signed> NBL_PARTIAL_REQ_BOT(concepts::IntegralScalar<I> && (sizeof(I) <= sizeof(uint32_t))) >
{
    using To = I;
    using From = emulated_int64_base<Signed>;

    // Return only the lowest bits
    NBL_CONSTEXPR_STATIC_INLINE_FUNC To cast(From val)
    {
        return _static_cast<To>(val.data.y);
    }
};

template<typename I, bool Signed> NBL_PARTIAL_REQ_TOP((is_same_v<I, uint64_t> || is_same_v<I, int64_t>) && (is_signed_v<I> == Signed))
struct static_cast_helper<I, emulated_int64_base<Signed> NBL_PARTIAL_REQ_BOT((is_same_v<I, uint64_t> || is_same_v<I, int64_t>) && (is_signed_v<I> == Signed)) >
{
    using To = I;
    using From = emulated_int64_base<Signed>;

    NBL_CONSTEXPR_STATIC_INLINE_FUNC To cast(From val)
    {
        const To highBits = _static_cast<To>(val.data.x) << To(32);
        return highBits | _static_cast<To>(val.data.y);
    }
};

template<typename I, bool Signed> NBL_PARTIAL_REQ_TOP(concepts::IntegralScalar<I> && (sizeof(I) <= sizeof(uint32_t)) && (is_signed_v<I> == Signed))
struct static_cast_helper<emulated_int64_base<Signed>, I NBL_PARTIAL_REQ_BOT(concepts::IntegralScalar<I> && (sizeof(I) <= sizeof(uint32_t)) && (is_signed_v<I> == Signed)) >
{
    using To = emulated_int64_base<Signed>;
    using From = I;

    // Set only lower bits
    NBL_CONSTEXPR_STATIC_INLINE_FUNC To cast(From i)
    {
        return To::create(uint32_t(0), _static_cast<uint32_t>(i));
    }
};

template<typename I, bool Signed> NBL_PARTIAL_REQ_TOP((is_same_v<I, uint64_t> || is_same_v<I, int64_t>) && (is_signed_v<I> == Signed))
struct static_cast_helper<emulated_int64_base<Signed>, I NBL_PARTIAL_REQ_BOT((is_same_v<I, uint64_t> || is_same_v<I, int64_t>) && (is_signed_v<I> == Signed)) >
{
    using To = emulated_int64_base<Signed>;
    using From = I;

    NBL_CONSTEXPR_STATIC_INLINE_FUNC To cast(From i)
    {
        return To::create(_static_cast<uint64_t>(i));
    }
};

} //namespace impl

// ---------------------- STD arithmetic operators ------------------------
// Specializations of the structs found in functional.hlsl
// These all have to be specialized because of the identity that can't be initialized inside the struct definition

template<bool Signed>
struct plus<emulated_int64_base<Signed> >
{
    using type_t = emulated_int64_base<Signed>;

    type_t operator()(NBL_CONST_REF_ARG(type_t) lhs, NBL_CONST_REF_ARG(type_t) rhs)
    {
        return lhs + rhs;
    }

    #ifndef __HLSL_VERSION
    NBL_CONSTEXPR_STATIC_INLINE type_t identity = _static_cast<emulated_uint64_t>(uint64_t(0));
    #else
    NBL_CONSTEXPR_STATIC_INLINE type_t identity;
    #endif
};

template<bool Signed>
struct minus<emulated_int64_base<Signed> >
{
    using type_t = emulated_int64_base<Signed>;

    type_t operator()(NBL_CONST_REF_ARG(type_t) lhs, NBL_CONST_REF_ARG(type_t) rhs)
    {
        return lhs - rhs;
    }

    #ifndef __HLSL_VERSION
    NBL_CONSTEXPR_STATIC_INLINE type_t identity = _static_cast<emulated_uint64_t>(uint64_t(0));
    #else
    NBL_CONSTEXPR_STATIC_INLINE type_t identity;
    #endif
};

#ifdef __HLSL_VERSION
template<>
NBL_CONSTEXPR emulated_uint64_t plus<emulated_uint64_t>::identity = _static_cast<emulated_uint64_t>(uint64_t(0));
template<>
NBL_CONSTEXPR emulated_int64_t plus<emulated_int64_t>::identity = _static_cast<emulated_int64_t>(int64_t(0));
template<>
NBL_CONSTEXPR emulated_uint64_t minus<emulated_uint64_t>::identity = _static_cast<emulated_uint64_t>(uint64_t(0));
template<>
NBL_CONSTEXPR emulated_int64_t minus<emulated_int64_t>::identity = _static_cast<emulated_int64_t>(int64_t(0));
#endif

// --------------------------------- Compound assignment operators ------------------------------------------
// Specializations of the structs found in functional.hlsl

template<bool Signed>
struct plus_assign<emulated_int64_base<Signed> >
{
    using type_t = emulated_int64_base<Signed>;
    using base_t = plus<type_t>;
    base_t baseOp;
    void operator()(NBL_REF_ARG(type_t) lhs, NBL_CONST_REF_ARG(type_t) rhs)
    {
        lhs = baseOp(lhs, rhs);
    }

    #ifndef __HLSL_VERSION
    NBL_CONSTEXPR_STATIC_INLINE type_t identity = base_t::identity;
    #else
    NBL_CONSTEXPR_STATIC_INLINE type_t identity;
    #endif
};

template<bool Signed>
struct minus_assign<emulated_int64_base<Signed> >
{
    using type_t = emulated_int64_base<Signed>;
    using base_t = minus<type_t>;
    base_t baseOp;
    void operator()(NBL_REF_ARG(type_t) lhs, NBL_CONST_REF_ARG(type_t) rhs)
    {
        lhs = baseOp(lhs, rhs);
    }

    #ifndef __HLSL_VERSION
    NBL_CONSTEXPR_STATIC_INLINE type_t identity = base_t::identity;
    #else
    NBL_CONSTEXPR_STATIC_INLINE type_t identity;
    #endif
};

#ifdef __HLSL_VERSION
template<>
NBL_CONSTEXPR emulated_uint64_t plus_assign<emulated_uint64_t>::identity = plus<emulated_uint64_t>::identity;
template<>
NBL_CONSTEXPR emulated_int64_t plus_assign<emulated_int64_t>::identity = plus<emulated_int64_t>::identity;
template<>
NBL_CONSTEXPR emulated_uint64_t minus_assign<emulated_uint64_t>::identity = minus<emulated_uint64_t>::identity;
template<>
NBL_CONSTEXPR emulated_int64_t minus_assign<emulated_int64_t>::identity = minus<emulated_int64_t>::identity;
#endif

} //namespace nbl
} //namespace hlsl

// Declare them as signed/unsigned versions of each other

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
