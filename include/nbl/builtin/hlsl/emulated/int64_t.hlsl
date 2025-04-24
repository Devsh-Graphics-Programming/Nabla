#ifndef _NBL_BUILTIN_HLSL_EMULATED_INT64_T_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_EMULATED_INT64_T_HLSL_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/functional.hlsl"
#include "nbl/builtin/hlsl/concepts/core.hlsl"
#include "nbl/builtin/hlsl/bit.hlsl"

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
    using this_signed_t = emulated_int64_base<true>;

    storage_t data;

    // ---------------------------------------------------- CONSTRUCTORS ---------------------------------------------------------------

    #ifndef __HLSL_VERSION

    emulated_int64_base() = default;

    // GLM requires these to cast vectors because it uses a native `static_cast`
    template<concepts::IntegralScalar I>
    constexpr explicit emulated_int64_base(const I& toEmulate);

    constexpr explicit emulated_int64_base(const emulated_int64_base<!Signed>& other) : data(other.data) {}

    #endif

    /**
    * @brief Creates an `emulated_int64` from a vector of two `uint32_t`s representing its bitpattern
    *
    * @param [in] _data Vector of `uint32_t` encoding the `uint64_t/int64_t` being emulated. Stored as little endian (first component are the lower 32 bits)
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
    * @param [in] lo Lowest 32 bits of the `uint64_t/int64_t` being emulated
    * @param [in] hi Highest 32 bits of the `uint64_t/int64_t` being emulated
    */
    NBL_CONSTEXPR_STATIC_FUNC this_t create(NBL_CONST_REF_ARG(uint32_t) lo, NBL_CONST_REF_ARG(uint32_t) hi)
    {
        return create(storage_t(lo, hi));
    }

    // ------------------------------------------------------- CONVERSION OPERATORS---------------------------------------------------------------
    // GLM requires these for vector casts

    #ifndef __HLSL_VERSION

    template<concepts::IntegralScalar I>
    constexpr explicit operator I() const noexcept;

    #endif

    // ------------------------------------------------------- INTERNAL GETTERS -------------------------------------------------

    NBL_CONSTEXPR_FUNC uint32_t __getLSB() NBL_CONST_MEMBER_FUNC
    {
        return data.x;
    }

    NBL_CONSTEXPR_FUNC uint32_t __getMSB() NBL_CONST_MEMBER_FUNC
    {
        return data.y;
    }

    // ------------------------------------------------------- BITWISE OPERATORS -------------------------------------------------

    NBL_CONSTEXPR_FUNC this_t operator&(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        this_t retVal = create(data & rhs.data);
        return retVal;
    }

    NBL_CONSTEXPR_FUNC this_t operator|(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        this_t retVal = create(data | rhs.data);
        return retVal;
    }

    NBL_CONSTEXPR_FUNC this_t operator^(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        this_t retVal = create(data ^ rhs.data);
        return retVal;
    }

    NBL_CONSTEXPR_FUNC this_t operator~() NBL_CONST_MEMBER_FUNC
    {
        this_t retVal = create(~data);
        return retVal;
    }

    // Only valid in CPP
    #ifndef __HLSL_VERSION
    constexpr inline this_t operator<<(uint32_t bits) const;
    constexpr inline this_t operator>>(uint32_t bits) const;

    #endif

    // ------------------------------------------------------- ARITHMETIC OPERATORS -------------------------------------------------

    NBL_CONSTEXPR_FUNC this_signed_t operator-() NBL_CONST_MEMBER_FUNC
    {
        vector<int32_t, 2> negated = -data;
        return this_signed_t::create(_static_cast<storage_t>(negated));
    }

    NBL_CONSTEXPR_FUNC this_t operator+(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        const spirv::AddCarryOutput<uint32_t> lowerAddResult = addCarry(__getLSB(), rhs.__getLSB());
        return create(lowerAddResult.result, __getMSB() + rhs.__getMSB() + lowerAddResult.carry);
    }

    NBL_CONSTEXPR_FUNC this_t operator-(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        const spirv::SubBorrowOutput<uint32_t> lowerSubResult = subBorrow(__getLSB(), rhs.__getLSB());
        return create(lowerSubResult.result, __getMSB() - rhs.__getMSB() - lowerSubResult.borrow);
    }

    // ------------------------------------------------------- COMPARISON OPERATORS -------------------------------------------------
    NBL_CONSTEXPR_FUNC bool operator==(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        equal_to<storage_t> equals;
        return all(equals(data, rhs.data));
    }

    NBL_CONSTEXPR_FUNC bool operator!=(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        not_equal_to<storage_t> notEquals;
        return any(notEquals(data, rhs.data));
    }

    NBL_CONSTEXPR_FUNC bool operator<(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        // Either the topmost bits, when interpreted with correct sign, are less than those of `rhs`, or they're equal and the lower bits are less
        // (lower bits are always positive in both unsigned and 2's complement so comparison can happen as-is)
        const bool MSBEqual = __getMSB() == rhs.__getMSB();
        const bool MSB = Signed ? (bit_cast<int32_t>(__getMSB()) < bit_cast<int32_t>(rhs.__getMSB())) : (__getMSB() < rhs.__getMSB());
        const bool LSB = __getLSB() < rhs.__getLSB();
        return MSBEqual ? LSB : MSB;
    }

    NBL_CONSTEXPR_FUNC bool operator>(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        // Same reasoning as above
        const bool MSBEqual = __getMSB() == rhs.__getMSB();
        const bool MSB = Signed ? (bit_cast<int32_t>(__getMSB()) > bit_cast<int32_t>(rhs.__getMSB())) : (__getMSB() > rhs.__getMSB());
        const bool LSB = __getLSB() > rhs.__getLSB();
        return MSBEqual ? LSB : MSB;
    }

    NBL_CONSTEXPR_FUNC bool operator<=(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        return !operator>(rhs);
    }

    NBL_CONSTEXPR_FUNC bool operator>=(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        return !operator<(rhs);
    }
};

using emulated_uint64_t = emulated_int64_base<false>;
using emulated_int64_t = emulated_int64_base<true>;

namespace impl
{

template<bool Signed>
struct static_cast_helper<emulated_int64_base<Signed>, emulated_int64_base<!Signed> >
{
    using To = emulated_int64_base<Signed>;
    using From = emulated_int64_base<!Signed>;

    NBL_CONSTEXPR_STATIC_FUNC To cast(NBL_CONST_REF_ARG(From) other)
    {
        To retVal;
        retVal.data = other.data;
        return retVal;
    }
};

template<typename I, bool Signed> NBL_PARTIAL_REQ_TOP(concepts::IntegralScalar<I> && (sizeof(I) <= sizeof(uint32_t)))
struct static_cast_helper<I, emulated_int64_base<Signed> NBL_PARTIAL_REQ_BOT(concepts::IntegralScalar<I> && (sizeof(I) <= sizeof(uint32_t))) >
{
    using To = I;
    using From = emulated_int64_base<Signed>;

    // Return only the lowest bits
    NBL_CONSTEXPR_STATIC_FUNC To cast(NBL_CONST_REF_ARG(From) val)
    {
        return _static_cast<To>(val.data.x);
    }
};

template<typename I, bool Signed> NBL_PARTIAL_REQ_TOP(concepts::IntegralScalar<I> && (sizeof(I) > sizeof(uint32_t)))
struct static_cast_helper<I, emulated_int64_base<Signed> NBL_PARTIAL_REQ_BOT(concepts::IntegralScalar<I> && (sizeof(I) > sizeof(uint32_t))) >
{
    using To = I;
    using From = emulated_int64_base<Signed>;

    NBL_CONSTEXPR_STATIC_FUNC To cast(NBL_CONST_REF_ARG(From) val)
    {
        return bit_cast<To>(val.data);
    }
};

template<typename I, bool Signed> NBL_PARTIAL_REQ_TOP(concepts::IntegralScalar<I> && (sizeof(I) <= sizeof(uint32_t)))
struct static_cast_helper<emulated_int64_base<Signed>, I NBL_PARTIAL_REQ_BOT(concepts::IntegralScalar<I> && (sizeof(I) <= sizeof(uint32_t))) >
{
    using To = emulated_int64_base<Signed>;
    using From = I;

    // Set only lower bits
    NBL_CONSTEXPR_STATIC_FUNC To cast(NBL_CONST_REF_ARG(From) i)
    {
        return To::create(_static_cast<uint32_t>(i), uint32_t(0));
    }
};

template<typename I, bool Signed> NBL_PARTIAL_REQ_TOP(concepts::IntegralScalar<I> && (sizeof(I) > sizeof(uint32_t)))
struct static_cast_helper<emulated_int64_base<Signed>, I NBL_PARTIAL_REQ_BOT(concepts::IntegralScalar<I> && (sizeof(I) > sizeof(uint32_t))) >
{
    using To = emulated_int64_base<Signed>;
    using From = I;

    NBL_CONSTEXPR_STATIC_FUNC To cast(NBL_CONST_REF_ARG(From) i)
    {
        // `bit_cast` blocked by GLM vectors using a union
        #ifndef __HLSL_VERSION
        return To::create(_static_cast<uint32_t>(i), _static_cast<uint32_t>(i >> 32));
        #else
        To retVal;
        retVal.data = bit_cast<vector<uint32_t, 2> >(i);
        return retVal;
        #endif 
    }
};

} //namespace impl

// Define constructor and conversion operators

#ifndef __HLSL_VERSION

template<bool Signed>
template<concepts::IntegralScalar I>
constexpr emulated_int64_base<Signed>::emulated_int64_base(const I& toEmulate)
{
    *this = _static_cast<emulated_int64_base<Signed>>(toEmulate);
}

template<bool Signed>
template<concepts::IntegralScalar I>
constexpr emulated_int64_base<Signed>::operator I() const noexcept
{
    return _static_cast<I>(*this);
}

#endif

// ---------------------- Functional operators ------------------------

template<bool Signed>
struct left_shift_operator<emulated_int64_base<Signed> >
{
    using type_t = emulated_int64_base<Signed>;
    NBL_CONSTEXPR_STATIC uint32_t ComponentBitWidth = uint32_t(8 * sizeof(uint32_t));

    // Can't do generic templated definition, see:
    //https://github.com/microsoft/DirectXShaderCompiler/issues/7325
    
    // If `_bits > 63` or `_bits < 0` the result is undefined
    NBL_CONSTEXPR_FUNC type_t operator()(NBL_CONST_REF_ARG(type_t) operand, uint32_t bits)
    {
        const bool bigShift = bits >= ComponentBitWidth; // Shift that completely rewrites LSB
        const uint32_t shift = bigShift ? bits - ComponentBitWidth : ComponentBitWidth - bits;
        const type_t shifted = type_t::create(bigShift ? vector<uint32_t, 2>(0, operand.__getLSB() << shift)
                                                       : vector<uint32_t, 2>(operand.__getLSB() << bits, (operand.__getMSB() << bits) | (operand.__getLSB() >> shift)));
        ternary_operator<type_t> ternary;
        return ternary(bool(bits), shifted, operand);
    }

    // If `_bits > 63` or `_bits < 0` the result is undefined
    NBL_CONSTEXPR_FUNC type_t operator()(NBL_CONST_REF_ARG(type_t) operand, type_t bits)
    {
        return operator()(operand, _static_cast<uint32_t>(bits));
    }
};

template<>
struct arithmetic_right_shift_operator<emulated_uint64_t>
{
    using type_t = emulated_uint64_t;
    NBL_CONSTEXPR_STATIC uint32_t ComponentBitWidth = uint32_t(8 * sizeof(uint32_t));

    // Can't do generic templated definition, see:
    //https://github.com/microsoft/DirectXShaderCompiler/issues/7325

    // If `_bits > 63` the result is undefined
    NBL_CONSTEXPR_FUNC type_t operator()(NBL_CONST_REF_ARG(type_t) operand, uint32_t bits)
    {
        const bool bigShift = bits >= ComponentBitWidth; // Shift that completely rewrites MSB
        const uint32_t shift = bigShift ? bits - ComponentBitWidth : ComponentBitWidth - bits;
        const type_t shifted = type_t::create(bigShift ? vector<uint32_t, 2>(operand.__getMSB() >> shift, 0)
                                                       : vector<uint32_t, 2>((operand.__getMSB() << shift) | (operand.__getLSB() >> bits), operand.__getMSB() >> bits));
        ternary_operator<type_t> ternary;
        return ternary(bool(bits), shifted, operand);
    }

    // If `_bits > 63` the result is undefined
    NBL_CONSTEXPR_FUNC type_t operator()(NBL_CONST_REF_ARG(type_t) operand, type_t bits)
    {
        return operator()(operand, _static_cast<uint32_t>(bits));
    }
};

template<>
struct arithmetic_right_shift_operator<emulated_int64_t>
{
    using type_t = emulated_int64_t;
    NBL_CONSTEXPR_STATIC uint32_t ComponentBitWidth = uint32_t(8 * sizeof(uint32_t));

    // Can't do generic templated definition, see:
    //https://github.com/microsoft/DirectXShaderCompiler/issues/7325

    // If `_bits > 63` or `_bits < 0` the result is undefined
    NBL_CONSTEXPR_FUNC type_t operator()(NBL_CONST_REF_ARG(type_t) operand, uint32_t bits)
    {
        const bool bigShift = bits >= ComponentBitWidth; // Shift that completely rewrites MSB
        const uint32_t shift = bigShift ? bits - ComponentBitWidth : ComponentBitWidth - bits;
        const type_t shifted = type_t::create(bigShift ? vector<uint32_t, 2>(uint32_t(int32_t(operand.__getMSB()) >> shift), int32_t(operand.__getMSB()) < 0 ? ~uint32_t(0) : uint32_t(0))
                                                                        : vector<uint32_t, 2>((operand.__getMSB() << shift) | (operand.__getLSB() >> bits), uint32_t(int32_t(operand.__getMSB()) >> bits)));
        ternary_operator<type_t> ternary;
        return ternary(bool(bits), shifted, operand);
    }

    // If `_bits > 63` or `_bits < 0` the result is undefined
    NBL_CONSTEXPR_FUNC type_t operator()(NBL_CONST_REF_ARG(type_t) operand, type_t bits)
    {
        return operator()(operand, _static_cast<uint32_t>(bits));
    }
};

#ifndef __HLSL_VERSION

template<bool Signed>
constexpr inline emulated_int64_base<Signed> emulated_int64_base<Signed>::operator<<(uint32_t bits) const
{
    left_shift_operator<emulated_uint64_t> leftShift;
    return leftShift(*this, bits);
}

constexpr inline emulated_uint64_t emulated_uint64_t::operator>>(uint32_t bits) const
{
    arithmetic_right_shift_operator<emulated_uint64_t> rightShift;
    return rightShift(*this, bits);
}

constexpr inline emulated_int64_t emulated_int64_t::operator>>(uint32_t bits) const
{
    arithmetic_right_shift_operator<emulated_int64_t> rightShift;
    return rightShift(*this, bits);
}

#endif

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

    const static type_t identity;
};

template<bool Signed>
struct minus<emulated_int64_base<Signed> >
{
    using type_t = emulated_int64_base<Signed>;

    type_t operator()(NBL_CONST_REF_ARG(type_t) lhs, NBL_CONST_REF_ARG(type_t) rhs)
    {
        return lhs - rhs;
    }

    const static type_t identity;
};

template<>
NBL_CONSTEXPR_INLINE emulated_uint64_t plus<emulated_uint64_t>::identity = _static_cast<emulated_uint64_t>(uint64_t(0));
template<>
NBL_CONSTEXPR_INLINE emulated_int64_t plus<emulated_int64_t>::identity = _static_cast<emulated_int64_t>(int64_t(0));
template<>
NBL_CONSTEXPR_INLINE emulated_uint64_t minus<emulated_uint64_t>::identity = _static_cast<emulated_uint64_t>(uint64_t(0));
template<>
NBL_CONSTEXPR_INLINE emulated_int64_t minus<emulated_int64_t>::identity = _static_cast<emulated_int64_t>(int64_t(0));

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

    const static type_t identity;
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

    const static type_t identity;
};

template<>
NBL_CONSTEXPR_INLINE emulated_uint64_t plus_assign<emulated_uint64_t>::identity = plus<emulated_uint64_t>::identity;
template<>
NBL_CONSTEXPR_INLINE emulated_int64_t plus_assign<emulated_int64_t>::identity = plus<emulated_int64_t>::identity;
template<>
NBL_CONSTEXPR_INLINE emulated_uint64_t minus_assign<emulated_uint64_t>::identity = minus<emulated_uint64_t>::identity;
template<>
NBL_CONSTEXPR_INLINE emulated_int64_t minus_assign<emulated_int64_t>::identity = minus<emulated_int64_t>::identity;

// ------------------------------------------------ TYPE TRAITS SATISFIED -----------------------------------------------------

template<>
struct is_signed<emulated_int64_t> : bool_constant<true> {};

template<>
struct is_unsigned<emulated_uint64_t> : bool_constant<true> {};

// --------------------------------------------------- CONCEPTS SATISFIED -----------------------------------------------------
namespace concepts
{
namespace impl
{
template<bool Signed>
struct is_emulating_integral_scalar<emulated_int64_base<Signed> >
{
    NBL_CONSTEXPR_STATIC_INLINE bool value = true;
};
}
}

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
