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

struct emulated_int64_t;

struct emulated_uint64_t
{
    using this_t = emulated_uint64_t;
    NBL_CONSTEXPR_STATIC_INLINE bool Signed = false;

    #include "int64_common_member_inc.hlsl" 

    #ifndef __HLSL_VERSION
    emulated_uint64_t() = default;
    // GLM requires these to cast vectors because it uses a native `static_cast`
    template<concepts::IntegralScalar I>
    constexpr explicit emulated_uint64_t(const I& toEmulate);

    constexpr explicit emulated_uint64_t(const emulated_int64_t& other);
    #endif
};


struct emulated_int64_t
{
    using this_t = emulated_int64_t;
    NBL_CONSTEXPR_STATIC_INLINE bool Signed = true;
    
    #include "int64_common_member_inc.hlsl"
    
    #ifndef __HLSL_VERSION
    emulated_int64_t() = default;
    // GLM requires these to cast vectors because it uses a native `static_cast`
    template<concepts::IntegralScalar I>
    constexpr explicit emulated_int64_t(const I& toEmulate);

    constexpr explicit emulated_int64_t(const emulated_uint64_t& other);
    #endif

    NBL_CONSTEXPR_FUNC emulated_int64_t operator-() NBL_CONST_MEMBER_FUNC;

};

// ------------------------------------------------ TYPE TRAITS SATISFIED -----------------------------------------------------

template<>
struct is_signed<emulated_int64_t> : bool_constant<true> {};

template<>
struct is_unsigned<emulated_uint64_t> : bool_constant<true> {};

// --------------------------------------------------- CONCEPTS SATISFIED -----------------------------------------------------
namespace concepts
{

template <typename T>
NBL_BOOL_CONCEPT ImitationIntegral64Scalar = same_as<T, emulated_uint64_t> || same_as<T, emulated_int64_t>;
  
namespace impl
{

template<>
struct is_emulating_integral_scalar<emulated_uint64_t>
{
    NBL_CONSTEXPR_STATIC_INLINE bool value = true;
};

template<>
struct is_emulating_integral_scalar<emulated_int64_t>
{
    NBL_CONSTEXPR_STATIC_INLINE bool value = true;
};
}


}


namespace impl
{

template<typename To, typename From> NBL_PARTIAL_REQ_TOP(concepts::ImitationIntegral64Scalar<To> && concepts::ImitationIntegral64Scalar<From> && !concepts::same_as<To, From>)
struct static_cast_helper<To, From NBL_PARTIAL_REQ_BOT(concepts::ImitationIntegral64Scalar<To> && concepts::ImitationIntegral64Scalar<From> && !concepts::same_as<To, From>) >
{

    NBL_CONSTEXPR_STATIC To cast(NBL_CONST_REF_ARG(From) other)
    {
        To retVal;
        retVal.data = other.data;
        return retVal;
    }
};

template<typename To, typename From> NBL_PARTIAL_REQ_TOP(concepts::IntegralScalar<To> && (sizeof(To) <= sizeof(uint32_t)) && concepts::ImitationIntegral64Scalar<From>)
struct static_cast_helper<To, From NBL_PARTIAL_REQ_BOT(concepts::IntegralScalar<To> && (sizeof(To) <= sizeof(uint32_t)) && concepts::ImitationIntegral64Scalar<From>) >
{
    // Return only the lowest bits
    NBL_CONSTEXPR_STATIC To cast(NBL_CONST_REF_ARG(From) val)
    {
        return _static_cast<To>(val.data.x);
    }
};

template<typename To, typename From> NBL_PARTIAL_REQ_TOP(concepts::IntegralScalar<To> && (sizeof(To) > sizeof(uint32_t)) && concepts::ImitationIntegral64Scalar<From>)
struct static_cast_helper<To, From NBL_PARTIAL_REQ_BOT(concepts::IntegralScalar<To> && (sizeof(To) > sizeof(uint32_t)) && concepts::ImitationIntegral64Scalar<From>) >
{
    NBL_CONSTEXPR_STATIC To cast(NBL_CONST_REF_ARG(From) val)
    {
        return bit_cast<To>(val.data);
    }
};

template<typename To, typename From> NBL_PARTIAL_REQ_TOP(concepts::IntegralScalar<From> && (sizeof(From) <= sizeof(uint32_t)) && concepts::ImitationIntegral64Scalar<To>)
struct static_cast_helper<To, From NBL_PARTIAL_REQ_BOT(concepts::IntegralScalar<From> && (sizeof(From) <= sizeof(uint32_t)) && concepts::ImitationIntegral64Scalar<To>) >
{
    // Set only lower bits
    NBL_CONSTEXPR_STATIC To cast(NBL_CONST_REF_ARG(From) i)
    {
        return To::create(_static_cast<uint32_t>(i), uint32_t(0));
    }
};

template<typename To, typename From> NBL_PARTIAL_REQ_TOP(concepts::IntegralScalar<From> && (sizeof(From) > sizeof(uint32_t)) && concepts::ImitationIntegral64Scalar<To>)
struct static_cast_helper<To, From NBL_PARTIAL_REQ_BOT(concepts::IntegralScalar<From> && (sizeof(From) > sizeof(uint32_t)) && concepts::ImitationIntegral64Scalar<To>) >
{
    NBL_CONSTEXPR_STATIC To cast(NBL_CONST_REF_ARG(From) i)
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

constexpr emulated_int64_t::emulated_int64_t(const emulated_uint64_t& other) : data(other.data) {}

constexpr emulated_uint64_t::emulated_uint64_t(const emulated_int64_t& other) : data(other.data) {}

template<concepts::IntegralScalar I>
constexpr emulated_int64_t::emulated_int64_t(const I& toEmulate)
{
    *this = _static_cast<emulated_int64_t>(toEmulate);
}

template<concepts::IntegralScalar I>
constexpr emulated_uint64_t::emulated_uint64_t(const I& toEmulate)
{
    *this = _static_cast<emulated_uint64_t>(toEmulate);
}

template<concepts::IntegralScalar I>
constexpr emulated_int64_t::operator I() const noexcept
{
    return _static_cast<I>(*this);
}

template<concepts::IntegralScalar I>
constexpr emulated_uint64_t::operator I() const noexcept
{
    return _static_cast<I>(*this);
}

#endif

// ---------------------- Functional operators ------------------------

template<typename T> NBL_PARTIAL_REQ_TOP(concepts::ImitationIntegral64Scalar<T>)
struct left_shift_operator<T NBL_PARTIAL_REQ_BOT(concepts::ImitationIntegral64Scalar<T>) >
{
    using type_t = T;
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

constexpr inline emulated_int64_t emulated_int64_t::operator<<(uint32_t bits) const
{
    left_shift_operator<emulated_int64_t> leftShift;
    return leftShift(*this, bits);
}

constexpr inline emulated_uint64_t emulated_uint64_t::operator<<(uint32_t bits) const
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

template<typename T> NBL_PARTIAL_REQ_TOP(concepts::ImitationIntegral64Scalar<T>)
struct plus<T NBL_PARTIAL_REQ_BOT(concepts::ImitationIntegral64Scalar<T>) >
{
    using type_t = T;

    type_t operator()(NBL_CONST_REF_ARG(type_t) lhs, NBL_CONST_REF_ARG(type_t) rhs)
    {
        return lhs + rhs;
    }

    const static type_t identity;
};

template<typename T> NBL_PARTIAL_REQ_TOP(concepts::ImitationIntegral64Scalar<T>)
struct minus<T NBL_PARTIAL_REQ_BOT(concepts::ImitationIntegral64Scalar<T>) >
{
    using type_t = T;

    type_t operator()(NBL_CONST_REF_ARG(type_t) lhs, NBL_CONST_REF_ARG(type_t) rhs)
    {
        return lhs - rhs;
    }

    const static type_t identity;
};

template<>
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR emulated_uint64_t plus<emulated_uint64_t>::identity = _static_cast<emulated_uint64_t>(uint64_t(0));
template<>
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR emulated_int64_t plus<emulated_int64_t>::identity = _static_cast<emulated_int64_t>(int64_t(0));
template<>
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR emulated_uint64_t minus<emulated_uint64_t>::identity = _static_cast<emulated_uint64_t>(uint64_t(0));
template<>
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR emulated_int64_t minus<emulated_int64_t>::identity = _static_cast<emulated_int64_t>(int64_t(0));

// --------------------------------- Compound assignment operators ------------------------------------------
// Specializations of the structs found in functional.hlsl

template<typename T> NBL_PARTIAL_REQ_TOP(concepts::ImitationIntegral64Scalar<T>)
struct plus_assign<T NBL_PARTIAL_REQ_BOT(concepts::ImitationIntegral64Scalar<T>) >
{
    using type_t = T;
    using base_t = plus<type_t>;
    base_t baseOp;
    void operator()(NBL_REF_ARG(type_t) lhs, NBL_CONST_REF_ARG(type_t) rhs)
    {
        lhs = baseOp(lhs, rhs);
    }

    const static type_t identity;
};

template<typename T> NBL_PARTIAL_REQ_TOP(concepts::ImitationIntegral64Scalar<T>)
struct minus_assign<T NBL_PARTIAL_REQ_BOT(concepts::ImitationIntegral64Scalar<T>) >
{
    using type_t = T;
    using base_t = minus<type_t>;
    base_t baseOp;
    void operator()(NBL_REF_ARG(type_t) lhs, NBL_CONST_REF_ARG(type_t) rhs)
    {
        lhs = baseOp(lhs, rhs);
    }

    const static type_t identity;
};

template<>
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR emulated_uint64_t plus_assign<emulated_uint64_t>::identity = plus<emulated_uint64_t>::identity;
template<>
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR emulated_int64_t plus_assign<emulated_int64_t>::identity = plus<emulated_int64_t>::identity;
template<>
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR emulated_uint64_t minus_assign<emulated_uint64_t>::identity = minus<emulated_uint64_t>::identity;
template<>
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR emulated_int64_t minus_assign<emulated_int64_t>::identity = minus<emulated_int64_t>::identity;

// --------------------------------- Unary operators ------------------------------------------
// Specializations of the structs found in functional.hlsl
template<>
struct unary_minus_operator<emulated_int64_t>
{
    using type_t = emulated_int64_t;

    NBL_CONSTEXPR_FUNC type_t operator()(NBL_CONST_REF_ARG(type_t) operand)
    {
        using storage_t = type_t::storage_t;
        storage_t inverted = ~operand.data;
        return type_t::create(_static_cast<storage_t>(inverted)) + _static_cast<type_t>(1);
    }
};

NBL_CONSTEXPR_INLINE_FUNC emulated_int64_t emulated_int64_t::operator-() NBL_CONST_MEMBER_FUNC
{
    unary_minus_operator<emulated_int64_t> unaryMinus;
    return unaryMinus(NBL_DEREF_THIS);
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
