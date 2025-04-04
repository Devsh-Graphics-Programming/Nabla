#ifndef _NBL_BUILTIN_HLSL_MORTON_INCLUDED_
#define _NBL_BUILTIN_HLSL_MORTON_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/concepts/core.hlsl"
#include "nbl/builtin/hlsl/bit.hlsl"
#include "nbl/builtin/hlsl/functional.hlsl"
#include "nbl/builtin/hlsl/emulated/uint64_t.hlsl"
#include "nbl/builtin/hlsl/mpl.hlsl"

namespace nbl
{
namespace hlsl
{
namespace morton
{

namespace impl
{

// Valid dimension for a morton code
template <uint16_t D>
NBL_BOOL_CONCEPT MortonDimension = 1 < D && D < 5;

// Basic decode masks

template<typename T, uint16_t Dim, uint16_t Bits = 8 * sizeof(T) / Dim>
struct decode_mask;

template<typename T, uint16_t Dim>
struct decode_mask<T, Dim, 1> : integral_constant<T, 1> {};

template<typename T, uint16_t Dim, uint16_t Bits>
struct decode_mask : integral_constant<T, (decode_mask<T, Dim, Bits - 1>::value << Dim) | T(1)> {};

template<typename T, uint16_t Dim, uint16_t Bits = 8 * sizeof(T) / Dim>
NBL_CONSTEXPR T decode_mask_v = decode_mask<T, Dim, Bits>::value;

// --------------------------------------------------------- MORTON ENCODE/DECODE MASKS ---------------------------------------------------
// Proper encode masks (either generic `T array[masksPerDImension]` or `morton_mask<T, Dim, MaskNumber>`) impossible to have until at best HLSL202y

#define NBL_MORTON_GENERIC_DECODE_MASK(DIM, MASK, HEX_VALUE) template<typename T> struct morton_mask_##DIM##_##MASK \
{\
    NBL_CONSTEXPR_STATIC_INLINE T value = _static_cast<T>(HEX_VALUE);\
};

#ifndef __HLSL_VERSION

#define NBL_MORTON_EMULATED_DECODE_MASK(DIM, MASK, HEX_VALUE) 

#else

#define NBL_MORTON_EMULATED_DECODE_MASK(DIM, MASK, HEX_VALUE) template<> struct morton_mask_##DIM##_##MASK##<emulated_uint64_t>\
{\
    NBL_CONSTEXPR_STATIC_INLINE emulated_uint64_t value;\
};\
NBL_CONSTEXPR_STATIC_INLINE emulated_uint64_t morton_mask_##DIM##_##MASK##<emulated_uint64_t>::value = _static_cast<emulated_uint64_t>(HEX_VALUE);
#endif

#define NBL_MORTON_DECODE_MASK(DIM, MASK, HEX_VALUE) template<typename T> struct morton_mask_##DIM##_##MASK ;\
        NBL_MORTON_EMULATED_DECODE_MASK(DIM, MASK, HEX_VALUE)\
        NBL_MORTON_GENERIC_DECODE_MASK(DIM, MASK, HEX_VALUE)\
        template<typename T>\
        NBL_CONSTEXPR T morton_mask_##DIM##_##MASK##_v = morton_mask_##DIM##_##MASK##<T>::value;

NBL_MORTON_DECODE_MASK(2, 0, uint64_t(0x5555555555555555)) // Groups bits by 1  on, 1  off
NBL_MORTON_DECODE_MASK(2, 1, uint64_t(0x3333333333333333)) // Groups bits by 2  on, 2  off
NBL_MORTON_DECODE_MASK(2, 2, uint64_t(0x0F0F0F0F0F0F0F0F)) // Groups bits by 4  on, 4  off
NBL_MORTON_DECODE_MASK(2, 3, uint64_t(0x00FF00FF00FF00FF)) // Groups bits by 8  on, 8  off
NBL_MORTON_DECODE_MASK(2, 4, uint64_t(0x0000FFFF0000FFFF)) // Groups bits by 16 on, 16 off
NBL_MORTON_DECODE_MASK(2, 5, uint64_t(0x00000000FFFFFFFF)) // Groups bits by 32 on, 32 off

NBL_MORTON_DECODE_MASK(3, 0, uint64_t(0x1249249249249249)) // Groups bits by 1  on, 2  off - also limits each dimension to 21 bits
NBL_MORTON_DECODE_MASK(3, 1, uint64_t(0x01C0E070381C0E07)) // Groups bits by 3  on, 6  off
NBL_MORTON_DECODE_MASK(3, 2, uint64_t(0x0FC003F000FC003F)) // Groups bits by 6  on, 12 off
NBL_MORTON_DECODE_MASK(3, 3, uint64_t(0x0000FFF000000FFF)) // Groups bits by 12 on, 24 off
NBL_MORTON_DECODE_MASK(3, 4, uint64_t(0x0000000000FFFFFF)) // Groups bits by 24 on, 48 off

NBL_MORTON_DECODE_MASK(4, 0, uint64_t(0x1111111111111111)) // Groups bits by 1  on, 3  off
NBL_MORTON_DECODE_MASK(4, 1, uint64_t(0x0303030303030303)) // Groups bits by 2  on, 6  off
NBL_MORTON_DECODE_MASK(4, 2, uint64_t(0x000F000F000F000F)) // Groups bits by 4  on, 12 off
NBL_MORTON_DECODE_MASK(4, 3, uint64_t(0x000000FF000000FF)) // Groups bits by 8  on, 24 off
NBL_MORTON_DECODE_MASK(4, 4, uint64_t(0x000000000000FFFF)) // Groups bits by 16 on, 48 off

#undef NBL_MORTON_DECODE_MASK
#undef NBL_MORTON_EMULATED_DECODE_MASK
#undef NBL_MORTON_GENERIC_DECODE_MASK

// ----------------------------------------------------------------- MORTON ENCODERS ---------------------------------------------------

template<uint16_t Dim, uint16_t Bits, typename encode_t>
struct MortonEncoder;

template<uint16_t Bits, typename encode_t>
struct MortonEncoder<2, Bits, encode_t>
{
    template<typename decode_t>
    NBL_CONSTEXPR_STATIC_INLINE_FUNC encode_t encode(NBL_CONST_REF_ARG(decode_t) decodedValue)
    {
        left_shift_operator<encode_t> leftShift;
        encode_t encoded = _static_cast<encode_t>(decodedValue);
        NBL_IF_CONSTEXPR(Bits > 16)
        {
            encoded = (encoded | leftShift(encoded, 16)) & morton_mask_2_4_v<encode_t>;
        }
        NBL_IF_CONSTEXPR(Bits > 8)
        {
            encoded = (encoded | leftShift(encoded, 8)) & morton_mask_2_3_v<encode_t>;
        }
        NBL_IF_CONSTEXPR(Bits > 4)
        {
            encoded = (encoded | leftShift(encoded, 4)) & morton_mask_2_2_v<encode_t>;
        }
        NBL_IF_CONSTEXPR(Bits > 2)
        {
            encoded = (encoded | leftShift(encoded, 2)) & morton_mask_2_1_v<encode_t>;
        }
        encoded = (encoded | leftShift(encoded, 1)) & morton_mask_2_0_v<encode_t>;
        return encoded;
    }
};

template<uint16_t Bits, typename encode_t>
struct MortonEncoder<3, Bits, encode_t>
{
    template<typename decode_t>
    NBL_CONSTEXPR_STATIC_INLINE_FUNC encode_t encode(NBL_CONST_REF_ARG(decode_t) decodedValue)
    {
        left_shift_operator<encode_t> leftShift;
        encode_t encoded = _static_cast<encode_t>(decodedValue);
        NBL_IF_CONSTEXPR(Bits > 12)
        {
            encoded = (encoded | leftShift(encoded, 24)) & morton_mask_3_3_v<encode_t>;
        }
        NBL_IF_CONSTEXPR(Bits > 6)
        {
            encoded = (encoded | leftShift(encoded, 12)) & morton_mask_3_2_v<encode_t>;
        }
        NBL_IF_CONSTEXPR(Bits > 3)
        {
            encoded = (encoded | leftShift(encoded, 6)) & morton_mask_3_1_v<encode_t>;
        }
        encoded = (encoded | leftShift(encoded, 2) | leftShift(encoded, 4)) & morton_mask_3_0_v<encode_t>;
        return encoded;
    }
};

template<uint16_t Bits, typename encode_t>
struct MortonEncoder<4, Bits, encode_t>
{
    template<typename decode_t>
    NBL_CONSTEXPR_STATIC_INLINE_FUNC encode_t encode(NBL_CONST_REF_ARG(decode_t) decodedValue)
    {
        left_shift_operator<encode_t> leftShift;
        encode_t encoded = _static_cast<encode_t>(decodedValue);
        NBL_IF_CONSTEXPR(Bits > 8)
        {
            encoded = (encoded | leftShift(encoded, 24)) & morton_mask_4_3_v<encode_t>;
        }
        NBL_IF_CONSTEXPR(Bits > 4)
        {
            encoded = (encoded | leftShift(encoded, 12)) & morton_mask_4_2_v<encode_t>;
        }
        NBL_IF_CONSTEXPR(Bits > 2)
        {
            encoded = (encoded | leftShift(encoded, 6)) & morton_mask_4_1_v<encode_t>;
        }
        encoded = (encoded | leftShift(encoded, 3)) & morton_mask_4_0_v<encode_t>;
        return encoded;
    }
};

// ----------------------------------------------------------------- MORTON DECODERS ---------------------------------------------------

template<uint16_t Dim, uint16_t Bits, typename encode_t>
struct MortonDecoder;

template<uint16_t Bits, typename encode_t>
struct MortonDecoder<2, Bits, encode_t>
{
    template<typename decode_t>
    NBL_CONSTEXPR_STATIC_INLINE_FUNC decode_t decode(NBL_CONST_REF_ARG(encode_t) encodedValue)
    {
        arithmetic_right_shift_operator<encode_t> rightShift;
        encode_t decoded = encodedValue & morton_mask_2_0_v<encode_t>;
        NBL_IF_CONSTEXPR(Bits > 1)
        {
            decoded = (decoded | rightShift(decoded, 1)) & morton_mask_2_1_v<encode_t>;
        }
        NBL_IF_CONSTEXPR(Bits > 2)
        {
            decoded = (decoded | rightShift(decoded, 2)) & morton_mask_2_2_v<encode_t>;
        }
        NBL_IF_CONSTEXPR(Bits > 4)
        {
            decoded = (decoded | rightShift(decoded, 4)) & morton_mask_2_3_v<encode_t>;
        }
        NBL_IF_CONSTEXPR(Bits > 8)
        {
            decoded = (decoded | rightShift(decoded, 8)) & morton_mask_2_4_v<encode_t>;
        }
        NBL_IF_CONSTEXPR(Bits > 16)
        {
            decoded = (decoded | rightShift(decoded, 16)) & morton_mask_2_5_v<encode_t>;
        }

        return _static_cast<decode_t>(decoded);
    }
};

template<uint16_t Bits, typename encode_t>
struct MortonDecoder<3, Bits, encode_t>
{
    template<typename decode_t>
    NBL_CONSTEXPR_STATIC_INLINE_FUNC decode_t decode(NBL_CONST_REF_ARG(encode_t) encodedValue)
    {
        arithmetic_right_shift_operator<encode_t> rightShift;
        encode_t decoded = encodedValue & morton_mask_3_0_v<encode_t>;
        NBL_IF_CONSTEXPR(Bits > 1)
        {
            decoded = (decoded | rightShift(decoded, 2) | rightShift(decoded, 4)) & morton_mask_3_1_v<encode_t>;
        }
        NBL_IF_CONSTEXPR(Bits > 3)
        {
            decoded = (decoded | rightShift(decoded, 6)) & morton_mask_3_2_v<encode_t>;
        }
        NBL_IF_CONSTEXPR(Bits > 6)
        {
            decoded = (decoded | rightShift(decoded, 12)) & morton_mask_3_3_v<encode_t>;
        }
        NBL_IF_CONSTEXPR(Bits > 12)
        {
            decoded = (decoded | rightShift(decoded, 24)) & morton_mask_3_4_v<encode_t>;
        }

        return _static_cast<decode_t>(decoded);
    }
};

template<uint16_t Bits, typename encode_t>
struct MortonDecoder<4, Bits, encode_t>
{
    template<typename decode_t>
    NBL_CONSTEXPR_STATIC_INLINE_FUNC decode_t decode(NBL_CONST_REF_ARG(encode_t) encodedValue)
    {
        arithmetic_right_shift_operator<encode_t> rightShift;
        encode_t decoded = encodedValue & morton_mask_4_0_v<encode_t>;
        NBL_IF_CONSTEXPR(Bits > 1)
        {
            decoded = (decoded | rightShift(decoded, 3)) & morton_mask_4_1_v<encode_t>;
        }
        NBL_IF_CONSTEXPR(Bits > 2)
        {
            decoded = (decoded | rightShift(decoded, 6)) & morton_mask_4_2_v<encode_t>;
        }
        NBL_IF_CONSTEXPR(Bits > 4)
        {
            decoded = (decoded | rightShift(decoded, 12)) & morton_mask_4_3_v<encode_t>;
        }
        NBL_IF_CONSTEXPR(Bits > 8)
        {
            decoded = (decoded | rightShift(decoded, 24)) & morton_mask_4_4_v<encode_t>;
        }

        return _static_cast<decode_t>(decoded);
    }
};

} //namespace impl

// Making this even slightly less ugly is blocked by https://github.com/microsoft/DirectXShaderCompiler/issues/7006
// In particular, `Masks` should be a `const static` member field instead of appearing in every method using it
template<bool Signed, uint16_t Bits, uint16_t D, typename _uint64_t = uint64_t NBL_PRIMARY_REQUIRES(impl::MortonDimension<D> && D * Bits <= 64)
struct code
{
    using this_t = code<Signed, Bits, D, _uint64_t>;
    NBL_CONSTEXPR_STATIC uint16_t TotalBitWidth = D * Bits;
    using storage_t = conditional_t<(TotalBitWidth > 16), conditional_t<(TotalBitWidth > 32), _uint64_t, uint32_t>, uint16_t>;

    
    storage_t value;

    // ---------------------------------------------------- CONSTRUCTORS ---------------------------------------------------------------

    #ifndef __HLSL_VERSION

    code() = default;

    #endif

    /**
    * @brief Creates a Morton code from a set of integral cartesian coordinates
    *
    * @param [in] cartesian Coordinates to encode. Signedness MUST match the signedness of this Morton code class
    */
    template<typename I>
    NBL_CONSTEXPR_STATIC_FUNC enable_if_t<is_integral_v<I> && is_scalar_v<I> && (is_signed_v<I> == Signed), this_t>
    create(NBL_CONST_REF_ARG(vector<I, D>) cartesian)
    {
        using U = make_unsigned_t<I>;
        left_shift_operator<storage_t> leftShift;
        storage_t encodedCartesian = _static_cast<storage_t>(uint64_t(0));
        [[unroll]]
        for (uint16_t i = 0; i < D; i++)
        {
            encodedCartesian = encodedCartesian | leftShift(impl::MortonEncoder<D, Bits, storage_t>::encode(_static_cast<U>(cartesian[i])), i);
        }
        this_t retVal;
        retVal.value = encodedCartesian;
        return retVal;
    }

    // CPP can also have an actual constructor
    #ifndef __HLSL_VERSION

    /**
    * @brief Creates a Morton code from a set of cartesian coordinates
    *
    * @param [in] cartesian Coordinates to encode
    */

    template<typename I>
    explicit code(NBL_CONST_REF_ARG(vector<I, D>) cartesian)
    {
        *this = create(cartesian);
    }

    // This one is defined later since it requires `static_cast_helper` specialization 

    /**
    * @brief Decodes this Morton code back to a set of cartesian coordinates
    */

    template<typename I>
    explicit operator vector<I, D>() const noexcept
    {
        return _static_cast<vector<I, D>, morton::code<is_signed_v<I>, Bits, D>>(*this);
    }

    #endif
};

} //namespace morton

// Specialize the `static_cast_helper`
namespace impl
{
// I must be of same signedness as the morton code, and be wide enough to hold each component
template<typename I, uint16_t Bits, uint16_t D, typename _uint64_t> NBL_PARTIAL_REQ_TOP(concepts::IntegralScalar<I> && (8 * sizeof(I) >= Bits))
struct static_cast_helper<vector<I, D>, morton::code<is_signed_v<I>, Bits, D, _uint64_t> NBL_PARTIAL_REQ_BOT(concepts::IntegralScalar<I> && (8 * sizeof(I) >= Bits)) >
{
    NBL_CONSTEXPR_STATIC_INLINE_FUNC vector<I, D> cast(NBL_CONST_REF_ARG(morton::code<is_signed_v<I>, Bits, D, _uint64_t>) val)
    {
        using U = make_unsigned_t<I>;
        using storage_t = typename morton::code<is_signed_v<I>, Bits, D, _uint64_t>::storage_t;
        arithmetic_right_shift_operator<storage_t> rightShift;
        vector<I, D> cartesian;
        for (uint16_t i = 0; i < D; i++)
        {
            cartesian[i] = _static_cast<I>(morton::impl::MortonDecoder<D, Bits, storage_t>::template decode<U>(rightShift(val.value, i)));
        }
        return cartesian;
    }
};

} // namespace impl

} //namespace hlsl
} //namespace nbl

#endif