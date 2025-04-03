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

#define NBL_MORTON_EMULATED_DECODE_MASK(DIM, MASK, HEX_HIGH_VALUE, HEX_LOW_VALUE) template<> struct morton_mask_##DIM##_##MASK##<emulated_uint64_t>\
{\
    NBL_CONSTEXPR_STATIC_INLINE emulated_uint64_t value = emulated_uint64_t::create(uint32_t(0x##HEX_HIGH_VALUE), uint32_t(0x##HEX_LOW_VALUE));\
};

#else

#define NBL_MORTON_EMULATED_DECODE_MASK(DIM, MASK, HEX_HIGH_VALUE, HEX_LOW_VALUE) template<> struct morton_mask_##DIM##_##MASK##<emulated_uint64_t>\
{\
    NBL_CONSTEXPR_STATIC_INLINE emulated_uint64_t value;\
};\
NBL_CONSTEXPR_STATIC_INLINE emulated_uint64_t morton_mask_##DIM##_##MASK##<emulated_uint64_t>::value = emulated_uint64_t::create(uint32_t(0x##HEX_HIGH_VALUE), uint32_t(0x##HEX_LOW_VALUE));
#endif

#define NBL_MORTON_DECODE_MASK(DIM, MASK, HEX_HIGH_VALUE, HEX_LOW_VALUE) template<typename T> struct morton_mask_##DIM##_##MASK ;\
        NBL_MORTON_EMULATED_DECODE_MASK(DIM, MASK, HEX_HIGH_VALUE, HEX_LOW_VALUE)\
        NBL_MORTON_GENERIC_DECODE_MASK(DIM, MASK, 0x##HEX_HIGH_VALUE##HEX_LOW_VALUE)\
        template<typename T>\
        NBL_CONSTEXPR T morton_mask_##DIM##_##MASK##_v = morton_mask_##DIM##_##MASK##<T>::value;

NBL_MORTON_DECODE_MASK(2, 0, 55555555, 55555555) // Groups bits by 1  on, 1  off
NBL_MORTON_DECODE_MASK(2, 1, 33333333, 33333333) // Groups bits by 2  on, 2  off
NBL_MORTON_DECODE_MASK(2, 2, 0F0F0F0F, 0F0F0F0F) // Groups bits by 4  on, 4  off
NBL_MORTON_DECODE_MASK(2, 3, 00FF00FF, 00FF00FF) // Groups bits by 8  on, 8  off
NBL_MORTON_DECODE_MASK(2, 4, 0000FFFF, 0000FFFF) // Groups bits by 16 on, 16 off
NBL_MORTON_DECODE_MASK(2, 5, 00000000, FFFFFFFF) // Groups bits by 32 on, 32 off

NBL_MORTON_DECODE_MASK(3, 0, 12492492, 49249249) // Groups bits by 1  on, 2  off - also limits each dimension to 21 bits
NBL_MORTON_DECODE_MASK(3, 1, 01C0E070, 381C0E07) // Groups bits by 3  on, 6  off
NBL_MORTON_DECODE_MASK(3, 2, 0FC003F0, 00FC003F) // Groups bits by 6  on, 12 off
NBL_MORTON_DECODE_MASK(3, 3, 0000FFF0, 00000FFF) // Groups bits by 12 on, 24 off
NBL_MORTON_DECODE_MASK(3, 4, 00000000, 00FFFFFF) // Groups bits by 24 on, 48 off

NBL_MORTON_DECODE_MASK(4, 0, 11111111, 11111111) // Groups bits by 1  on, 3  off
NBL_MORTON_DECODE_MASK(4, 1, 03030303, 03030303) // Groups bits by 2  on, 6  off
NBL_MORTON_DECODE_MASK(4, 2, 000F000F, 000F000F) // Groups bits by 4  on, 12 off
NBL_MORTON_DECODE_MASK(4, 3, 000000FF, 000000FF) // Groups bits by 8  on, 24 off
NBL_MORTON_DECODE_MASK(4, 4, 00000000, 0000FFFF) // Groups bits by 16 on, 48 off

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

/*
template<uint16_t Bits>
struct MortonDecoder<2, Bits, emulated_uint64_t>
{
    template<typename decode_t>
    NBL_CONSTEXPR_STATIC_INLINE_FUNC decode_t decode(NBL_CONST_REF_ARG(emulated_uint64_t) encodedValue)
    {
        NBL_CONSTEXPR_STATIC uint16_t MaxIterations = uint16_t(mpl::log2_v<Bits>) + uint16_t(!mpl::is_pot_v<Bits>);

        NBL_CONSTEXPR_STATIC emulated_uint64_t DecodeMasks[6] = { emulated_uint64_t::create(uint32_t(0x55555555), uint32_t(0x55555555)),  
                                                                  emulated_uint64_t::create(uint32_t(0x33333333), uint32_t(0x33333333)),  
                                                                  emulated_uint64_t::create(uint32_t(0x0F0F0F0F), uint32_t(0x0F0F0F0F)),  
                                                                  emulated_uint64_t::create(uint32_t(0x00FF00FF), uint32_t(0x00FF00FF)),  
                                                                  emulated_uint64_t::create(uint32_t(0x0000FFFF), uint32_t(0x0000FFFF)),  
                                                                  emulated_uint64_t::create(uint32_t(0x00000000), uint32_t(0xFFFFFFFF)) };

        arithmetic_right_shift_operator<emulated_uint64_t> rightShift;

        emulated_uint64_t decoded = encodedValue & DecodeMasks[0];
        [[unroll]]
        for (uint16_t i = 0, shift = 1; i < MaxIterations; i++, shift <<= 1)
        {
            decoded = (decoded | rightShift(decoded, shift)) & DecodeMasks[i + 1];
        }
        return _static_cast<decode_t>(decoded.data.y);
    }
};

template<uint16_t Bits>
struct MortonDecoder<3, Bits, emulated_uint64_t>
{
    template<typename decode_t>
    NBL_CONSTEXPR_STATIC_INLINE_FUNC decode_t decode(NBL_CONST_REF_ARG(emulated_uint64_t) encodedValue)
    {
        NBL_CONSTEXPR_STATIC uint16_t MaxIterations = conditional_value<(Bits <= 3), uint16_t, uint16_t(1),
                                                      conditional_value<(Bits <= 6), uint16_t, uint16_t(2),
                                                      conditional_value<(Bits <= 12), uint16_t, uint16_t(3), uint16_t(4)>::value>::value>::value;

        NBL_CONSTEXPR_STATIC emulated_uint64_t DecodeMasks[5] = { emulated_uint64_t::create(uint32_t(0x12492492), uint32_t(0x49249249)),   (also only considers 21 bits)
                                                                  emulated_uint64_t::create(uint32_t(0x01C0E070), uint32_t(0x381C0E07)),  
                                                                  emulated_uint64_t::create(uint32_t(0x0FC003F0), uint32_t(0x00FC003F)),  
                                                                  emulated_uint64_t::create(uint32_t(0x0000FFF0), uint32_t(0x00000FFF)),  
                                                                  emulated_uint64_t::create(uint32_t(0x00000000), uint32_t(0x00FFFFFF)) }; (40 off if you're feeling pedantic)

        arithmetic_right_shift_operator<emulated_uint64_t> rightShift;

        emulated_uint64_t decoded = encodedValue & DecodeMasks[0];
        // First iteration is special
        decoded = (decoded | rightShift(decoded, 2) | rightShift(decoded, 4)) & DecodeMasks[1];
        [[unroll]]
        for (uint16_t i = 0, shift = 6; i < MaxIterations - 1; i++, shift <<= 1)
        {
            decoded = (decoded | rightShift(decoded, shift)) & DecodeMasks[i + 2];
        }
        return _static_cast<decode_t>(decoded.data.y);
    }
};

template<uint16_t Bits>
struct MortonDecoder<4, Bits, emulated_uint64_t>
{
    template<typename decode_t>
    NBL_CONSTEXPR_STATIC_INLINE_FUNC decode_t decode(NBL_CONST_REF_ARG(emulated_uint64_t) encodedValue)
    {
        NBL_CONSTEXPR_STATIC uint16_t MaxIterations = uint16_t(mpl::log2_v<Bits>) + uint16_t(!mpl::is_pot_v<Bits>);

        NBL_CONSTEXPR_STATIC emulated_uint64_t DecodeMasks[5] = { emulated_uint64_t::create(uint32_t(0x11111111), uint32_t(0x11111111)),  
                                                                  emulated_uint64_t::create(uint32_t(0x03030303), uint32_t(0x03030303)),  
                                                                  emulated_uint64_t::create(uint32_t(0x000F000F), uint32_t(0x000F000F)),  
                                                                  emulated_uint64_t::create(uint32_t(0x000000FF), uint32_t(0x000000FF)),  
                                                                  emulated_uint64_t::create(uint32_t(0x00000000), uint32_t(0x0000FFFF)) };

        arithmetic_right_shift_operator<emulated_uint64_t> rightShift;

        emulated_uint64_t decoded = encodedValue & DecodeMasks[0];
        [[unroll]]
        for (uint16_t i = 0, shift = 3; i < MaxIterations; i++, shift <<= 1)
        {
            decoded = (decoded | rightShift(decoded, shift)) & DecodeMasks[i + 1];
        }
        return _static_cast<decode_t>(decoded.data.y);
    }
};

template<uint16_t Bits, typename encode_t>
struct MortonDecoder<2, Bits, encode_t>
{
    template<typename decode_t>
    NBL_CONSTEXPR_STATIC_INLINE_FUNC decode_t decode(NBL_CONST_REF_ARG(encode_t) encodedValue)
    {
        NBL_CONSTEXPR_STATIC uint16_t MaxIterations = uint16_t(mpl::log2_v<Bits>) + uint16_t(!mpl::is_pot_v<Bits>);

        NBL_CONSTEXPR_STATIC encode_t DecodeMasks[6] = { _static_cast<encode_t>(0x5555555555555555),  // Groups bits by 1  on, 1  off
                                                         _static_cast<encode_t>(0x3333333333333333),  // Groups bits by 2  on, 2  off
                                                         _static_cast<encode_t>(0x0F0F0F0F0F0F0F0F),  // Groups bits by 4  on, 4  off
                                                         _static_cast<encode_t>(0x00FF00FF00FF00FF),  // Groups bits by 8  on, 8  off
                                                         _static_cast<encode_t>(0x0000FFFF0000FFFF),  // Groups bits by 16 on, 16 off
                                                         _static_cast<encode_t>(0x00000000FFFFFFFF) };// Groups bits by 32 on, 32 off
        
        encode_t decoded = encodedValue & DecodeMasks[0];
        [[unroll]]
        for (uint16_t i = 0, shift = 1; i < MaxIterations; i++, shift <<= 1)
        {
            decoded = (decoded | (decoded >> shift)) & DecodeMasks[i + 1];
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
        NBL_CONSTEXPR_STATIC uint16_t MaxIterations = conditional_value<(Bits <= 3), uint16_t, uint16_t(1),
                                                      conditional_value<(Bits <= 6), uint16_t, uint16_t(2), 
                                                      conditional_value<(Bits <= 12), uint16_t, uint16_t(3), uint16_t(4)>::value>::value>::value;

        NBL_CONSTEXPR_STATIC encode_t DecodeMasks[5] = { _static_cast<encode_t>(0x1249249249249249),  // Groups bits by 1  on, 2  off (also only considers 21 bits)
                                                         _static_cast<encode_t>(0x01C0E070381C0E07),  // Groups bits by 3  on, 6  off
                                                         _static_cast<encode_t>(0x0FC003F000FC003F),  // Groups bits by 6  on, 12 off
                                                         _static_cast<encode_t>(0x0000FFF000000FFF),  // Groups bits by 12 on, 24 off
                                                         _static_cast<encode_t>(0x0000000000FFFFFF) };// Groups bits by 24 on, 48 off (40 off if you're feeling pedantic)

        encode_t decoded = encodedValue & DecodeMasks[0];
        // First iteration is special
        decoded = (decoded | (decoded >> 2) | (decoded >> 4)) & DecodeMasks[1];
        [[unroll]]
        for (uint16_t i = 0, shift = 6; i < MaxIterations - 1; i++, shift <<= 1)
        {
            decoded = (decoded | (decoded >> shift)) & DecodeMasks[i + 2];
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
        NBL_CONSTEXPR_STATIC uint16_t MaxIterations = uint16_t(mpl::log2_v<Bits>) + uint16_t(!mpl::is_pot_v<Bits>);

        NBL_CONSTEXPR_STATIC encode_t DecodeMasks[5] = { _static_cast<encode_t>(0x1111111111111111),  // Groups bits by 1  on, 3  off
                                                         _static_cast<encode_t>(0x0303030303030303),  // Groups bits by 2  on, 6  off
                                                         _static_cast<encode_t>(0x000F000F000F000F),  // Groups bits by 4  on, 12 off
                                                         _static_cast<encode_t>(0x000000FF000000FF),  // Groups bits by 8  on, 24 off
                                                         _static_cast<encode_t>(0x000000000000FFFF) };// Groups bits by 16 on, 48 off

        encode_t decoded = encodedValue & DecodeMasks[0];
        [[unroll]]
        for (uint16_t i = 0, shift = 3; i < MaxIterations; i++, shift <<= 1)
        {
            decoded = (decoded | (decoded >> shift)) & DecodeMasks[i + 1];
        }
        return _static_cast<decode_t>(decoded);
    }
};

*/

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
};

} //namespace morton
} //namespace hlsl
} //namespace nbl

#endif