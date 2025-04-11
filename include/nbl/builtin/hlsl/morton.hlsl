#ifndef _NBL_BUILTIN_HLSL_MORTON_INCLUDED_
#define _NBL_BUILTIN_HLSL_MORTON_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/concepts/core.hlsl"
#include "nbl/builtin/hlsl/bit.hlsl"
#include "nbl/builtin/hlsl/functional.hlsl"
#include "nbl/builtin/hlsl/emulated/int64_t.hlsl"
#include "nbl/builtin/hlsl/mpl.hlsl"
#include "nbl/builtin/hlsl/portable/vector_t.hlsl"

// TODO: mega macro to get functional plus, minus, plus_assign, minus_assign

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
NBL_BOOL_CONCEPT Dimension = 1 < D && D < 5;

// --------------------------------------------------------- MORTON ENCODE/DECODE MASKS ---------------------------------------------------

NBL_CONSTEXPR uint16_t CodingStages = 5;

template<uint16_t Dim, uint16_t Bits, uint16_t Stage>
struct coding_mask;

template<uint16_t Dim, uint16_t Bits, uint16_t Stage>
NBL_CONSTEXPR uint64_t coding_mask_v = coding_mask<Dim, Bits, Stage>::value;

// 0th stage will be special: to avoid masking twice during encode/decode, and to get a proper mask that only gets the relevant bits out of a morton code, the 0th stage
// mask also considers the total number of bits we're cnsidering for a code (all other masks operate on a bit-agnostic basis).
#define NBL_HLSL_MORTON_SPECIALIZE_FIRST_CODING_MASK(DIM, BASE_VALUE) template<uint16_t Bits> struct coding_mask<DIM, Bits, 0>\
{\
    enum : uint64_t { _Bits = Bits };\
    NBL_CONSTEXPR_STATIC_INLINE uint64_t KilloffMask = _Bits * DIM < 64 ? (uint64_t(1) << (_Bits * DIM)) - 1 : ~uint64_t(0);\
    NBL_CONSTEXPR_STATIC_INLINE uint64_t value = uint64_t(BASE_VALUE) & KilloffMask;\
};

#define NBL_HLSL_MORTON_SPECIALIZE_CODING_MASK(DIM, STAGE, BASE_VALUE) template<uint16_t Bits> struct coding_mask<DIM, Bits, STAGE>\
{\
    NBL_CONSTEXPR_STATIC_INLINE uint64_t value = uint64_t(BASE_VALUE);\
};

// Final stage mask also counts exact number of bits, although maybe it's not necessary
#define NBL_HLSL_MORTON_SPECIALIZE_LAST_CODING_MASKS template<uint16_t Dim, uint16_t Bits> struct coding_mask<Dim, Bits, CodingStages>\
{\
    enum : uint64_t { _Bits = Bits };\
    NBL_CONSTEXPR_STATIC_INLINE uint64_t value = (uint64_t(1) << _Bits) - 1;\
};

NBL_HLSL_MORTON_SPECIALIZE_FIRST_CODING_MASK(2, 0x5555555555555555)        // Groups bits by 1  on, 1  off
NBL_HLSL_MORTON_SPECIALIZE_CODING_MASK(2, 1, uint64_t(0x3333333333333333)) // Groups bits by 2  on, 2  off
NBL_HLSL_MORTON_SPECIALIZE_CODING_MASK(2, 2, uint64_t(0x0F0F0F0F0F0F0F0F)) // Groups bits by 4  on, 4  off
NBL_HLSL_MORTON_SPECIALIZE_CODING_MASK(2, 3, uint64_t(0x00FF00FF00FF00FF)) // Groups bits by 8  on, 8  off
NBL_HLSL_MORTON_SPECIALIZE_CODING_MASK(2, 4, uint64_t(0x0000FFFF0000FFFF)) // Groups bits by 16 on, 16 off

NBL_HLSL_MORTON_SPECIALIZE_FIRST_CODING_MASK(3, 0x9249249249249249)        // Groups bits by 1  on, 2  off
NBL_HLSL_MORTON_SPECIALIZE_CODING_MASK(3, 1, uint64_t(0x30C30C30C30C30C3)) // Groups bits by 2  on, 4  off
NBL_HLSL_MORTON_SPECIALIZE_CODING_MASK(3, 2, uint64_t(0xF00F00F00F00F00F)) // Groups bits by 4  on, 8  off
NBL_HLSL_MORTON_SPECIALIZE_CODING_MASK(3, 3, uint64_t(0x00FF0000FF0000FF)) // Groups bits by 8  on, 16 off
NBL_HLSL_MORTON_SPECIALIZE_CODING_MASK(3, 4, uint64_t(0xFFFF00000000FFFF)) // Groups bits by 16 on, 32 off

NBL_HLSL_MORTON_SPECIALIZE_FIRST_CODING_MASK(4, 0x1111111111111111)        // Groups bits by 1  on, 3  off
NBL_HLSL_MORTON_SPECIALIZE_CODING_MASK(4, 1, uint64_t(0x0303030303030303)) // Groups bits by 2  on, 6  off
NBL_HLSL_MORTON_SPECIALIZE_CODING_MASK(4, 2, uint64_t(0x000F000F000F000F)) // Groups bits by 4  on, 12 off
NBL_HLSL_MORTON_SPECIALIZE_CODING_MASK(4, 3, uint64_t(0x000000FF000000FF)) // Groups bits by 8  on, 24 off
NBL_HLSL_MORTON_SPECIALIZE_CODING_MASK(4, 4, uint64_t(0x000000000000FFFF)) // Groups bits by 16 on, 48 off (unused but here for completion + likely keeps compiler from complaining)

NBL_HLSL_MORTON_SPECIALIZE_LAST_CODING_MASKS

#undef NBL_HLSL_MORTON_SPECIALIZE_LAST_CODING_MASK
#undef NBL_HLSL_MORTON_SPECIALIZE_CODING_MASK
#undef NBL_HLSL_MORTON_SPECIALIZE_FIRST_CODING_MASK

// ----------------------------------------------------------------- MORTON ENCODER ---------------------------------------------------

template<uint16_t Dim, uint16_t Bits, typename encode_t NBL_PRIMARY_REQUIRES(Dimension<Dim> && (Dim * Bits <= 64) && (sizeof(encode_t) * 8 >= Dim * Bits))
struct MortonEncoder
{
    template<typename decode_t>
    NBL_CONSTEXPR_STATIC_INLINE_FUNC encode_t encode(NBL_CONST_REF_ARG(decode_t) decodedValue)
    {
        left_shift_operator<encode_t> leftShift;
        encode_t encoded = _static_cast<encode_t>(decodedValue);
        NBL_IF_CONSTEXPR(Bits > 16)
        {
            encoded = encoded | leftShift(encoded, 16 * (Dim - 1));
        }
        NBL_IF_CONSTEXPR(Bits > 8)
        {
            encoded = encoded & _static_cast<encode_t>(coding_mask_v<Dim, Bits, 4>);
            encoded = encoded | leftShift(encoded, 8 * (Dim - 1));
        }
        NBL_IF_CONSTEXPR(Bits > 4)
        {
            encoded = encoded & _static_cast<encode_t>(coding_mask_v<Dim, Bits, 3>);
            encoded = encoded | leftShift(encoded, 4 * (Dim - 1));
        }
        NBL_IF_CONSTEXPR(Bits > 2)
        {
            encoded = encoded & _static_cast<encode_t>(coding_mask_v<Dim, Bits, 2>);
            encoded = encoded | leftShift(encoded, 2 * (Dim - 1));
        }
        NBL_IF_CONSTEXPR(Bits > 1)
        {
            encoded = encoded & _static_cast<encode_t>(coding_mask_v<Dim, Bits, 1>);
            encoded = encoded | leftShift(encoded, 1 * (Dim - 1));
        }
        return encoded & _static_cast<encode_t>(coding_mask_v<Dim, Bits, 0>);
    }
};

// ----------------------------------------------------------------- MORTON DECODER ---------------------------------------------------

template<uint16_t Dim, uint16_t Bits, typename encode_t NBL_PRIMARY_REQUIRES(Dimension<Dim> && (Dim* Bits <= 64) && (sizeof(encode_t) * 8 >= Dim * Bits))
struct MortonDecoder
{
    template<typename decode_t = conditional_t<(Bits > 16), uint32_t, uint16_t>
    NBL_FUNC_REQUIRES(concepts::IntVector<decode_t> && sizeof(vector_traits<decode_t>::scalar_type) * 8 >= Bits)
    NBL_CONSTEXPR_STATIC_INLINE_FUNC decode_t decode(NBL_CONST_REF_ARG(encode_t) encodedValue)
    {
        arithmetic_right_shift_operator<portable_vector_t<encode_t, Dim> > rightShift;
        portable_vector_t<encode_t, Dim> decoded;
        NBL_IF_CONSTEXPR(Bits > 1)
        {
            decoded = decoded & _static_cast<encode_t>(coding_mask_v<Dim, Bits, 0>);
            decoded = decoded | rightShift(decoded, 1 * (Dim - 1));
        }
        NBL_IF_CONSTEXPR(Bits > 2)
        {
            decoded = decoded & _static_cast<encode_t>(coding_mask_v<Dim, Bits, 1>);
            decoded = decoded | rightShift(decoded, 2 * (Dim - 1));
        }
        NBL_IF_CONSTEXPR(Bits > 4)
        {
            decoded = decoded & _static_cast<encode_t>(coding_mask_v<Dim, Bits, 2>);
            decoded = decoded | rightShift(decoded, 4 * (Dim - 1));
        }
        NBL_IF_CONSTEXPR(Bits > 8)
        {
            decoded = decoded & _static_cast<encode_t>(coding_mask_v<Dim, Bits, 3>);
            decoded = decoded | rightShift(decoded, 8 * (Dim - 1));
        }
        NBL_IF_CONSTEXPR(Bits > 16)
        {
            decoded = decoded & _static_cast<encode_t>(coding_mask_v<Dim, Bits, 4>);
            decoded = decoded | rightShift(decoded, 16 * (Dim - 1));
        }

        return _static_cast<decode_t>(decoded & _static_cast<encode_t>(coding_mask_v<Dim, Bits, 5>));
    }
};

// ---------------------------------------------------- COMPARISON OPERATORS ---------------------------------------------------------------
// Here because no partial specialization of methods

template<bool Signed, uint16_t Bits, uint16_t D, typename storage_t, bool BitsAlreadySpread>
struct Equals;

template<bool Signed, uint16_t Bits, uint16_t D, typename storage_t>
struct Equals<Signed, Bits, D, storage_t, true>
{
    NBL_CONSTEXPR_INLINE_FUNC vector<bool, D> operator()(NBL_CONST_REF_ARG(storage_t) value, NBL_CONST_REF_ARG(vector<storage_t, D>) rhs)
    {
        NBL_CONSTEXPR_STATIC storage_t Mask = _static_cast<storage_t>(impl::coding_mask_v<D, Bits, 0>);
        left_shift_operator<storage_t> leftShift;
        vector<bool, D> retVal;
        [[unroll]]
        for (uint16_t i = 0; i < D; i++)
        {
            retVal[i] = (value & leftShift(Mask, i)) == leftShift(rhs[i], i);
        }
        return retVal;
    }
};

template<bool Signed, uint16_t Bits, uint16_t D, typename storage_t>
struct Equals<Signed, Bits, D, storage_t, false>
{
    template <typename I>
    NBL_CONSTEXPR_INLINE_FUNC enable_if_t<is_integral_v<I>&& is_scalar_v<I> && (is_signed_v<I> == Signed), vector<bool, D> >
    operator()(NBL_CONST_REF_ARG(storage_t) value, NBL_CONST_REF_ARG(vector<I, D>) rhs)
    {
        using U = make_unsigned_t<I>;
        vector<storage_t, D> interleaved;
        [[unroll]]
        for (uint16_t i = 0; i < D; i++)
        {
            interleaved[i] = impl::MortonEncoder<D, Bits, storage_t>::encode(_static_cast<U>(rhs[i]));
        }
        Equals<Signed, Bits, D, storage_t, true> equals;
        return equals(value, interleaved);
    }
};

template<bool Signed, uint16_t Bits, uint16_t D, typename storage_t, bool BitsAlreadySpread, typename ComparisonOp>
struct BaseComparison;

// Aux method for extracting highest bit, used by the comparison below
template<uint16_t Bits, uint16_t D, typename storage_t>
NBL_CONSTEXPR_INLINE_FUNC storage_t extractHighestBit(storage_t value, uint16_t coord)
{
    // Like above, if the number encoded in `coord` gets `bits(coord) = ceil((BitWidth - coord)/D)` bits for representation, then the highest index of these
    // bits is `bits(coord) - 1`
    const uint16_t coordHighestBitIdx = Bits / D - ((coord < Bits % D) ? uint16_t(0) : uint16_t(1));
    // This is the index of that bit as an index in the encoded value
    const uint16_t shift = coordHighestBitIdx * D + coord;
    left_shift_operator<storage_t> leftShift;
    return value & leftShift(_static_cast<storage_t>(uint16_t(1)), shift);
}

template<bool Signed, uint16_t Bits, uint16_t D, typename storage_t, typename ComparisonOp>
struct BaseComparison<Signed, Bits, D, storage_t, true, ComparisonOp>
{
    NBL_CONSTEXPR_INLINE_FUNC vector<bool, D> operator()(NBL_CONST_REF_ARG(storage_t) value, NBL_CONST_REF_ARG(vector<storage_t, D>) rhs)
    {
        NBL_CONSTEXPR_STATIC storage_t Mask = _static_cast<storage_t>(impl::coding_mask_v<D, Bits, 0>);
        left_shift_operator<storage_t> leftShift;
        vector<bool, D> retVal;
        ComparisonOp comparison;
        [[unroll]]
        for (uint16_t i = 0; i < D; i++)
        {
            storage_t thisCoord = value & leftShift(Mask, i);
            storage_t rhsCoord = leftShift(rhs[i], i);
            // If coordinate is negative, we add 1s in every bit not corresponding to coord
            if (extractHighestBit<Bits, D, storage_t>(thisCoord) != _static_cast<storage_t>(uint64_t(0)))
                thisCoord = thisCoord | ~leftShift(Mask, i);
            if (extractHighestBit<Bits, D, storage_t>(rhsCoord) != _static_cast<storage_t>(uint64_t(0)))
                rhsCoord = rhsCoord | ~leftShift(Mask, i);
            retVal[i] = comparison(thisCoord, rhsCoord);
        }
        return retVal;
    }
};

template<bool Signed, uint16_t Bits, uint16_t D, typename storage_t, typename ComparisonOp>
struct BaseComparison<Signed, Bits, D, storage_t, false, ComparisonOp>
{
    template <typename I>
    NBL_CONSTEXPR_INLINE_FUNC enable_if_t<is_integral_v<I>&& is_scalar_v<I> && (is_signed_v<I> == Signed), vector<bool, D> >
    operator()(NBL_CONST_REF_ARG(storage_t) value, NBL_CONST_REF_ARG(vector<I, D>) rhs)
    {
        using U = make_unsigned_t<I>;
        vector<storage_t, D> interleaved;
        [[unroll]]
        for (uint16_t i = 0; i < D; i++)
        {
            interleaved[i] = impl::MortonEncoder<D, Bits, storage_t>::encode(_static_cast<U>(rhs[i]));
        }
        BaseComparison<Signed, Bits, D, storage_t, true, ComparisonOp> baseComparison;
        return baseComparison(value, interleaved);
    }
};

template<bool Signed, uint16_t Bits, uint16_t D, typename storage_t, bool BitsAlreadySpread>
struct LessThan : BaseComparison<Signed, Bits, D, storage_t, BitsAlreadySpread, less<storage_t> > {};

template<bool Signed, uint16_t Bits, uint16_t D, typename storage_t, bool BitsAlreadySpread>
struct LessEquals : BaseComparison<Signed, Bits, D, storage_t, BitsAlreadySpread, less_equal<storage_t> > {};


} //namespace impl

// Making this even slightly less ugly is blocked by https://github.com/microsoft/DirectXShaderCompiler/issues/7006
// In particular, `Masks` should be a `const static` member field instead of appearing in every method using it
template<bool Signed, uint16_t Bits, uint16_t D, typename _uint64_t = uint64_t NBL_PRIMARY_REQUIRES(impl::Dimension<D> && D * Bits <= 64)
struct code
{
    using this_t = code<Signed, Bits, D, _uint64_t>;
    using this_signed_t = code<true, Bits, D, _uint64_t>;
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

    /**
    * @brief Decodes this Morton code back to a set of cartesian coordinates
    */
    template<typename I>
    constexpr inline explicit operator vector<I, D>() const noexcept
    {
        return _static_cast<vector<I, D>, morton::code<is_signed_v<I>, Bits, D>>(*this);
    }

    #endif

    // ------------------------------------------------------- BITWISE OPERATORS -------------------------------------------------

    NBL_CONSTEXPR_INLINE_FUNC this_t operator&(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        this_t retVal;
        retVal.value = value & rhs.value;
        return retVal;
    }

    NBL_CONSTEXPR_INLINE_FUNC this_t operator|(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        this_t retVal;
        retVal.value = value | rhs.value;
        return retVal;
    }

    NBL_CONSTEXPR_INLINE_FUNC this_t operator^(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        this_t retVal;
        retVal.value = value ^ rhs.value;
        return retVal;
    }

    NBL_CONSTEXPR_INLINE_FUNC this_t operator~() NBL_CONST_MEMBER_FUNC
    {
        this_t retVal;
        retVal.value = ~value;
        return retVal;
    }

    // Only valid in CPP
    #ifndef __HLSL_VERSION

    constexpr inline this_t operator<<(uint16_t bits) const;

    constexpr inline this_t operator>>(uint16_t bits) const;

    #endif

    // ------------------------------------------------------- UNARY ARITHMETIC OPERATORS -------------------------------------------------

    NBL_CONSTEXPR_INLINE_FUNC this_signed_t operator-() NBL_CONST_MEMBER_FUNC
    {
        left_shift_operator<storage_t> leftShift;
        // allOnes encodes a cartesian coordinate with all values set to 1
        this_t allOnes;
        allOnes.value = leftShift(_static_cast<storage_t>(1), D) - _static_cast<storage_t>(1);
        // Using 2's complement property that arithmetic negation can be obtained by bitwise negation then adding 1
        this_signed_t retVal;
        retVal.value = (operator~() + allOnes).value;
        return retVal;
    }

    // ------------------------------------------------------- BINARY ARITHMETIC OPERATORS -------------------------------------------------

    NBL_CONSTEXPR_INLINE_FUNC this_t operator+(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        NBL_CONSTEXPR_STATIC storage_t Mask = _static_cast<storage_t>(impl::coding_mask_v<D, Bits, 0>);
        left_shift_operator<storage_t> leftShift;
        this_t retVal;
        retVal.value = _static_cast<storage_t>(uint64_t(0));
        [[unroll]]
        for (uint16_t i = 0; i < D; i++)
        {
            // put 1 bits everywhere in the bits the current axis is not using
            // then extract just the axis bits for the right hand coordinate
            // carry-1 will propagate the bits across the already set bits
            // then clear out the bits not belonging to current axis
            // Note: Its possible to clear on `this` and fill on `rhs` but that will
            // disable optimizations, we expect the compiler to optimize a lot if the
            // value of `rhs` is known at compile time, e.g. `static_cast<Morton<N>>(glm::ivec3(1,0,0))`
            retVal.value |= ((value | (~leftShift(Mask, i))) + (rhs.value & leftShift(Mask, i))) & leftShift(Mask, i);
        }
        return retVal;
    }

    NBL_CONSTEXPR_INLINE_FUNC this_t operator-(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        NBL_CONSTEXPR_STATIC storage_t Mask = _static_cast<storage_t>(impl::coding_mask_v<D, Bits, 0>);
        left_shift_operator<storage_t> leftShift;
        this_t retVal;
        retVal.value = _static_cast<storage_t>(uint64_t(0));
        [[unroll]]
        for (uint16_t i = 0; i < D; i++)
        {
            // This is the dual trick of the one used for addition: set all other bits to 0 so borrows propagate
            retVal.value |= ((value & leftShift(Mask, i)) - (rhs.value & leftShift(Mask, i))) & leftShift(Mask, i);
        }
        return retVal;
    }

    // ------------------------------------------------------- COMPARISON OPERATORS -------------------------------------------------

    NBL_CONSTEXPR_INLINE_FUNC bool operator==(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        return value == rhs.value;
    }

    template<bool BitsAlreadySpread, typename I>
    enable_if_t<(is_signed_v<I> == Signed) || (is_same_v<I, storage_t> && BitsAlreadySpread), vector<bool, D> > operator==(NBL_CONST_REF_ARG(vector<I, D>) rhs)
    {
        impl::Equals<Signed, Bits, D, storage_t, BitsAlreadySpread> equals;
        return equals(value, rhs);
    }

    NBL_CONSTEXPR_INLINE_FUNC bool operator!=(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        return value != rhs.value;
    }

    template<bool BitsAlreadySpread, typename I>
    enable_if_t<(is_signed_v<I> == Signed) || (is_same_v<I, storage_t> && BitsAlreadySpread), vector<bool, D> > operator!=(NBL_CONST_REF_ARG(vector<I, D>) rhs)
    {
        return !operator== <BitsAlreadySpread, I>(rhs);
    }

    template<bool BitsAlreadySpread, typename I>
    enable_if_t<(is_signed_v<I> == Signed) || (is_same_v<I, storage_t> && BitsAlreadySpread), vector<bool, D> > operator<(NBL_CONST_REF_ARG(vector<I, D>) rhs)
    {
        impl::LessThan<Signed, Bits, D, storage_t, BitsAlreadySpread> lessThan;
        return lessThan(value, rhs);
    }

    template<bool BitsAlreadySpread, typename I>
    enable_if_t<(is_signed_v<I> == Signed) || (is_same_v<I, storage_t> && BitsAlreadySpread), vector<bool, D> > operator<=(NBL_CONST_REF_ARG(vector<I, D>) rhs)
    {
        impl::LessEquals<Signed, Bits, D, storage_t, BitsAlreadySpread> lessEquals;
        return lessEquals(value, rhs);
    }

    template<bool BitsAlreadySpread, typename I>
    enable_if_t<(is_signed_v<I> == Signed) || (is_same_v<I, storage_t> && BitsAlreadySpread), vector<bool, D> > operator>(NBL_CONST_REF_ARG(vector<I, D>) rhs)
    {
        return !operator<= <BitsAlreadySpread, I>(rhs);
    }

    template<bool BitsAlreadySpread, typename I>
    enable_if_t<(is_signed_v<I> == Signed) || (is_same_v<I, storage_t> && BitsAlreadySpread), vector<bool, D> > operator>=(NBL_CONST_REF_ARG(vector<I, D>) rhs)
    {
        return !operator< <BitsAlreadySpread, I>(rhs);
    }

};

} //namespace morton

template<bool Signed, uint16_t Bits, uint16_t D, typename _uint64_t>
struct left_shift_operator<morton::code<Signed, Bits, D, _uint64_t> >
{
    using type_t = morton::code<Signed, Bits, D, _uint64_t>;
    using storage_t = typename type_t::storage_t;

    NBL_CONSTEXPR_INLINE_FUNC type_t operator()(NBL_CONST_REF_ARG(type_t) operand, uint16_t bits)
    {
        left_shift_operator<storage_t> valueLeftShift;
        type_t retVal;
        // Shift every coordinate by `bits`
        retVal.value = valueLeftShift(operand.value, bits * D);
        return retVal;
    }
};

template<uint16_t Bits, uint16_t D, typename _uint64_t>
struct arithmetic_right_shift_operator<morton::code<false, Bits, D, _uint64_t> >
{
    using type_t = morton::code<false, Bits, D, _uint64_t>;
    using storage_t = typename type_t::storage_t;

    NBL_CONSTEXPR_INLINE_FUNC type_t operator()(NBL_CONST_REF_ARG(type_t) operand, uint16_t bits)
    {
        arithmetic_right_shift_operator<storage_t> valueArithmeticRightShift;
        type_t retVal;
        // Shift every coordinate by `bits`
        retVal.value = valueArithmeticRightShift(operand.value, bits * D);
        return retVal;
    }
};

// This one's uglier - have to unpack to get the expected behaviour
template<uint16_t Bits, uint16_t D, typename _uint64_t>
struct arithmetic_right_shift_operator<morton::code<true, Bits, D, _uint64_t> >
{
    using type_t = morton::code<true, Bits, D, _uint64_t>;
    using scalar_t = conditional_t<(Bits > 16), int32_t, int16_t>;

    NBL_CONSTEXPR_INLINE_FUNC type_t operator()(NBL_CONST_REF_ARG(type_t) operand, uint16_t bits)
    {
        vector<scalar_t, D> cartesian = _static_cast<vector<scalar_t, D> >(operand);
        cartesian >> scalar_t(bits);
        return type_t::create(cartesian);
    }
};

#ifndef __HLSL_VERSION

template<bool Signed, uint16_t Bits, uint16_t D, typename _uint64_t NBL_FUNC_REQUIRES(morton::impl::Dimension<D>&& D* Bits <= 64)
constexpr inline morton::code<Signed, Bits, D, _uint64_t> morton::code<Signed, Bits, D, _uint64_t>::operator<<(uint16_t bits) const
{
    left_shift_operator<morton::code<Signed, Bits, D, _uint64_t>> leftShift;
    return leftShift(*this, bits);
}

template<bool Signed, uint16_t Bits, uint16_t D, typename _uint64_t NBL_FUNC_REQUIRES(morton::impl::Dimension<D>&& D* Bits <= 64)
constexpr inline morton::code<Signed, Bits, D, _uint64_t> morton::code<Signed, Bits, D, _uint64_t>::operator>>(uint16_t bits) const
{
    arithmetic_right_shift_operator<morton::code<Signed, Bits, D, _uint64_t>> rightShift;
    return rightShift(*this, bits);
}

#endif

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
        [[unroll]]
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