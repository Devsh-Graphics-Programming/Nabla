#ifndef _NBL_BUILTIN_HLSL_MORTON_INCLUDED_
#define _NBL_BUILTIN_HLSL_MORTON_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/concepts/core.hlsl"
#include "nbl/builtin/hlsl/bit.hlsl"
#include "nbl/builtin/hlsl/functional.hlsl"
#include "nbl/builtin/hlsl/emulated/int64_t.hlsl"
#include "nbl/builtin/hlsl/mpl.hlsl"
#include "nbl/builtin/hlsl/portable/vector_t.hlsl"
#include "nbl/builtin/hlsl/mpl.hlsl"

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

template<uint16_t Dim, uint16_t Bits, typename encode_t NBL_PRIMARY_REQUIRES(Dimension<Dim> && Dim * Bits <= 64 && 8 * sizeof(encode_t) == mpl::round_up_to_pot_v<Dim * Bits>)
struct MortonEncoder
{
    template<typename decode_t = conditional_t<(Bits > 16), vector<uint32_t, Dim>, vector<uint16_t, Dim> >
    NBL_FUNC_REQUIRES(concepts::IntVector<decode_t> && 8 * sizeof(typename vector_traits<decode_t>::scalar_type) >= Bits)
    /**
    * @brief Interleaves each coordinate with `Dim - 1` zeros inbetween each bit, and left-shifts each by their coordinate index
    *
    * @param [in] decodedValue Cartesian coordinates to interleave and shift
    */
    NBL_CONSTEXPR_STATIC_INLINE_FUNC portable_vector_t<encode_t, Dim> interleaveShift(NBL_CONST_REF_ARG(decode_t) decodedValue)
    {
        NBL_CONSTEXPR_STATIC encode_t EncodeMasks[CodingStages + 1] = { _static_cast<encode_t>(coding_mask_v<Dim, Bits, 0>), _static_cast<encode_t>(coding_mask_v<Dim, Bits, 1>), _static_cast<encode_t>(coding_mask_v<Dim, Bits, 2>) , _static_cast<encode_t>(coding_mask_v<Dim, Bits, 3>) , _static_cast<encode_t>(coding_mask_v<Dim, Bits, 4>) , _static_cast<encode_t>(coding_mask_v<Dim, Bits, 5>) };
        left_shift_operator<portable_vector_t<encode_t, Dim> > leftShift;
        portable_vector_t<encode_t, Dim> interleaved = _static_cast<portable_vector_t<encode_t, Dim> >(decodedValue)& EncodeMasks[CodingStages];

        NBL_CONSTEXPR_STATIC uint16_t Stages = mpl::log2_ceil_v<Bits>;
        [[unroll]]
        for (uint16_t i = Stages; i > 0; i--)
        {
            interleaved = interleaved | leftShift(interleaved, (uint32_t(1) << (i - 1)) * (Dim - 1));
            interleaved = interleaved & EncodeMasks[i - 1];
        }

        // After interleaving, shift each coordinate left by their index
        return leftShift(interleaved, _static_cast<vector<uint32_t, Dim> >(vector<uint32_t, 4>(0, 1, 2, 3)));
    }

    template<typename decode_t = conditional_t<(Bits > 16), vector<uint32_t, Dim>, vector<uint16_t, Dim> >
    NBL_FUNC_REQUIRES(concepts::IntVector<decode_t> && 8 * sizeof(typename vector_traits<decode_t>::scalar_type) >= Bits)
    /**
    * @brief Encodes a vector of cartesian coordinates as a Morton code
    *
    * @param [in] decodedValue Cartesian coordinates to encode
    */
    NBL_CONSTEXPR_STATIC_INLINE_FUNC encode_t encode(NBL_CONST_REF_ARG(decode_t) decodedValue)
    {
        portable_vector_t<encode_t, Dim> interleaveShifted = interleaveShift<decode_t>(decodedValue);

        encode_t encoded = _static_cast<encode_t>(uint64_t(0));
        array_get<portable_vector_t<encode_t, Dim>, encode_t> getter;
        [[unroll]]
        for (uint16_t i = 0; i < Dim; i++)
            encoded = encoded | getter(interleaveShifted, i);

        return encoded;
    }
};

// ----------------------------------------------------------------- MORTON DECODER ---------------------------------------------------

template<uint16_t Dim, uint16_t Bits, typename encode_t NBL_PRIMARY_REQUIRES(Dimension<Dim> && Dim * Bits <= 64 && 8 * sizeof(encode_t) == mpl::round_up_to_pot_v<Dim * Bits>)
struct MortonDecoder
{
    template<typename decode_t = conditional_t<(Bits > 16), vector<uint32_t, Dim>, vector<uint16_t, Dim> >
    NBL_FUNC_REQUIRES(concepts::IntVector<decode_t> && 8 * sizeof(typename vector_traits<decode_t>::scalar_type) >= Bits)
    NBL_CONSTEXPR_STATIC_INLINE_FUNC decode_t decode(NBL_CONST_REF_ARG(encode_t) encodedValue)
    {
        NBL_CONSTEXPR_STATIC encode_t DecodeMasks[CodingStages + 1] = { _static_cast<encode_t>(coding_mask_v<Dim, Bits, 0>), _static_cast<encode_t>(coding_mask_v<Dim, Bits, 1>), _static_cast<encode_t>(coding_mask_v<Dim, Bits, 2>) , _static_cast<encode_t>(coding_mask_v<Dim, Bits, 3>) , _static_cast<encode_t>(coding_mask_v<Dim, Bits, 4>) , _static_cast<encode_t>(coding_mask_v<Dim, Bits, 5>) };
        arithmetic_right_shift_operator<portable_vector_t<encode_t, Dim> > rightShift;
        portable_vector_t<encode_t, Dim> decoded;
        array_set<portable_vector_t<encode_t, Dim>, encode_t> setter;
        // Write initial values into decoded
        [[unroll]]
        for (uint16_t i = 0; i < Dim; i++)
            setter(decoded, i, encodedValue);
        decoded = rightShift(decoded, _static_cast<vector<uint32_t, Dim> >(vector<uint32_t, 4>(0, 1, 2, 3)));

        NBL_CONSTEXPR_STATIC uint16_t Stages = mpl::log2_ceil_v<Bits>;
        [[unroll]]
        for (uint16_t i = 0; i < Stages; i++)
        {
            decoded = decoded & DecodeMasks[i];
            decoded = decoded | rightShift(decoded, (uint32_t(1) << i) * (Dim - 1));
        }

        // If `Bits` is greater than half the bitwidth of the decode type, then we can avoid `&`ing against the last mask since duplicated MSB get truncated
        NBL_IF_CONSTEXPR(Bits > 4 * sizeof(typename vector_traits<decode_t>::scalar_type))
            return _static_cast<decode_t>(decoded);
        else
            return _static_cast<decode_t>(decoded & DecodeMasks[CodingStages]);
    }
};

// ---------------------------------------------------- COMPARISON OPERATORS ---------------------------------------------------------------
// Here because no partial specialization of methods
// `BitsAlreadySpread` assumes both pre-interleaved and pre-shifted

template<bool Signed, uint16_t Bits, typename storage_t, bool BitsAlreadySpread, typename I>
NBL_BOOL_CONCEPT Comparable = concepts::IntegralLikeScalar<I> && is_signed_v<I> == Signed && ((BitsAlreadySpread && sizeof(I) == sizeof(storage_t)) || (!BitsAlreadySpread && 8 * sizeof(I) == mpl::round_up_to_pot_v<Bits>));

template<bool Signed, uint16_t Bits, uint16_t D, typename storage_t, bool BitsAlreadySpread>
struct Equals;

template<bool Signed, uint16_t Bits, uint16_t D, typename storage_t>
struct Equals<Signed, Bits, D, storage_t, true>
{
    template<typename I NBL_FUNC_REQUIRES(Comparable<Signed, Bits, storage_t, true, I>)
    NBL_CONSTEXPR_STATIC_INLINE_FUNC vector<bool, D> __call(NBL_CONST_REF_ARG(storage_t) value, NBL_CONST_REF_ARG(portable_vector_t<I, D>) rhs)
    {
        NBL_CONSTEXPR portable_vector_t<storage_t, D> zeros = _static_cast<portable_vector_t<storage_t, D> >(_static_cast<vector<uint64_t, D> >(vector<uint64_t, 4>(0,0,0,0)));
        
        portable_vector_t<storage_t, D> rhsCasted = _static_cast<portable_vector_t<storage_t, D> >(rhs);
        portable_vector_t<storage_t, D> xored = rhsCasted ^ value;
        return xored == zeros;
    }
};

template<bool Signed, uint16_t Bits, uint16_t D, typename storage_t>
struct Equals<Signed, Bits, D, storage_t, false>
{
    template<typename I NBL_FUNC_REQUIRES(Comparable<Signed, Bits, storage_t, false, I>)
    NBL_CONSTEXPR_STATIC_INLINE_FUNC vector<bool, D> __call(NBL_CONST_REF_ARG(storage_t) value, NBL_CONST_REF_ARG(portable_vector_t<I, D>) rhs)
    {
        const portable_vector_t<storage_t, D> interleaved = MortonEncoder<D, Bits, storage_t>::interleaveShift(rhs);
        return Equals<Signed, Bits, D, storage_t, true>::__call(value, interleaved);
    }
};

template<bool Signed, uint16_t Bits, uint16_t D, typename storage_t, bool BitsAlreadySpread, typename ComparisonOp>
struct BaseComparison;

// Aux variable that has only the sign bit for the first of D dimensions
template<uint16_t Bits, uint16_t D>
NBL_CONSTEXPR uint64_t SignMask = uint64_t(1) << (D * (Bits - 1));

template<bool Signed, uint16_t Bits, uint16_t D, typename storage_t, typename ComparisonOp>
struct BaseComparison<Signed, Bits, D, storage_t, true, ComparisonOp>
{
    template<typename I NBL_FUNC_REQUIRES(Comparable<Signed, Bits, storage_t, true, I>)
    NBL_CONSTEXPR_STATIC_INLINE_FUNC vector<bool, D> __call(NBL_CONST_REF_ARG(storage_t) value, NBL_CONST_REF_ARG(portable_vector_t<I, D>) rhs)
    {
        NBL_CONSTEXPR_STATIC portable_vector_t<storage_t, D> InterleaveMasks = _static_cast<portable_vector_t<storage_t, D> >(_static_cast<vector<uint64_t, D> >(vector<uint64_t, 4>(coding_mask_v<D, Bits, 0>, coding_mask_v<D, Bits, 0> << 1, coding_mask_v<D, Bits, 0> << 2, coding_mask_v<D, Bits, 0> << 3)));
        NBL_CONSTEXPR_STATIC portable_vector_t<storage_t, D> SignMasks = _static_cast<portable_vector_t<storage_t, D> >(_static_cast<vector<uint64_t, D> >(vector<uint64_t, 4>(SignMask<Bits, D>, SignMask<Bits, D> << 1, SignMask<Bits, D> << 2, SignMask<Bits, D> << 3)));
        ComparisonOp comparison;
        // Obtain a vector of deinterleaved coordinates and flip their sign bits
        const portable_vector_t<storage_t, D> thisCoord = (InterleaveMasks & value) ^ SignMasks;
        // rhs already deinterleaved, just have to cast type and flip sign
        const portable_vector_t<storage_t, D> rhsCoord = _static_cast<portable_vector_t<storage_t, D> >(rhs) ^ SignMasks;

        return comparison(thisCoord, rhsCoord);
    }
};

template<bool Signed, uint16_t Bits, uint16_t D, typename storage_t, typename ComparisonOp>
struct BaseComparison<Signed, Bits, D, storage_t, false, ComparisonOp>
{
    template<typename I NBL_FUNC_REQUIRES(Comparable<Signed, Bits, storage_t, false, I>)
    NBL_CONSTEXPR_STATIC_INLINE_FUNC vector<bool, D> __call(NBL_CONST_REF_ARG(storage_t) value, NBL_CONST_REF_ARG(portable_vector_t<I, D>) rhs)
    {
        const vector<storage_t, D> interleaved = MortonEncoder<D, Bits, storage_t>::interleaveShift(rhs);
        BaseComparison<Signed, Bits, D, storage_t, true, ComparisonOp> baseComparison;
        return baseComparison(value, interleaved);
    }
};

template<bool Signed, uint16_t Bits, uint16_t D, typename storage_t, bool BitsAlreadySpread>
struct LessThan : BaseComparison<Signed, Bits, D, storage_t, BitsAlreadySpread, less<portable_vector_t<storage_t, D> > > {};

template<bool Signed, uint16_t Bits, uint16_t D, typename storage_t, bool BitsAlreadySpread>
struct LessEquals : BaseComparison<Signed, Bits, D, storage_t, BitsAlreadySpread, less_equal<portable_vector_t<storage_t, D> > > {};

template<bool Signed, uint16_t Bits, uint16_t D, typename storage_t, bool BitsAlreadySpread>
struct GreaterThan : BaseComparison<Signed, Bits, D, storage_t, BitsAlreadySpread, greater<portable_vector_t<storage_t, D> > > {};

template<bool Signed, uint16_t Bits, uint16_t D, typename storage_t, bool BitsAlreadySpread>
struct GreaterEquals : BaseComparison<Signed, Bits, D, storage_t, BitsAlreadySpread, greater_equal<portable_vector_t<storage_t, D> > > {};

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
    NBL_CONSTEXPR_STATIC_FUNC enable_if_t<is_integral_v<I> && is_scalar_v<I> && (is_signed_v<I> == Signed) && (8 * sizeof(I) >= Bits), this_t>
    create(NBL_CONST_REF_ARG(vector<I, D>) cartesian)
    {
        this_t retVal;
        retVal.value = impl::MortonEncoder<D, Bits, storage_t>::encode(cartesian);
        return retVal;
    }

    // CPP can also have an actual constructor
    #ifndef __HLSL_VERSION

    /**
    * @brief Creates a Morton code from a set of cartesian coordinates
    *
    * @param [in] cartesian Coordinates to encode
    */
    template<typename I NBL_FUNC_REQUIRES(8 * sizeof(I) >= Bits)
    explicit code(NBL_CONST_REF_ARG(vector<I, D>) cartesian)
    {
        *this = create(cartesian);
    }

    /**
    * @brief Decodes this Morton code back to a set of cartesian coordinates
    */
    template<typename I NBL_FUNC_REQUIRES(8 * sizeof(I) >= Bits)
    constexpr inline explicit operator vector<I, D>() const noexcept;

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
        this_t zero;
        zero.value = _static_cast<storage_t>(0);
        #ifndef __HLSL_VERSION
        return zero - *this;
        #else
        return zero - this;
        #endif
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

    template<bool BitsAlreadySpread, typename I 
    NBL_FUNC_REQUIRES(impl::Comparable<Signed, Bits, storage_t, BitsAlreadySpread, I>)
    NBL_CONSTEXPR_INLINE_FUNC vector<bool, D> equals(NBL_CONST_REF_ARG(vector<I, D>) rhs) NBL_CONST_MEMBER_FUNC
    {
        return impl::Equals<Signed, Bits, D, storage_t, BitsAlreadySpread>::__call(value, rhs);
    }  

    NBL_CONSTEXPR_INLINE_FUNC bool operator!=(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        return value != rhs.value;
    }

    template<bool BitsAlreadySpread, typename I
    NBL_FUNC_REQUIRES(impl::Comparable<Signed, Bits, storage_t, BitsAlreadySpread, I>)
    NBL_CONSTEXPR_INLINE_FUNC vector<bool, D> notEquals(NBL_CONST_REF_ARG(vector<I, D>) rhs) NBL_CONST_MEMBER_FUNC
    {
        return !equals<BitsAlreadySpread, I>(rhs);
    }

    template<bool BitsAlreadySpread, typename I
    NBL_FUNC_REQUIRES(impl::Comparable<Signed, Bits, storage_t, BitsAlreadySpread, I>)
    NBL_CONSTEXPR_INLINE_FUNC vector<bool, D> less(NBL_CONST_REF_ARG(vector<I, D>) rhs) NBL_CONST_MEMBER_FUNC
    {
        return impl::LessThan<Signed, Bits, D, storage_t, BitsAlreadySpread>::__call(value, rhs);
    }

    template<bool BitsAlreadySpread, typename I
    NBL_FUNC_REQUIRES(impl::Comparable<Signed, Bits, storage_t, BitsAlreadySpread, I>)
    NBL_CONSTEXPR_INLINE_FUNC vector<bool, D> lessEquals(NBL_CONST_REF_ARG(vector<I, D>) rhs) NBL_CONST_MEMBER_FUNC
    {
        return impl::LessEquals<Signed, Bits, D, storage_t, BitsAlreadySpread>::__call(value, rhs);
    }

    template<bool BitsAlreadySpread, typename I
    NBL_FUNC_REQUIRES(impl::Comparable<Signed, Bits, storage_t, BitsAlreadySpread, I>)
    NBL_CONSTEXPR_INLINE_FUNC vector<bool, D> greater(NBL_CONST_REF_ARG(vector<I, D>) rhs) NBL_CONST_MEMBER_FUNC
    {
        return impl::GreaterThan<Signed, Bits, D, storage_t, BitsAlreadySpread>::__call(value, rhs);
    }

    template<bool BitsAlreadySpread, typename I
    NBL_FUNC_REQUIRES(impl::Comparable<Signed, Bits, storage_t, BitsAlreadySpread, I>)
    NBL_CONSTEXPR_INLINE_FUNC vector<bool, D> greaterEquals(NBL_CONST_REF_ARG(vector<I, D>) rhs) NBL_CONST_MEMBER_FUNC
    {
        return impl::GreaterEquals<Signed, Bits, D, storage_t, BitsAlreadySpread>::__call(value, rhs);
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
template<typename I, uint16_t Bits, uint16_t D, typename _uint64_t> NBL_PARTIAL_REQ_TOP(concepts::IntegralScalar<I> && 8 * sizeof(I) >= Bits)
struct static_cast_helper<vector<I, D>, morton::code<is_signed_v<I>, Bits, D, _uint64_t> NBL_PARTIAL_REQ_BOT(concepts::IntegralScalar<I> && 8 * sizeof(I) >= Bits) >
{
    NBL_CONSTEXPR_STATIC_INLINE_FUNC vector<I, D> cast(NBL_CONST_REF_ARG(morton::code<is_signed_v<I>, Bits, D, _uint64_t>) val)
    {
        using storage_t = typename morton::code<is_signed_v<I>, Bits, D, _uint64_t>::storage_t;
        return morton::impl::MortonDecoder<D, Bits, storage_t>::decode(val.value);
    }
};

} // namespace impl

#ifndef __HLSL_VERSION

template <bool Signed, uint16_t Bits, uint16_t D, typename _uint64_t NBL_PRIMARY_REQUIRES(morton::impl::Dimension<D>&& D* Bits <= 64)
template <typename I NBL_FUNC_REQUIRES(8 * sizeof(I) >= Bits)
constexpr inline morton::code<Signed, Bits, D, _uint64_t>::operator vector<I, D>() const noexcept
{
    return _static_cast<vector<I, D>, morton::code<is_signed_v<I>, Bits, D>>(*this);
}

#endif

} //namespace hlsl
} //namespace nbl

#endif