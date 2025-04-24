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

template<uint16_t Dim, uint16_t Bits, uint16_t Stage, typename T = uint64_t>
NBL_CONSTEXPR T coding_mask_v = _static_cast<T>(coding_mask<Dim, Bits, Stage>::value);

template<typename storage_t, uint16_t Dim, uint16_t Bits>
NBL_CONSTEXPR portable_vector_t<storage_t, Dim> InterleaveMasks = _static_cast<portable_vector_t<storage_t, Dim> >(
                                                                  truncate<vector<uint64_t, Dim> >(
                                                                  vector<uint64_t, 4>(coding_mask_v<Dim, Bits, 0>, 
                                                                                     coding_mask_v<Dim, Bits, 0> << 1, 
                                                                                     coding_mask_v<Dim, Bits, 0> << 2, 
                                                                                     coding_mask_v<Dim, Bits, 0> << 3)));

template<uint16_t Dim, uint16_t Bits>
struct sign_mask : integral_constant<uint64_t, uint64_t(1) << ((Bits - 1) * Dim)> {};

template<uint16_t Dim, uint16_t Bits, typename T = uint64_t>
NBL_CONSTEXPR T sign_mask_v = _static_cast<T>(sign_mask<Dim, Bits>::value);

template<typename storage_t, uint16_t Dim, uint16_t Bits>
NBL_CONSTEXPR portable_vector_t<storage_t, Dim> SignMasks = _static_cast<portable_vector_t<storage_t, Dim> >(
                                                            truncate<vector<uint64_t, Dim> >(
                                                            vector<uint64_t, 4>(sign_mask_v<Dim, Bits>, 
                                                                                sign_mask_v<Dim, Bits> << 1, 
                                                                                sign_mask_v<Dim, Bits> << 2, 
                                                                                sign_mask_v<Dim, Bits> << 3)));

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

NBL_HLSL_MORTON_SPECIALIZE_FIRST_CODING_MASK(2, 0x5555555555555555ull)        // Groups bits by 1  on, 1  off
NBL_HLSL_MORTON_SPECIALIZE_CODING_MASK(2, 1, 0x3333333333333333ull) // Groups bits by 2  on, 2  off
NBL_HLSL_MORTON_SPECIALIZE_CODING_MASK(2, 2, 0x0F0F0F0F0F0F0F0Full) // Groups bits by 4  on, 4  off
NBL_HLSL_MORTON_SPECIALIZE_CODING_MASK(2, 3, 0x00FF00FF00FF00FFull) // Groups bits by 8  on, 8  off
NBL_HLSL_MORTON_SPECIALIZE_CODING_MASK(2, 4, 0x0000FFFF0000FFFFull) // Groups bits by 16 on, 16 off

NBL_HLSL_MORTON_SPECIALIZE_FIRST_CODING_MASK(3, 0x9249249249249249ull)        // Groups bits by 1  on, 2  off
NBL_HLSL_MORTON_SPECIALIZE_CODING_MASK(3, 1, 0x30C30C30C30C30C3ull) // Groups bits by 2  on, 4  off
NBL_HLSL_MORTON_SPECIALIZE_CODING_MASK(3, 2, 0xF00F00F00F00F00Full) // Groups bits by 4  on, 8  off
NBL_HLSL_MORTON_SPECIALIZE_CODING_MASK(3, 3, 0x00FF0000FF0000FFull) // Groups bits by 8  on, 16 off
NBL_HLSL_MORTON_SPECIALIZE_CODING_MASK(3, 4, 0xFFFF00000000FFFFull) // Groups bits by 16 on, 32 off

NBL_HLSL_MORTON_SPECIALIZE_FIRST_CODING_MASK(4, 0x1111111111111111ull)        // Groups bits by 1  on, 3  off
NBL_HLSL_MORTON_SPECIALIZE_CODING_MASK(4, 1, 0x0303030303030303ull) // Groups bits by 2  on, 6  off
NBL_HLSL_MORTON_SPECIALIZE_CODING_MASK(4, 2, 0x000F000F000F000Full) // Groups bits by 4  on, 12 off
NBL_HLSL_MORTON_SPECIALIZE_CODING_MASK(4, 3, 0x000000FF000000FFull) // Groups bits by 8  on, 24 off
NBL_HLSL_MORTON_SPECIALIZE_CODING_MASK(4, 4, 0x000000000000FFFFull) // Groups bits by 16 on, 48 off (unused but here for completion + likely keeps compiler from complaining)

NBL_HLSL_MORTON_SPECIALIZE_LAST_CODING_MASKS

#undef NBL_HLSL_MORTON_SPECIALIZE_LAST_CODING_MASK
#undef NBL_HLSL_MORTON_SPECIALIZE_CODING_MASK
#undef NBL_HLSL_MORTON_SPECIALIZE_FIRST_CODING_MASK

// ----------------------------------------------------------------- MORTON TRANSCODER ---------------------------------------------------
template<uint16_t Dim, uint16_t Bits, typename encode_t NBL_PRIMARY_REQUIRES(Dimension<Dim> && Dim * Bits <= 64 && 8 * sizeof(encode_t) == mpl::max_v<uint64_t, mpl::round_up_to_pot_v<Dim * Bits>, uint64_t(16)>)
struct Transcoder
{
    template<typename decode_t = conditional_t<(Bits > 16), vector<uint32_t, Dim>, vector<uint16_t, Dim> >
    NBL_FUNC_REQUIRES(concepts::IntVector<decode_t> && 8 * sizeof(typename vector_traits<decode_t>::scalar_type) >= Bits)
    /**
    * @brief Interleaves each coordinate with `Dim - 1` zeros inbetween each bit, and left-shifts each by their coordinate index
    *
    * @param [in] decodedValue Cartesian coordinates to interleave and shift
    */
    NBL_CONSTEXPR_STATIC_FUNC portable_vector_t<encode_t, Dim> interleaveShift(NBL_CONST_REF_ARG(decode_t) decodedValue)
    {
        left_shift_operator<portable_vector_t<encode_t, Dim> > leftShift;
        portable_vector_t<encode_t, Dim> interleaved = _static_cast<portable_vector_t<encode_t, Dim> >(decodedValue) & coding_mask_v<Dim, Bits, CodingStages, encode_t>;

        #define ENCODE_LOOP_ITERATION(I) NBL_IF_CONSTEXPR(Bits > (uint16_t(1) << I))\
        {\
            interleaved = interleaved | leftShift(interleaved, (uint16_t(1) << I) * (Dim - 1));\
            interleaved = interleaved & coding_mask_v<Dim, Bits, I, encode_t>;\
        }
        ENCODE_LOOP_ITERATION(4)
        ENCODE_LOOP_ITERATION(3)
        ENCODE_LOOP_ITERATION(2)
        ENCODE_LOOP_ITERATION(1)
        ENCODE_LOOP_ITERATION(0)

        #undef ENCODE_LOOP_ITERATION

        // After interleaving, shift each coordinate left by their index
        return leftShift(interleaved, truncate<vector<uint16_t, Dim> >(vector<uint16_t, 4>(0, 1, 2, 3)));
    }

    template<typename decode_t = conditional_t<(Bits > 16), vector<uint32_t, Dim>, vector<uint16_t, Dim> >
    NBL_FUNC_REQUIRES(concepts::IntVector<decode_t> && 8 * sizeof(typename vector_traits<decode_t>::scalar_type) >= Bits)
    /**
    * @brief Encodes a vector of cartesian coordinates as a Morton code
    *
    * @param [in] decodedValue Cartesian coordinates to encode
    */
    NBL_CONSTEXPR_STATIC_FUNC encode_t encode(NBL_CONST_REF_ARG(decode_t) decodedValue)
    {
        const portable_vector_t<encode_t, Dim> interleaveShifted = interleaveShift<decode_t>(decodedValue);

        array_get<portable_vector_t<encode_t, Dim>, encode_t> getter;
        encode_t encoded = getter(interleaveShifted, 0);

        [[unroll]]
        for (uint16_t i = 1; i < Dim; i++)
            encoded = encoded | getter(interleaveShifted, i);

        return encoded;
    }

    template<typename decode_t = conditional_t<(Bits > 16), vector<uint32_t, Dim>, vector<uint16_t, Dim> >
    NBL_FUNC_REQUIRES(concepts::IntVector<decode_t> && 8 * sizeof(typename vector_traits<decode_t>::scalar_type) >= Bits)
    /**
    * @brief Decodes a Morton code back to a vector of cartesian coordinates
    *
    * @param [in] encodedValue Representation of a Morton code (binary code, not the morton class defined below)
    */
    NBL_CONSTEXPR_STATIC_FUNC decode_t decode(NBL_CONST_REF_ARG(encode_t) encodedValue)
    {
        arithmetic_right_shift_operator<encode_t> encodedRightShift;
        portable_vector_t<encode_t, Dim> decoded;
        array_set<portable_vector_t<encode_t, Dim>, encode_t> setter;
        // Write initial values into decoded
        [[unroll]]
        for (uint16_t i = 0; i < Dim; i++)
            setter(decoded, i, encodedRightShift(encodedValue, i));

        arithmetic_right_shift_operator<portable_vector_t<encode_t, Dim> > rightShift;

        #define DECODE_LOOP_ITERATION(I) NBL_IF_CONSTEXPR(Bits > (uint16_t(1) << I))\
        {\
            decoded = decoded & coding_mask_v<Dim, Bits, I, encode_t>;\
            decoded = decoded | rightShift(decoded, (uint16_t(1) << I) * (Dim - 1));\
        }

        DECODE_LOOP_ITERATION(0)
        DECODE_LOOP_ITERATION(1)
        DECODE_LOOP_ITERATION(2)
        DECODE_LOOP_ITERATION(3)
        DECODE_LOOP_ITERATION(4)

        #undef DECODE_LOOP_ITERATION

        // If `Bits` is greater than half the bitwidth of the decode type, then we can avoid `&`ing against the last mask since duplicated MSB get truncated
        NBL_IF_CONSTEXPR(Bits > 4 * sizeof(typename vector_traits<decode_t>::scalar_type))
            return _static_cast<decode_t>(decoded);
        else
            return _static_cast<decode_t>(decoded & coding_mask_v<Dim, Bits, CodingStages, encode_t>);
    }
};

// ---------------------------------------------------- COMPARISON OPERATORS ---------------------------------------------------------------
// Here because no partial specialization of methods
// `BitsAlreadySpread` assumes both pre-interleaved and pre-shifted

template<bool Signed, uint16_t Bits, typename storage_t, bool BitsAlreadySpread, typename I>
NBL_BOOL_CONCEPT Comparable = concepts::IntegralLikeScalar<I> && is_signed_v<I> == Signed && ((BitsAlreadySpread && sizeof(I) == sizeof(storage_t)) || (!BitsAlreadySpread && 8 * sizeof(I) == mpl::max_v<uint64_t, mpl::round_up_to_pot_v<Bits>, uint64_t(16)>));

template<bool Signed, uint16_t Bits, uint16_t D, typename storage_t, bool BitsAlreadySpread>
struct Equals;

template<bool Signed, uint16_t Bits, uint16_t D, typename storage_t>
struct Equals<Signed, Bits, D, storage_t, true>
{
    template<typename I NBL_FUNC_REQUIRES(Comparable<Signed, Bits, storage_t, true, I>)
    NBL_CONSTEXPR_STATIC_FUNC vector<bool, D> __call(NBL_CONST_REF_ARG(storage_t) value, NBL_CONST_REF_ARG(portable_vector_t<I, D>) rhs)
    {
        const portable_vector_t<storage_t, D> zeros = _static_cast<portable_vector_t<storage_t, D> >(truncate<vector<uint64_t, D> >(vector<uint64_t, 4>(0,0,0,0)));
        
        const portable_vector_t<storage_t, D> rhsCasted = _static_cast<portable_vector_t<storage_t, D> >(rhs);
        const portable_vector_t<storage_t, D> xored = rhsCasted ^ value;
        equal_to<portable_vector_t<storage_t, D> > equal;
        return equal(xored, zeros);
    }
};

template<bool Signed, uint16_t Bits, uint16_t D, typename storage_t>
struct Equals<Signed, Bits, D, storage_t, false>
{
    template<typename I NBL_FUNC_REQUIRES(Comparable<Signed, Bits, storage_t, false, I>)
    NBL_CONSTEXPR_STATIC_FUNC vector<bool, D> __call(NBL_CONST_REF_ARG(storage_t) value, NBL_CONST_REF_ARG(vector<I, D>) rhs)
    {
        using right_sign_t = conditional_t<Signed, make_signed_t<storage_t>, make_unsigned_t<storage_t> >;
        const portable_vector_t<right_sign_t, D> interleaved = _static_cast<portable_vector_t<right_sign_t, D> >(Transcoder<D, Bits, storage_t>::interleaveShift(rhs));
        return Equals<Signed, Bits, D, storage_t, true>::template __call<right_sign_t>(value, interleaved);
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
    NBL_CONSTEXPR_STATIC_FUNC vector<bool, D> __call(NBL_CONST_REF_ARG(storage_t) value, NBL_CONST_REF_ARG(portable_vector_t<I, D>) rhs)
    {
        ComparisonOp comparison;
        NBL_IF_CONSTEXPR(Signed)
        {
            // Obtain a vector of deinterleaved coordinates and flip their sign bits
            portable_vector_t<storage_t, D> thisCoord = (InterleaveMasks<storage_t, D, Bits> & value) ^ SignMasks<storage_t, D, Bits>;
            // rhs already deinterleaved, just have to cast type and flip sign
            const portable_vector_t<storage_t, D> rhsCoord = _static_cast<portable_vector_t<storage_t, D> >(rhs) ^ SignMasks<storage_t, D, Bits>;

            return comparison(thisCoord, rhsCoord);
        }
        else 
        {
            // Obtain a vector of deinterleaved coordinates
            portable_vector_t<storage_t, D> thisCoord = InterleaveMasks<storage_t, D, Bits> & value;
            // rhs already deinterleaved, just have to cast type
            const portable_vector_t<storage_t, D> rhsCoord = _static_cast<portable_vector_t<storage_t, D> >(rhs);

            return comparison(thisCoord, rhsCoord);
        }
        
    }
};

template<bool Signed, uint16_t Bits, uint16_t D, typename storage_t, typename ComparisonOp>
struct BaseComparison<Signed, Bits, D, storage_t, false, ComparisonOp>
{
    template<typename I NBL_FUNC_REQUIRES(Comparable<Signed, Bits, storage_t, false, I>)
    NBL_CONSTEXPR_STATIC_FUNC vector<bool, D> __call(NBL_CONST_REF_ARG(storage_t) value, NBL_CONST_REF_ARG(vector<I, D>) rhs)
    {
        using right_sign_t = conditional_t<Signed, make_signed_t<storage_t>, make_unsigned_t<storage_t> >;
        const portable_vector_t<right_sign_t, D> interleaved = _static_cast<portable_vector_t<right_sign_t, D> >(Transcoder<D, Bits, storage_t>::interleaveShift(rhs));
        return BaseComparison<Signed, Bits, D, storage_t, true, ComparisonOp>::template __call<right_sign_t>(value, interleaved);
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
        retVal.value = impl::Transcoder<D, Bits, storage_t>::encode(cartesian);
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
    inline explicit code(NBL_CONST_REF_ARG(vector<I, D>) cartesian)
    {
        *this = create(cartesian);
    }

    /**
    * @brief Decodes this Morton code back to a set of cartesian coordinates
    */
    template<typename I NBL_FUNC_REQUIRES(8 * sizeof(I) >= Bits && is_signed_v<I> == Signed)
    constexpr explicit operator vector<I, D>() const noexcept;

    #endif

    // ------------------------------------------------------- BITWISE OPERATORS -------------------------------------------------

    NBL_CONSTEXPR_FUNC this_t operator&(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        this_t retVal;
        retVal.value = value & rhs.value;
        return retVal;
    }

    NBL_CONSTEXPR_FUNC this_t operator|(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        this_t retVal;
        retVal.value = value | rhs.value;
        return retVal;
    }

    NBL_CONSTEXPR_FUNC this_t operator^(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        this_t retVal;
        retVal.value = value ^ rhs.value;
        return retVal;
    }

    NBL_CONSTEXPR_FUNC this_t operator~() NBL_CONST_MEMBER_FUNC
    {
        this_t retVal;
        retVal.value = ~value;
        return retVal;
    }

    // Only valid in CPP
    #ifndef __HLSL_VERSION

    constexpr this_t operator<<(uint16_t bits) const;

    constexpr this_t operator>>(uint16_t bits) const;

    #endif

    // ------------------------------------------------------- UNARY ARITHMETIC OPERATORS -------------------------------------------------

    NBL_CONSTEXPR_FUNC this_signed_t operator-() NBL_CONST_MEMBER_FUNC
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

    // put 1 bits everywhere in the bits the current axis is not using
    // then extract just the axis bits for the right hand coordinate
    // carry-1 will propagate the bits across the already set bits
    // then clear out the bits not belonging to current axis
    // Note: Its possible to clear on `this` and fill on `rhs` but that will
    // disable optimizations, we expect the compiler to optimize a lot if the
    // value of `rhs` is known at compile time, e.g. `static_cast<Morton<N>>(glm::ivec3(1,0,0))`
    NBL_CONSTEXPR_FUNC this_t operator+(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        bit_not<portable_vector_t<storage_t, D> > bitnot;
        // For each coordinate, leave its bits intact and turn every other bit ON
        const portable_vector_t<storage_t, D> counterMaskedValue = bitnot(impl::InterleaveMasks<storage_t, D, Bits>) | value;
        // For each coordinate in rhs, leave its bits intact and turn every other bit OFF
        const portable_vector_t<storage_t, D> maskedRhsValue = impl::InterleaveMasks<storage_t, D, Bits> & rhs.value;
        // Add these coordinate-wise, then turn all bits not belonging to the current coordinate OFF
        const portable_vector_t<storage_t, D> interleaveShiftedResult = (counterMaskedValue + maskedRhsValue) & impl::InterleaveMasks<storage_t, D, Bits>;
        // Re-encode the result
        array_get<portable_vector_t<storage_t, D>, storage_t> getter;
        this_t retVal;
        retVal.value = getter(interleaveShiftedResult, 0);
        [[unroll]]
        for (uint16_t i = 1; i < D; i++)
            retVal.value = retVal.value | getter(interleaveShiftedResult, i);

        return retVal;
    }

    // This is the dual trick of the one used for addition: set all other bits to 0 so borrows propagate
    NBL_CONSTEXPR_FUNC this_t operator-(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        // For each coordinate, leave its bits intact and turn every other bit OFF
        const portable_vector_t<storage_t, D> maskedValue = impl::InterleaveMasks<storage_t, D, Bits> & value;
        // Do the same for each coordinate in rhs
        const portable_vector_t<storage_t, D> maskedRhsValue = impl::InterleaveMasks<storage_t, D, Bits> & rhs.value;
        // Subtract these coordinate-wise, then turn all bits not belonging to the current coordinate OFF
        const portable_vector_t<storage_t, D> interleaveShiftedResult = (maskedValue - maskedRhsValue) & impl::InterleaveMasks<storage_t, D, Bits>;
        // Re-encode the result
        array_get<portable_vector_t<storage_t, D>, storage_t> getter;
        this_t retVal;
        retVal.value = getter(interleaveShiftedResult, 0);
        [[unroll]]
        for (uint16_t i = 1; i < D; i++)
            retVal.value = retVal.value | getter(interleaveShiftedResult, i);

        return retVal;
    }

    // ------------------------------------------------------- COMPARISON OPERATORS -------------------------------------------------

    NBL_CONSTEXPR_FUNC bool operator==(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        return value == rhs.value;
    }

    template<bool BitsAlreadySpread, typename I 
    NBL_FUNC_REQUIRES(impl::Comparable<Signed, Bits, storage_t, BitsAlreadySpread, I>)
    NBL_CONSTEXPR_FUNC vector<bool, D> equal(NBL_CONST_REF_ARG(vector<I, D>) rhs) NBL_CONST_MEMBER_FUNC
    {
        return impl::Equals<Signed, Bits, D, storage_t, BitsAlreadySpread>::template __call<I>(value, rhs);
    }  

    NBL_CONSTEXPR_FUNC bool operator!=(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        return value != rhs.value;
    }

    template<bool BitsAlreadySpread, typename I
    NBL_FUNC_REQUIRES(impl::Comparable<Signed, Bits, storage_t, BitsAlreadySpread, I>)
    NBL_CONSTEXPR_FUNC vector<bool, D> notEqual(NBL_CONST_REF_ARG(vector<I, D>) rhs) NBL_CONST_MEMBER_FUNC
    {
        return !equal<BitsAlreadySpread, I>(rhs);
    }

    template<bool BitsAlreadySpread, typename I
    NBL_FUNC_REQUIRES(impl::Comparable<Signed, Bits, storage_t, BitsAlreadySpread, I>)
    NBL_CONSTEXPR_FUNC vector<bool, D> lessThan(NBL_CONST_REF_ARG(vector<I, D>) rhs) NBL_CONST_MEMBER_FUNC
    {
        return impl::LessThan<Signed, Bits, D, storage_t, BitsAlreadySpread>::template __call<I>(value, rhs);
    }

    template<bool BitsAlreadySpread, typename I
    NBL_FUNC_REQUIRES(impl::Comparable<Signed, Bits, storage_t, BitsAlreadySpread, I>)
    NBL_CONSTEXPR_FUNC vector<bool, D> lessThanEquals(NBL_CONST_REF_ARG(vector<I, D>) rhs) NBL_CONST_MEMBER_FUNC
    {
        return impl::LessEquals<Signed, Bits, D, storage_t, BitsAlreadySpread>::template __call<I>(value, rhs);
    }

    template<bool BitsAlreadySpread, typename I
    NBL_FUNC_REQUIRES(impl::Comparable<Signed, Bits, storage_t, BitsAlreadySpread, I>)
    NBL_CONSTEXPR_FUNC vector<bool, D> greaterThan(NBL_CONST_REF_ARG(vector<I, D>) rhs) NBL_CONST_MEMBER_FUNC
    {
        return impl::GreaterThan<Signed, Bits, D, storage_t, BitsAlreadySpread>::template __call<I>(value, rhs);
    }

    template<bool BitsAlreadySpread, typename I
    NBL_FUNC_REQUIRES(impl::Comparable<Signed, Bits, storage_t, BitsAlreadySpread, I>)
    NBL_CONSTEXPR_FUNC vector<bool, D> greaterThanEquals(NBL_CONST_REF_ARG(vector<I, D>) rhs) NBL_CONST_MEMBER_FUNC
    {
        return impl::GreaterEquals<Signed, Bits, D, storage_t, BitsAlreadySpread>::template __call<I>(value, rhs);
    }

};

} //namespace morton

// Specialize the `static_cast_helper`
namespace impl
{

// I must be of same signedness as the morton code, and be wide enough to hold each component
template<typename I, uint16_t Bits, uint16_t D, typename _uint64_t> NBL_PARTIAL_REQ_TOP(concepts::IntegralScalar<I> && 8 * sizeof(I) >= Bits)
struct static_cast_helper<vector<I, D>, morton::code<is_signed_v<I>, Bits, D, _uint64_t> NBL_PARTIAL_REQ_BOT(concepts::IntegralScalar<I> && 8 * sizeof(I) >= Bits) >
{
    NBL_CONSTEXPR_STATIC_FUNC vector<I, D> cast(NBL_CONST_REF_ARG(morton::code<is_signed_v<I>, Bits, D, _uint64_t>) val)
    {
        using storage_t = typename morton::code<is_signed_v<I>, Bits, D, _uint64_t>::storage_t;
        return morton::impl::Transcoder<D, Bits, storage_t>::decode(val.value);
    }
};

} // namespace impl

template<bool Signed, uint16_t Bits, uint16_t D, typename _uint64_t>
struct left_shift_operator<morton::code<Signed, Bits, D, _uint64_t> >
{
    using type_t = morton::code<Signed, Bits, D, _uint64_t>;
    using storage_t = typename type_t::storage_t;

    NBL_CONSTEXPR_FUNC type_t operator()(NBL_CONST_REF_ARG(type_t) operand, uint16_t bits)
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

    NBL_CONSTEXPR_FUNC type_t operator()(NBL_CONST_REF_ARG(type_t) operand, uint16_t bits)
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

    NBL_CONSTEXPR_FUNC type_t operator()(NBL_CONST_REF_ARG(type_t) operand, uint16_t bits)
    {
        vector<scalar_t, D> cartesian = _static_cast<vector<scalar_t, D> >(operand);
        cartesian >>= scalar_t(bits);
        return type_t::create(cartesian);
    }
};

#ifndef __HLSL_VERSION

template<bool Signed, uint16_t Bits, uint16_t D, typename _uint64_t NBL_FUNC_REQUIRES(morton::impl::Dimension<D>&& D* Bits <= 64)
constexpr morton::code<Signed, Bits, D, _uint64_t> morton::code<Signed, Bits, D, _uint64_t>::operator<<(uint16_t bits) const
{
    left_shift_operator<morton::code<Signed, Bits, D, _uint64_t>> leftShift;
    return leftShift(*this, bits);
}

template<bool Signed, uint16_t Bits, uint16_t D, typename _uint64_t NBL_FUNC_REQUIRES(morton::impl::Dimension<D>&& D* Bits <= 64)
constexpr morton::code<Signed, Bits, D, _uint64_t> morton::code<Signed, Bits, D, _uint64_t>::operator>>(uint16_t bits) const
{
    arithmetic_right_shift_operator<morton::code<Signed, Bits, D, _uint64_t>> rightShift;
    return rightShift(*this, bits);
}

template <bool Signed, uint16_t Bits, uint16_t D, typename _uint64_t NBL_PRIMARY_REQUIRES(morton::impl::Dimension<D>&& D* Bits <= 64)
template <typename I NBL_FUNC_REQUIRES(8 * sizeof(I) >= Bits && is_signed_v<I> == Signed)
constexpr morton::code<Signed, Bits, D, _uint64_t>::operator vector<I, D>() const noexcept
{
    return _static_cast<vector<I, D>, morton::code<Signed, Bits, D>>(*this);
}

#endif

} //namespace hlsl
} //namespace nbl

#endif