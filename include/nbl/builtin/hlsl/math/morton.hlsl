#ifndef _NBL_BUILTIN_HLSL_MATH_MORTON_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_MORTON_INCLUDED_

#include "nbl/builtin/hlsl/concepts/core.hlsl"
#include "nbl/builtin/hlsl/bit.hlsl"
#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/functional.hlsl"

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

template<typename T, uint16_t Dim, uint16_t Bits = 8 * sizeof(T) / Dim>
struct decode_mask;

template<typename T, uint16_t Dim>
struct decode_mask<T, Dim, 0> : integral_constant<T, 1> {};

template<typename T, uint16_t Dim, uint16_t Bits>
struct decode_mask : integral_constant<T, (decode_mask<T, Dim, Bits - 1>::value << Dim) | T(1)> {};

template<typename T, uint16_t Dim, uint16_t Bits = 8 * sizeof(T) / Dim>
NBL_CONSTEXPR T decode_mask_v = decode_mask<T, Dim, Bits>::value;

} //namespace impl

// Up to D = 4 supported
#define NBL_HLSL_MORTON_MASKS(U, D) _static_cast<vector<U,D> > (vector< U , 4 >(impl::decode_mask_v< U , D >,\
                                                    impl::decode_mask_v< U , D > << U (1),\
                                                    impl::decode_mask_v< U , D > << U (2),\
                                                    impl::decode_mask_v< U , D > << U (3)\
                                                   ))

// Making this even slightly less ugly is blocked by https://github.com/microsoft/DirectXShaderCompiler/issues/7006
// In particular, `Masks` should be a `const static` member field instead of appearing in every method using it
template<typename I, uint16_t D NBL_PRIMARY_REQUIRES(concepts::IntegralScalar<I> && impl::MortonDimension<D>)
struct code
{
    using this_t = code<I, D>;
    using U = make_unsigned_t<I>;
    NBL_CONSTEXPR_STATIC U BitWidth = U(8 * sizeof(U));

    // ---------------------------------------------------- CONSTRUCTORS ---------------------------------------------------------------

    #ifndef __HLSL_VERSION

    code() = default;

    // To immediately get compound operators and functional structs in CPP side
    code(const U _value) : value(_value) {}

    #endif

    /**
    * @brief Creates a Morton code from a set of cartesian coordinates
    *
    * @param [in] cartesian Coordinates to encode
    */
    NBL_CONSTEXPR_STATIC_FUNC this_t create(NBL_CONST_REF_ARG(vector<I, D>) cartesian)
    {
        const vector<U, D> unsignedCartesian = bit_cast<vector<U, D>, vector<I, D> >(cartesian);
        this_t retVal = { U(0) };
        
        [[unroll]]
        for (U coord = 0; coord < U(D); coord++)
        {
            [[unroll]]
            // Control can be simplified by running a bound on just coordBit based on `BitWidth` and `coord`, but I feel this is clearer
            for (U valBitIdx = coord, coordBit = U(1), shift = coord; valBitIdx < BitWidth; valBitIdx += U(D), coordBit <<= 1, shift += U(D) - 1)
            {
                retVal.value |= (unsignedCartesian[coord] & coordBit) << shift;
            }
        }
        
        return retVal;
    }

    // CPP can also have a constructor
    #ifndef __HLSL_VERSION

    /**
    * @brief Creates a Morton code from a set of cartesian coordinates
    *
    * @param [in] cartesian Coordinates to encode
    */
    code(NBL_CONST_REF_ARG(vector<I, D>) cartesian)
    {
        *this = create(cartesian);
    }

    /**
    * @brief Decodes this Morton code back to a set of cartesian coordinates
    */
    explicit operator vector<I, D>() const noexcept
    {
        // Definition below, we override `impl::static_cast_helper` to have this conversion in both CPP/HLSL
        return _static_cast<vector<I, D>, this_t>(*this);
    }

    #endif

    // --------------------------------------------------------- AUX METHODS -------------------------------------------------------------------

    /**
    * @brief Extracts a single coordinate
    * 
    * @param [in] coord The coordinate to extract
    */
    NBL_CONSTEXPR_INLINE_FUNC I getCoordinate(uint16_t coord) NBL_CONST_MEMBER_FUNC
    {
        // Converting back has an issue with bit-width: when encoding (if template parameter `I` is signed) we cut off the highest bits 
        // that actually indicated sign. Therefore what we do is set the highest bits instead of the lowest then do an arithmetic right shift 
        // at the end to preserve sign. 
        // To this end, we first notice that the coordinate of index `coord` gets 
        // `bits(coord) = ceil((BitWidth - coord)/D)` bits when encoded (so the first dimensions get more bits than the last ones if `D` does not 
        // divide `BitWidth perfectly`).
        // Then instead of unpacking all the bits for that coordinate as the lowest bits, we unpack them as the highest ones
        // by shifting everything `BitWidth - bits(coord)` bits to the left, then at the end do a final *arithmetic* bitshift right by the same amount.
        
        const U bitsCoord = BitWidth / U(D) + ((coord < BitWidth % D) ? U(1) : U(0)); // <- this computes the ceil
        U coordVal = U(0);
        // Control can be simplified by running a bound on just coordBit based on `BitWidth` and `coord`, but I feel this is clearer
        [[unroll]]
        for (U valBitIdx = U(coord), coordBit = U(1) << U(coord), shift = U(coord); valBitIdx < BitWidth; valBitIdx += U(D), coordBit <<= U(D), shift += U(D) - 1)
        {
            coordVal |= (value & coordBit) << (BitWidth - bitsCoord - shift);
        }
        return bit_cast<I, U>(coordVal) >> (BitWidth - bitsCoord);
    }

    /**
    * @brief Returns an element of type U by extracting only the highest bit (of the bits used to encode `coord`)
    *
    * @param [in] coord The coordinate whose highest bit we want to extract.
    */
    NBL_CONSTEXPR_INLINE_FUNC U extractHighestBit(uint16_t coord) NBL_CONST_MEMBER_FUNC
    {
        // Like above, if the number encoded in `coord` gets `bits(coord) = ceil((BitWidth - coord)/D)` bits for representation, then the highest index of these
        // bits is `bits(coord) - 1`
        const U coordHighestBitIdx = BitWidth / U(D) - ((U(coord) < BitWidth % U(D)) ? U(0) : U(1));
        // This is the index of that bit as an index in the encoded value
        const U shift = coordHighestBitIdx * U(D) + U(coord);
        return value & (U(1) << shift);
    }

    // ------------------------------------------------------- BITWISE OPERATORS -------------------------------------------------
    
    NBL_CONSTEXPR_INLINE_FUNC this_t operator&(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        this_t retVal = { value & rhs.value };
        return retVal;
    }

    NBL_CONSTEXPR_INLINE_FUNC this_t operator|(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        this_t retVal = { value | rhs.value };
        return retVal;
    }

    NBL_CONSTEXPR_INLINE_FUNC this_t operator^(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        this_t retVal = { value ^ rhs.value };
        return retVal;
    }

    NBL_CONSTEXPR_INLINE_FUNC this_t operator~() NBL_CONST_MEMBER_FUNC
    {
        this_t retVal = { ~value };
        return retVal;
    }

    // Only valid in CPP
    #ifndef __HLSL_VERSION

    NBL_CONSTEXPR_INLINE_FUNC this_t operator<<(uint16_t bits) NBL_CONST_MEMBER_FUNC
    {
        this_t retVal = { value << U(bits) };
        return retVal;
    }

    NBL_CONSTEXPR_INLINE_FUNC this_t operator>>(uint16_t bits) NBL_CONST_MEMBER_FUNC
    {
        this_t retVal = { value >> U(bits) };
        return retVal;
    }

    #endif

    // ------------------------------------------------------- UNARY ARITHMETIC OPERATORS -------------------------------------------------

    NBL_CONSTEXPR_INLINE_FUNC this_t operator-() NBL_CONST_MEMBER_FUNC
    {
        // allOnes encodes a cartesian coordinate with all values set to 1
        const static this_t allOnes = { (U(1) << D) - U(1) };
        // Using 2's complement property that arithmetic negation can be obtained by bitwise negation then adding 1
        return operator~() + allOnes;
    }

    // ------------------------------------------------------- BINARY ARITHMETIC OPERATORS -------------------------------------------------

    // CHANGED FOR DEBUG: REMEMBER TO CHANGE BACK

    NBL_CONSTEXPR_INLINE_FUNC this_t operator+(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        NBL_CONSTEXPR_STATIC vector<U, D> Masks = NBL_HLSL_MORTON_MASKS(U, D);
        this_t retVal = { U(0) };
        [[unroll]]
        for (uint16_t coord = 0; coord < D; coord++)
        {
            // put 1 bits everywhere in the bits the current axis is not using
            // then extract just the axis bits for the right hand coordinate
            // carry-1 will propagate the bits across the already set bits
            // then clear out the bits not belonging to current axis
            // Note: Its possible to clear on `this` and fill on `rhs` but that will
            // disable optimizations, we expect the compiler to optimize a lot if the
            // value of `rhs` is known at compile time, e.g. `static_cast<Morton<N>>(glm::ivec3(1,0,0))`
            retVal.value |= ((value | (~Masks[coord])) + (rhs.value & Masks[coord])) & Masks[coord];
        }
        return retVal;
    }

    NBL_CONSTEXPR_INLINE_FUNC this_t operator-(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        NBL_CONSTEXPR_STATIC vector<U, D> Masks = NBL_HLSL_MORTON_MASKS(U, D);
        this_t retVal = { U(0) };
        [[unroll]]
        for (uint16_t coord = 0; coord < D; coord++)
        {
            // This is the dual trick of the one used for addition: set all other bits to 0 so borrows propagate
            retVal.value |= ((value & Masks[coord]) - (rhs.value & Masks[coord])) & Masks[coord];
        }
        return retVal;
    }

    // ------------------------------------------------------- COMPARISON OPERATORS -------------------------------------------------

    NBL_CONSTEXPR_INLINE_FUNC bool operator!() NBL_CONST_MEMBER_FUNC
    {
        return value.operator!();
    }

    NBL_CONSTEXPR_INLINE_FUNC bool coordEquals(NBL_CONST_REF_ARG(this_t) rhs, uint16_t coord) NBL_CONST_MEMBER_FUNC
    {
        NBL_CONSTEXPR_STATIC vector<U, D> Masks = NBL_HLSL_MORTON_MASKS(U, D);
        return (value & Masks[coord]) == (rhs.value & Masks[coord]);
    }

    NBL_CONSTEXPR_INLINE_FUNC vector<bool, D> operator==(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        vector<bool, D> retVal;
        [[unroll]]
        for (uint16_t coord = 0; coord < D; coord++)
            retVal[coord] = coordEquals(rhs, coord);
        return retVal;
    }

    NBL_CONSTEXPR_INLINE_FUNC bool allEqual(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        return value == rhs.value;
    }

    NBL_CONSTEXPR_INLINE_FUNC bool coordNotEquals(NBL_CONST_REF_ARG(this_t) rhs, uint16_t coord) NBL_CONST_MEMBER_FUNC
    {
        return !coordEquals(rhs, coord);
    }

    NBL_CONSTEXPR_INLINE_FUNC vector<bool, D> operator!=(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        vector<bool, D> retVal;
        [[unroll]]
        for (uint16_t coord = 0; coord < D; coord++)
            retVal[coord] = coordNotEquals(rhs, coord);
        return retVal;
    }

    NBL_CONSTEXPR_INLINE_FUNC bool notAllEqual(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        return !allEqual(rhs);
    }

    template<class Comparison, class OnSignMismatch>
    NBL_CONSTEXPR_INLINE_FUNC bool coordOrderCompare(NBL_CONST_REF_ARG(this_t) rhs, uint16_t coord) NBL_CONST_MEMBER_FUNC
    {
        NBL_CONSTEXPR_STATIC vector<U, D> Masks = NBL_HLSL_MORTON_MASKS(U, D);
        Comparison comparison;
        OnSignMismatch onSignMismatch;

        // When unsigned, bit representation is the same but with 0s inbetween bits. In particular, we can still use unsigned comparison
        #ifndef __HLSL_VERSION
        if constexpr (is_unsigned_v<I>)
        #else
        if (is_unsigned_v<I>)
        #endif
        {
            return comparison(value & Masks[coord], rhs.value & Masks[coord]);
        }
        // When signed, since the representation is unsigned, we need to divide behaviour based on highest bit
        else
        {
            // I will give an example for `operator<` but the same reasoning holds for all others. Some abuse of notation but hopefully it's clear.
            
            // If `this[coord] >= 0` and `rhs[coord] < 0` then `this[coord] < rhs[coord]` returns false. Notice that in this case, the highest bit of 
            // `value` (of the bits representing the number encoded in `coord`) is `0`, while the highest bit for rhs is `1`. 
            // Similarly, if `this[coord] < 0` and `rhs[coord] >= 0` then `this[coord] < rhs[coord]` returns true, and the highest bit situation is inverted.
            // This means that if the signs of `this[coord]` and `rhs[coord]` are not equal, the result depends on the sign of `this[coord]`.
            // What that result should be is controlled by `OnSignMismatch`.
            // Finally, notice that if only one of those bits is set to 1, then the `xor` of that highest bit yields 1 as well
            const U highestBit = extractHighestBit(coord);
            const U rhsHighestBit = rhs.extractHighestBit(coord);
            if (highestBit ^ rhsHighestBit)
                return onSignMismatch(highestBit);
            // If both are nonnegative, then we can just use the comparison as it comes.
            // If both are negative, it just so happens that applying the same operator to their unsigned bitcasted representations yields the same result.
            // For `operator<`, for example, consider two negative numbers. Starting from the MSB (we know it's `1` for both in this case) and moving to the right,
            // consider what happens when we encounter the first bit where they mismatch: the one with a `0` at position `k` (by position I mean counted from the
            // left, starting at 0) is adding at most `2^k - 1` in the lowest bits, while the one with a `1` is adding exactly `2^k`. This means that the one
            // with a 0 is "more negative". 
            else
                return comparison(value & Masks[coord], rhs.value & Masks[coord]);
        }
    }

    struct OnSignMismatchLessThan
    {
        // On a sign mismatch, `this<rhs` is true if this is negative (`highestBit` set to `1`) and false otherwise
        // Therefore since it takes a number with only the highest bit set we only have to return whether there is in fact a bit set
        bool operator()(U highestBit) 
        {
            return bool(highestBit);
        }
    };

    struct OnSignMismatchGreaterThan
    {
        // On a sign mismatch, `this>rhs` is true if this is non-negative (`highestBit` set to `0`) and false otherwise
        // Therefore since it takes a number with only the highest bit set we only have to return the opposite of whether there is in fact a bit set
        bool operator()(U highestBit)
        {
            return !bool(highestBit);
        }
    };
    
    NBL_CONSTEXPR_INLINE_FUNC bool coordLessThan(NBL_CONST_REF_ARG(this_t) rhs, uint16_t coord) NBL_CONST_MEMBER_FUNC
    {
        return coordOrderCompare<less<U>, OnSignMismatchLessThan>(rhs, coord);
    }

    NBL_CONSTEXPR_INLINE_FUNC bool coordLessThanEquals(NBL_CONST_REF_ARG(this_t) rhs, uint16_t coord) NBL_CONST_MEMBER_FUNC
    {
        return coordOrderCompare<less_equal<U>, OnSignMismatchLessThan>(rhs, coord);
    }

    NBL_CONSTEXPR_INLINE_FUNC bool coordGreaterThan(NBL_CONST_REF_ARG(this_t) rhs, uint16_t coord) NBL_CONST_MEMBER_FUNC
    {
        return coordOrderCompare<greater<U>, OnSignMismatchGreaterThan>(rhs, coord);
    }

    NBL_CONSTEXPR_INLINE_FUNC bool coordGreaterThanEquals(NBL_CONST_REF_ARG(this_t) rhs, uint16_t coord) NBL_CONST_MEMBER_FUNC
    {
        return coordOrderCompare<greater_equal<U>, OnSignMismatchGreaterThan>(rhs, coord);
    }

    #define DEFINE_OPERATOR(OP, COMPARISON) NBL_CONSTEXPR_INLINE_FUNC vector<bool, D> operator##OP##(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC \
    { \
        vector<bool, D> retVal; \
        [[unroll]] \
        for (uint16_t coord = 0; coord < D; coord++) \
            retVal[coord] = COMPARISON (rhs, coord); \
        return retVal; \
    }

    DEFINE_OPERATOR(< , coordLessThan);
    DEFINE_OPERATOR(<= , coordLessThanEquals);
    DEFINE_OPERATOR(> , coordGreaterThan);
    DEFINE_OPERATOR(>= , coordGreaterThanEquals);

    #undef DEFINE_OPERATOR

    U value;
};

// Don't forget to delete this macro after usage
#undef NBL_HLSL_MORTON_MASKS

} //namespace morton

// Still in nbl::hlsl we can go to nbl::hlsl::impl and specialize the `static_cast_helper`
namespace impl
{

template<typename I, uint16_t D>
struct static_cast_helper<vector<I, D>, morton::code<I,D> >
{
    NBL_CONSTEXPR_STATIC_INLINE_FUNC vector<I, D> cast(NBL_CONST_REF_ARG(morton::code<I, D>) val)
    {
        vector<I, D> cartesian;
        [[unroll]]
        for (uint16_t coord = 0; coord < D; coord++)
        {
            cartesian[coord] = val.getCoordinate(coord);
        }
        return cartesian;
    }
};

} // namespace impl

} //namespace hlsl
} //namespace nbl



#endif