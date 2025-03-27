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
    code(const I _value) : value(bit_cast<U,I>(_value)){}

    #endif

    /**
    * @brief Creates a Morton code from a set of cartesian coordinates
    *
    * @param [in] cartesian Coordinates to encode
    */
    NBL_CONSTEXPR_STATIC_FUNC this_t create(NBL_CONST_REF_ARG(vector<I, D>) cartesian)
    {
        const vector<U, D> unsignedCartesian = bit_cast<vector<U, D>, vector<I, D> >(cartesian);
        U val = U(0);
        
        [[unroll]]
        for (U coord = 0; coord < U(D); coord++)
        {
            [[unroll]]
            // Control can be simplified by running a bound on just coordBit based on `BitWidth` and `coord`, but I feel this is clearer
            for (U valBitIdx = coord, coordBit = U(1), shift = coord; valBitIdx < BitWidth; valBitIdx += U(D), coordBit <<= 1, shift += U(D) - 1)
            {
                val |= (unsignedCartesian[coord] & coordBit) << shift;
            }
        }
        
        this_t retVal;
        retVal.value = val;
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
    * @brief Returns an element of type U by `or`ing this with rhs and extracting only the highest bit. Useful to know if either coord 
    * (for each value) has its highest bit set to 1.
    *
    * @param [in] coord The coordinate whose highest bit we want to get
    */
    NBL_CONSTEXPR_INLINE_FUNC U logicalOrHighestBits(NBL_CONST_REF_ARG(this_t) rhs, uint16_t coord) NBL_CONST_MEMBER_FUNC
    {
        // Like above, if the number encoded in `coord` gets `bits(coord) = ceil((BitWidth - coord)/D)` bits for representation, then the highest index of these
        // bits is `bits(coord) - 1`
        const U coordHighestBitIdx = BitWidth / U(D) - ((U(coord) < BitWidth % U(D)) ? U(0) : U(1));
        // This is the index of that bit as an index in the encoded value
        const U shift = coordHighestBitIdx * U(D) + U(coord);
        return (value | rhs.value) & (U(1) << shift);
    }

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

    NBL_CONSTEXPR_INLINE_FUNC this_t operator<<(uint16_t bits) NBL_CONST_MEMBER_FUNC
    {
        this_t retVal;
        retVal.value = value << bits;
        return retVal;
    }

    NBL_CONSTEXPR_INLINE_FUNC this_t operator>>(uint16_t bits) NBL_CONST_MEMBER_FUNC
    {
        this_t retVal;
        retVal.value = value >> bits;
        return retVal;
    }

    #endif

    // ------------------------------------------------------- UNARY ARITHMETIC OPERATORS -------------------------------------------------

    NBL_CONSTEXPR_INLINE_FUNC this_t operator-() NBL_CONST_MEMBER_FUNC
    {
        this_t allOnes;
        // allOnes encodes a cartesian coordinate with all values set to 1
        allOnes.value = (U(1) << D) - U(1);
        // Using 2's complement property that arithmetic negation can be obtained by bitwise negation then adding 1
        return operator~() + allOnes;
    }

    // ------------------------------------------------------- BINARY ARITHMETIC OPERATORS -------------------------------------------------

    NBL_CONSTEXPR_INLINE_FUNC this_t operator+(NBL_CONST_REF_ARG(this_t) rhs) NBL_CONST_MEMBER_FUNC
    {
        NBL_CONSTEXPR_STATIC vector<U, D> Masks = NBL_HLSL_MORTON_MASKS(U, D);
        this_t retVal;
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
        this_t retVal;
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
        return ! allEqual(rhs);
    }
    
    

    template<class Comparison, class OppositeComparison>
    NBL_CONSTEXPR_INLINE_FUNC bool coordOrderCompare(NBL_CONST_REF_ARG(this_t) rhs, uint16_t coord) NBL_CONST_MEMBER_FUNC
    {
        NBL_CONSTEXPR_STATIC vector<U, D> Masks = NBL_HLSL_MORTON_MASKS(U, D);
        Comparison comparison;
        OppositeComparison oppositeComparison;

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
            // I will give an example for the case of `Comparison` being `functional::less`, but other cases are similar
            // If both are negative (both bits set to 1) then `x < y` iff `z > w` when `z,w` are the bit representations of `x,y` as unsigned
            // If this is nonnegative and rhs is negative, it should return false. Since in this case `highestBit = 0` and `rhsHighestBit = 1` this
            // is the same as doing `z > w` again
            // If this is negative and rhs is nonnegative, it should return true. But in this case we have `highestBit = 1` and `rhsHighestBit = 0`
            // so again we can just return `z > w`.
            // All three cases end up in the same expression.
            if (logicalOrHighestBits(rhs, coord))
                return oppositeComparison(value & Masks[coord], rhs.value & Masks[coord]);
            // If neither of them have their highest bit set, both are nonnegative. Therefore, we can return the unsigned comparison
            else
                return comparison(value & Masks[coord], rhs.value & Masks[coord]);
        }
    }
    
    NBL_CONSTEXPR_INLINE_FUNC bool coordLessThan(NBL_CONST_REF_ARG(this_t) rhs, uint16_t coord) NBL_CONST_MEMBER_FUNC
    {
        return coordOrderCompare<less<U>, greater<U> >(rhs, coord);
    }

    NBL_CONSTEXPR_INLINE_FUNC bool coordLessThanEquals(NBL_CONST_REF_ARG(this_t) rhs, uint16_t coord) NBL_CONST_MEMBER_FUNC
    {
        return coordOrderCompare<less_equal<U>, greater_equal<U> >(rhs, coord);
    }

    NBL_CONSTEXPR_INLINE_FUNC bool coordGreaterThan(NBL_CONST_REF_ARG(this_t) rhs, uint16_t coord) NBL_CONST_MEMBER_FUNC
    {
        return coordOrderCompare<greater<U>, less<U> >(rhs, coord);
    }

    NBL_CONSTEXPR_INLINE_FUNC bool coordGreaterThanEquals(NBL_CONST_REF_ARG(this_t) rhs, uint16_t coord) NBL_CONST_MEMBER_FUNC
    {
        return coordOrderCompare<greater_equal<U>, less_equal<U> >(rhs, coord);
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