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
        NBL_CONSTEXPR_STATIC vector<U, D> Masks = NBL_HLSL_MORTON_MASKS(U, D);
        const vector<U, D> unsignedCartesian = bit_cast<vector<U, D>, vector<I, D> >(cartesian);
        U val = U(0);
        
        [[unroll]]
        for (U dim = 0; dim < U(D); dim++)
        {
            [[unroll]]
            // Control can be simplified by running a bound on just coordBit based on `BitWidth` and `dim`, but I feel this is clearer
            for (U valBit = dim, coordBit = U(1), shift = dim; valBit < BitWidth; valBit += U(D), coordBit <<= 1, shift += U(D) - 1)
            {
                val |= (unsignedCartesian[dim] & coordBit) << shift;
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



    //operator+, operator-, operator>>, operator<<, and other bitwise ops

    U value;
};

// Don't forget to delete this macro after usage
#undef NBL_HLSL_MORTON_MASKS

} //namespace morton

namespace impl
{

template<typename I, uint16_t D>
struct static_cast_helper<vector<I, D>, morton::code<I,D> >
{
    NBL_CONSTEXPR_STATIC_INLINE_FUNC vector<I, D> cast(NBL_CONST_REF_ARG(morton::code<I, D>) val)
    {
        using U = typename morton::code<I, D>::U;
        NBL_CONSTEXPR_STATIC U BitWidth = morton::code<I, D>::BitWidth;
        // Converting back has an issue with bit-width: when encoding (if template parameter `I` is signed) we cut off the highest bits 
        // that actually indicated sign. Therefore what we do is set the highest bits instead of the lowest then do an arithmetic right shift 
        // at the end to preserve sign. 
        // To this end, we first notice that the coordinate/dimension of index `dim` gets 
        // `bits(dim) = ceil((BitWidth - dim)/D)` bits when encoded (so the first dimensions get more bits than the last ones if `D` does not 
        // divide `BitWidth perfectly`).
        // Then instead of unpacking all the bits for that coordinate as the lowest bits, we unpack them as the highest ones
        // by shifting everything `BitWidth - bits(dim)` bits to the left, then at the end do a final *arithmetic* bitshift right by the same amount.

        vector<I, D> cartesian;
        for (U dim = 0; dim < U(D); dim++)
        {
            const U bitsDim = (BitWidth - dim + U(D) - 1) / U(D); // <- this computes the ceil
            U coordVal = U(0);
            // Control can be simplified by running a bound on just coordBit based on `BitWidth` and `dim`, but I feel this is clearer
            for (U valBit = dim, coordBit = U(1) << dim, shift = dim; valBit < BitWidth; valBit += U(D), coordBit <<= U(D), shift += U(D) - 1)
            {
                coordVal |= (val.value & coordBit) << (BitWidth - bitsDim - shift);
            }
            cartesian[dim] = (bit_cast<I, U>(coordVal) >> (BitWidth - bitsDim));
        }
        return cartesian;
    }
};

} // namespace impl

} //namespace hlsl
} //namespace nbl



#endif