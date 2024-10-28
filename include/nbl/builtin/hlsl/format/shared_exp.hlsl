#ifndef _NBL_HLSL_FORMAT_SHARED_EXP_HLSL_
#define _NBL_HLSL_FORMAT_SHARED_EXP_HLSL_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/type_traits.hlsl"
#include "nbl/builtin/hlsl/limits.hlsl"

namespace nbl
{
namespace hlsl
{

namespace format
{

template<typename IntT, uint16_t _Components, uint16_t _ExponentBits>
struct shared_exp// : enable_if_t<_ExponentBits<16> need a way to static_assert in SPIRV!
{
    using this_t = shared_exp<IntT,_Components,_ExponentBits>;
    using storage_t = typename make_unsigned<IntT>::type;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t Components = _Components;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t ExponentBits = _ExponentBits;

    // Not even going to consider fp16 and fp64 dependence on device traits
    using decode_t = float32_t;

    inline bool operator==(const this_t other)
    {
        return storage==other.storage;
    }
    inline bool operator!=(const this_t other)
    {
        return storage==other.storage;
    }

    storage_t storage;
};

// all of this because DXC has bugs in partial template spec
namespace impl
{
template<typename IntT, uint16_t _Components, uint16_t _ExponentBits>
struct numeric_limits_shared_exp
{
    using type = format::shared_exp<IntT,_Components,_ExponentBits>;
    using value_type = typename type::decode_t;
    using __storage_t = typename type::storage_t;

    NBL_CONSTEXPR_STATIC_INLINE bool is_specialized = true;
    NBL_CONSTEXPR_STATIC_INLINE bool is_signed = is_signed_v<IntT>;
    NBL_CONSTEXPR_STATIC_INLINE bool is_integer = false;
    NBL_CONSTEXPR_STATIC_INLINE bool is_exact = false;
    // infinity and NaN are not representable in shared exponent formats
    NBL_CONSTEXPR_STATIC_INLINE bool has_infinity = false;
    NBL_CONSTEXPR_STATIC_INLINE bool has_quiet_NaN = false;
    NBL_CONSTEXPR_STATIC_INLINE bool has_signaling_NaN = false;
    // shared exponent formats have no leading 1 in the mantissa, therefore denormalized values aren't really a concept, although one can argue all values are denorm then?
    NBL_CONSTEXPR_STATIC_INLINE bool has_denorm = false;
    NBL_CONSTEXPR_STATIC_INLINE bool has_denorm_loss = false;
    // truncation
//    NBL_CONSTEXPR_STATIC_INLINE float_round_style round_style = round_to_nearest;
    NBL_CONSTEXPR_STATIC_INLINE bool is_iec559 = false;
    NBL_CONSTEXPR_STATIC_INLINE bool is_bounded = true;
    NBL_CONSTEXPR_STATIC_INLINE bool is_modulo = false;
    NBL_CONSTEXPR_STATIC_INLINE int32_t digits = (sizeof(IntT)*8-(is_signed ? _Components:0)-_ExponentBits)/_Components;
    NBL_CONSTEXPR_STATIC_INLINE int32_t radix = 2;
    NBL_CONSTEXPR_STATIC_INLINE int32_t max_exponent = 1<<(_ExponentBits-1);
    NBL_CONSTEXPR_STATIC_INLINE int32_t min_exponent = 1-max_exponent;
    NBL_CONSTEXPR_STATIC_INLINE bool traps = false;
    
    // extras
    NBL_CONSTEXPR_STATIC_INLINE __storage_t MantissaMask = ((__storage_t(1))<<digits)-__storage_t(1);
    NBL_CONSTEXPR_STATIC_INLINE uint16_t ExponentBits = _ExponentBits;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t ExponentMask = uint16_t((1<<_ExponentBits)-1);

 // TODO: functions done as vars
//    NBL_CONSTEXPR_STATIC_INLINE value_type min = base::min();
    // shift down by 1 to get rid of explicit 1 in mantissa that is now implicit, then +1 in the exponent to compensate
    NBL_CONSTEXPR_STATIC_INLINE __storage_t max =
        ((max_exponent+1-numeric_limits<value_type>::min_exponent)<<(numeric_limits<value_type>::digits-1))|
        ((MantissaMask>>1)<<(numeric_limits<value_type>::digits-digits));
    NBL_CONSTEXPR_STATIC_INLINE __storage_t lowest = is_signed ? ((__storage_t(1)<<(sizeof(__storage_t)*8-1))|max):__storage_t(0);
/*
    NBL_CONSTEXPR_STATIC_INLINE value_type epsilon = base::epsilon();
    NBL_CONSTEXPR_STATIC_INLINE value_type round_error = base::round_error();
*/
};
}

}

// specialize the limits
template<typename IntT, uint16_t _Components, uint16_t _ExponentBits>
struct numeric_limits<format::shared_exp<IntT,_Components,_ExponentBits> > : format::impl::numeric_limits_shared_exp<IntT,_Components,_ExponentBits>
{
};

namespace impl
{
// TODO: versions for `float16_t`

// decode
template<typename IntT, uint16_t _Components, uint16_t _ExponentBits>
struct static_cast_helper<
    vector<typename format::shared_exp<IntT,_Components,_ExponentBits>::decode_t,_Components>,
    format::shared_exp<IntT,_Components,_ExponentBits>
>
{
    using U = format::shared_exp<IntT,_Components,_ExponentBits>;
    using T = vector<typename U::decode_t,_Components>;

    static inline T cast(U val)
    {
        using storage_t = typename U::storage_t;
        // DXC error: error: expression class 'DependentScopeDeclRefExpr' unimplemented, doesn't matter as decode_t is always float32_t for now
        //using decode_t = typename T::decode_t;
        using decode_t = float32_t;
        // no clue why the compiler doesn't pick up the partial specialization and tries to use the general one
        using limits_t = format::impl::numeric_limits_shared_exp<IntT,_Components,_ExponentBits>;

        T retval;
        for (uint16_t i=0; i<_Components; i++)
            retval[i] = decode_t((val.storage>>storage_t(limits_t::digits*i))&limits_t::MantissaMask);
        uint16_t exponent = uint16_t(val.storage>>storage_t(limits_t::digits*3));
        if (limits_t::is_signed)
        {
            for (uint16_t i=0; i<_Components; i++)
            if (exponent&(uint16_t(1)<<(_ExponentBits+i)))
                retval[i] = -retval[i];
            exponent &= limits_t::ExponentMask;
        }
        return retval*exp2(int32_t(exponent-limits_t::digits)+limits_t::min_exponent);
    }
};
// encode (WARNING DOES NOT CHECK THAT INPUT IS IN THE RANGE!)
template<typename IntT, uint16_t _Components, uint16_t _ExponentBits>
struct static_cast_helper<
    format::shared_exp<IntT,_Components,_ExponentBits>,
    vector<typename format::shared_exp<IntT,_Components,_ExponentBits>::decode_t,_Components>
>
{
    using T = format::shared_exp<IntT,_Components,_ExponentBits>;
    using U = vector<typename T::decode_t,_Components>;

    static inline T cast(U val)
    {
        using storage_t = typename T::storage_t;
        // DXC error: error: expression class 'DependentScopeDeclRefExpr' unimplemented, doesn't matter as decode_t is always float32_t for now
        //using decode_t = typename T::decode_t;
        using decode_t = float32_t;
        //
        using decode_bits_t = unsigned_integer_of_size<sizeof(decode_t)>::type;
        // no clue why the compiler doesn't pick up the partial specialization and tries to use the general one
        using limits_t = format::impl::numeric_limits_shared_exp<IntT,_Components,_ExponentBits>;

        // get exponents
        vector<uint16_t,_Components> exponentsDecBias;
        const int32_t dec_MantissaStoredBits = numeric_limits<decode_t>::digits-1;
        for (uint16_t i=0; i<_Components; i++)
        {
            decode_t v = val[i];
            if (limits_t::is_signed)
                v = abs(v);
            exponentsDecBias[i] = uint16_t(asuint(v)>>dec_MantissaStoredBits);
        }

        // get the maximum exponent
        uint16_t sharedExponentDecBias = exponentsDecBias[0];
        for (uint16_t i=1; i<_Components; i++)
            sharedExponentDecBias = max(exponentsDecBias[i],sharedExponentDecBias);

        // NOTE: we don't consider clamping against `limits_t::max_exponent`, should be ensured by clamping the inputs against `limits_t::max` before casting!

        // we need to stop "shifting up" implicit leading 1. to farthest left position if the exponent too small
        uint16_t clampedSharedExponentDecBias;
        if (limits_t::min_exponent>numeric_limits<decode_t>::min_exponent) // if ofc its needed at all
            clampedSharedExponentDecBias = max(sharedExponentDecBias,uint16_t(limits_t::min_exponent-numeric_limits<decode_t>::min_exponent));
        else
            clampedSharedExponentDecBias = sharedExponentDecBias;

        // we always shift down, the question is how much
        vector<uint16_t,_Components> mantissaShifts;
        for (uint16_t i=0; i<_Components; i++)
            mantissaShifts[i] = min(clampedSharedExponentDecBias+uint16_t(-limits_t::min_exponent)-exponentsDecBias[i],uint16_t(numeric_limits<decode_t>::digits));
        
        // finally lets re-bias our exponent (it will always be positive), note the -1 because IEEE754 floats reserve the lowest exponent values for denorm
        const uint16_t sharedExponentEncBias = int16_t(clampedSharedExponentDecBias+int16_t(-limits_t::min_exponent))-uint16_t(1-numeric_limits<decode_t>::min_exponent);

        //
        T retval;
        retval.storage = storage_t(sharedExponentEncBias)<<(limits_t::digits*3);
        const decode_bits_t dec_MantissaMask = (decode_bits_t(1)<<dec_MantissaStoredBits)-1;
        for (uint16_t i=0; i<_Components; i++)
        {
            decode_bits_t origBitPattern = bit_cast<decode_bits_t>(val[i])&dec_MantissaMask;
            // put the implicit 1 in (don't care about denormalized because its probably less than our `limits_t::min` (TODO: static assert it)
            origBitPattern |= decode_bits_t(1)<<dec_MantissaStoredBits;
            // shift and put in the right place
            retval.storage |= storage_t(origBitPattern>>mantissaShifts[i])<<(limits_t::digits*i);
        }
        if (limits_t::is_signed)
        {
            // doing ops on smaller integers is faster
            decode_bits_t SignMask = 0x1<<(sizeof(decode_t)*8-1);
            decode_bits_t signs = bit_cast<decode_bits_t>(val[0])&SignMask;
            for (uint16_t i=1; i<_Components; i++)
                signs |= (bit_cast<decode_bits_t>(val[i])&SignMask)>>i;
            retval.storage |= storage_t(signs)<<((sizeof(storage_t)-sizeof(decode_t))*8);
        }
        return retval;
    }
};
}
}
}
#endif