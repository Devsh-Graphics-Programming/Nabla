#ifndef _NBL_BUILTIN_HLSL_EMULATED_FLOAT64_T_INCLUDED_
#define _NBL_BUILTIN_HLSL_EMULATED_FLOAT64_T_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/ieee754.hlsl>
#include <nbl/builtin/hlsl/algorithm.hlsl>
#include <nbl/builtin/hlsl/tgmath.hlsl>
#include <nbl/builtin/hlsl/impl/emulated_float64_t_impl.hlsl>
   
#define FLOAT_ROUND_NEAREST_EVEN    0
#define FLOAT_ROUND_TO_ZERO         1
#define FLOAT_ROUND_DOWN            2
#define FLOAT_ROUND_UP              3
#define FLOAT_ROUNDING_MODE         FLOAT_ROUND_NEAREST_EVEN

namespace nbl
{
namespace hlsl
{
    template<bool FastMath = true, bool FlushDenromToZero = true>
    struct emulated_float64_t_impl
    {
        using storage_t = uint64_t;

        storage_t data;

        // constructors
        /*static emulated_float64_t_impl create(uint16_t val)
        {
            return emulated_float64_t_impl(bit_cast<uint64_t>(float64_t(val)));
        }*/

        static emulated_float64_t_impl create(int32_t val)
        {
            return emulated_float64_t_impl(bit_cast<uint64_t>(float64_t(val)));
        }

        static emulated_float64_t_impl create(int64_t val)
        {
            return emulated_float64_t_impl(bit_cast<uint64_t>(float64_t(val)));
        }

        static emulated_float64_t_impl create(uint32_t val)
        {
            return emulated_float64_t_impl(bit_cast<uint64_t>(float64_t(val)));
        }

        static emulated_float64_t_impl create(uint64_t val)
        {
            return emulated_float64_t_impl(bit_cast<uint64_t>(float64_t(val)));
        }

        static emulated_float64_t_impl create(float64_t val)
        {
            return emulated_float64_t_impl(bit_cast<uint64_t>(val));
        }
        
        // TODO: unresolved external symbol imath_half_to_float_table
        /*static emulated_float64_t_impl create(float16_t val)
        {
            return emulated_float64_t_impl(bit_cast<uint64_t>(float64_t(val)));
        }*/

        static emulated_float64_t_impl create(float32_t val)
        {
            return emulated_float64_t_impl(bit_cast<uint64_t>(float64_t(val)));
        }

        static emulated_float64_t_impl createPreserveBitPattern(uint64_t val)
        {
            return emulated_float64_t_impl(val);
        }

        // arithmetic operators
        emulated_float64_t_impl operator+(const emulated_float64_t_impl rhs)
        {
            emulated_float64_t_impl retval = createPreserveBitPattern(0u);

            uint64_t mantissa;
            uint32_t3 mantissaExtended;
            int biasedExp;

            uint64_t lhsSign = data & ieee754::traits<float64_t>::signMask;
            uint64_t rhsSign = rhs.data & ieee754::traits<float64_t>::signMask;
            uint64_t lhsMantissa = ieee754::extractMantissa(data);
            uint64_t rhsMantissa = ieee754::extractMantissa(rhs.data);
            int lhsBiasedExp = ieee754::extractBiasedExponent(data);
            int rhsBiasedExp = ieee754::extractBiasedExponent(rhs.data);
            
            int expDiff = lhsBiasedExp - rhsBiasedExp;

            if (lhsSign == rhsSign)
            {
                if (expDiff == 0)
                {
                    //if (lhsExp == 0x7FF)
                    //{
                    //   bool propagate = (lhsMantissa | rhsMantissa) != 0u;
                    //   return createPreserveBitPattern(lerp(data, impl::propagateFloat64NaN(data, rhs.data), propagate));
                    //}
                   
                    mantissa = lhsMantissa + rhsMantissa;
                    if (lhsBiasedExp == 0)
                      return createPreserveBitPattern(impl::assembleFloat64(lhsSign, 0, mantissa));
                    mantissaExtended.xy = impl::packUint64(mantissa);
                    mantissaExtended.x |= 0x00200000u;
                    mantissaExtended.z = 0u;
                    biasedExp = lhsBiasedExp;

                    mantissaExtended = impl::shift64ExtraRightJamming(mantissaExtended, 1);
                }
                else
                {
                     if (expDiff < 0)
                     {
                        swap<uint64_t>(lhsMantissa, rhsMantissa);
                        swap<int>(lhsBiasedExp, rhsBiasedExp);
                     }

                     if (lhsBiasedExp == 0x7FF)
                     {
                        const bool propagate = (lhsMantissa) != 0u;
                        return createPreserveBitPattern(lerp(ieee754::traits<float64_t>::exponentMask | lhsSign, impl::propagateFloat64NaN(data, rhs.data), propagate));
                     }

                     expDiff = lerp(abs(expDiff), abs(expDiff) - 1, rhsBiasedExp == 0);
                     rhsMantissa = lerp(rhsMantissa | 0x0010000000000000ull, rhsMantissa, rhsBiasedExp == 0);
                     const uint32_t3 shifted = impl::shift64ExtraRightJamming(uint32_t3(impl::packUint64(rhsMantissa), 0u), expDiff);
                     rhsMantissa = impl::unpackUint64(shifted.xy);
                     mantissaExtended.z = shifted.z;
                     biasedExp = lhsBiasedExp;

                     lhsMantissa |= 0x0010000000000000ull;
                     mantissaExtended.xy = impl::packUint64(lhsMantissa + rhsMantissa);
                     --biasedExp;
                     if (!(mantissaExtended.x < 0x00200000u))
                     {
                         mantissaExtended = impl::shift64ExtraRightJamming(mantissaExtended, 1);
                         ++biasedExp;
                     }
                     
                     return createPreserveBitPattern(impl::roundAndPackFloat64(lhsSign, biasedExp, mantissaExtended.xyz));
                }
                
                // cannot happen but compiler cries about not every path returning value
                return createPreserveBitPattern(0xdeadbeefbadcaffeull);
            }
            else
            {
                lhsMantissa = impl::shortShift64Left(lhsMantissa, 10);
                rhsMantissa = impl::shortShift64Left(rhsMantissa, 10);
                
                if (expDiff != 0)
                {
                    uint32_t2 frac;
                
                    if (expDiff < 0)
                    {
                        swap<uint64_t>(lhsMantissa, rhsMantissa);
                        swap<int>(lhsBiasedExp, rhsBiasedExp);
                        lhsSign ^= ieee754::traits<float64_t>::signMask;
                    }
                    
                    //if (lhsExp == 0x7FF)
                    //{
                    //   bool propagate = (lhsHigh | lhsLow) != 0u;
                    //   return nbl::hlsl::lerp(__packFloat64(lhsSign, 0x7ff, 0u, 0u), __propagateFloat64NaN(a, b), propagate);
                    //}
                    
                    expDiff = lerp(abs(expDiff), abs(expDiff) - 1, rhsBiasedExp == 0);
                    rhsMantissa = lerp(rhsMantissa | 0x4000000000000000ull, rhsMantissa, rhsBiasedExp == 0);
                    rhsMantissa = impl::unpackUint64(impl::shift64RightJamming(impl::packUint64(rhsMantissa), expDiff));
                    lhsMantissa |= 0x4000000000000000ull;
                    frac.xy = impl::packUint64(lhsMantissa - rhsMantissa);
                    biasedExp = lhsBiasedExp;
                    --biasedExp;
                    return createPreserveBitPattern(impl::normalizeRoundAndPackFloat64(lhsSign, biasedExp - 10, frac.x, frac.y));
                }
                //if (lhsExp == 0x7FF)
                //{
                //   bool propagate = ((lhsHigh | rhsHigh) | (lhsLow | rhsLow)) != 0u;
                //   return nbl::hlsl::lerp(0xFFFFFFFFFFFFFFFFUL, __propagateFloat64NaN(a, b), propagate);
                //}
                rhsBiasedExp = lerp(rhsBiasedExp, 1, lhsBiasedExp == 0);
                lhsBiasedExp = lerp(lhsBiasedExp, 1, lhsBiasedExp == 0);
                

                const uint32_t2 lhsMantissaPacked = impl::packUint64(lhsMantissa);
                const uint32_t2 rhsMantissaPacked = impl::packUint64(rhsMantissa);

                uint32_t2 frac;
                uint64_t signOfDifference = 0;
                if (rhsMantissaPacked.x < lhsMantissaPacked.x)
                {
                    frac.xy = impl::packUint64(lhsMantissa - rhsMantissa);
                }
                else if (lhsMantissaPacked.x < rhsMantissaPacked.x)
                {
                    frac.xy = impl::packUint64(rhsMantissa - lhsMantissa);
                    signOfDifference = ieee754::traits<float64_t>::signMask;
                }
                else if (rhsMantissaPacked.y <= lhsMantissaPacked.y)
                {
                    /* It is possible that frac.x and frac.y may be zero after this. */
                    frac.xy = impl::packUint64(lhsMantissa - rhsMantissa);
                }
                else
                {
                    frac.xy = impl::packUint64(rhsMantissa - lhsMantissa);
                    signOfDifference = ieee754::traits<float64_t>::signMask;
                }
                
                biasedExp = lerp(rhsBiasedExp, lhsBiasedExp, signOfDifference == 0u);
                lhsSign ^= signOfDifference;
                uint64_t retval_0 = impl::packFloat64(uint32_t(FLOAT_ROUNDING_MODE == FLOAT_ROUND_DOWN) << 31, 0, 0u, 0u);
                uint64_t retval_1 = impl::normalizeRoundAndPackFloat64(lhsSign, biasedExp - 11, frac.x, frac.y);
                return createPreserveBitPattern(lerp(retval_0, retval_1, frac.x != 0u || frac.y != 0u));
            }
        }

        emulated_float64_t_impl operator-(emulated_float64_t_impl rhs)
        {
            emulated_float64_t_impl lhs = createPreserveBitPattern(data);
            emulated_float64_t_impl rhsFlipped = rhs.flipSign();
            
            return lhs + rhsFlipped;
        }

        emulated_float64_t_impl operator*(emulated_float64_t_impl rhs)
        {
            emulated_float64_t_impl retval = emulated_float64_t_impl::createPreserveBitPattern(0u);

            uint64_t lhsSign = data & ieee754::traits<float64_t>::signMask;
            uint64_t rhsSign = rhs.data & ieee754::traits<float64_t>::signMask;
            uint64_t lhsMantissa = ieee754::extractMantissa(data);
            uint64_t rhsMantissa = ieee754::extractMantissa(rhs.data);
            int lhsBiasedExp = ieee754::extractBiasedExponent(data);
            int rhsBiasedExp = ieee754::extractBiasedExponent(rhs.data);

            int exp = int(lhsBiasedExp + rhsBiasedExp) - ieee754::traits<float64_t>::exponentBias;
            uint64_t sign = (data ^ rhs.data) & ieee754::traits<float64_t>::signMask;
            
            if (lhsBiasedExp == 0x7FF)
            {
                if ((lhsMantissa != 0u) ||
                    ((rhsBiasedExp == 0x7FF) && (rhsMantissa != 0u))) {
                    return createPreserveBitPattern(impl::propagateFloat64NaN(data, rhs.data));
                }
                if ((uint64_t(rhsBiasedExp) | rhsMantissa) == 0u)
                    return createPreserveBitPattern(0xFFFFFFFFFFFFFFFFull);

                return createPreserveBitPattern(impl::assembleFloat64(sign, ieee754::traits<float64_t>::exponentMask, 0ull));
            }
            if (rhsBiasedExp == 0x7FF)
            {
                /* a cannot be NaN, but is b NaN? */
                if (rhsMantissa != 0u)
#ifdef RELAXED_NAN_PROPAGATION
                    return rhs.data;
#else
                    return createPreserveBitPattern(impl::propagateFloat64NaN(data, rhs.data));
#endif
                if ((uint64_t(lhsBiasedExp) | lhsMantissa) == 0u)
                    return createPreserveBitPattern(0xFFFFFFFFFFFFFFFFull);

                return createPreserveBitPattern(sign | ieee754::traits<float64_t>::exponentMask);
            }
            if (lhsBiasedExp == 0)
            {
                if (lhsMantissa == 0u)
                    return createPreserveBitPattern(sign);
                impl::normalizeFloat64Subnormal(lhsMantissa, lhsBiasedExp, lhsMantissa);
            }
            if (rhsBiasedExp == 0)
            {
                if (rhsMantissa == 0u)
                    return createPreserveBitPattern(sign);
                impl::normalizeFloat64Subnormal(rhsMantissa, rhsBiasedExp, rhsMantissa);
            }

            if (false)
            {
                lhsMantissa |= 1ull << 52;
                rhsMantissa = impl::shortShift64Left(rhsMantissa, 12);

                uint32_t4 mantissasPacked;
                mantissasPacked.xy = impl::packUint64(lhsMantissa);
                mantissasPacked.zw = impl::packUint64(rhsMantissa);

                mantissasPacked = impl::mul64to128(mantissasPacked);

                mantissasPacked.xy = impl::packUint64(impl::unpackUint64(mantissasPacked.xy) + lhsMantissa);
                mantissasPacked.z |= uint32_t(mantissasPacked.w != 0u);
                if (0x00200000u <= mantissasPacked.x)
                {
                    mantissasPacked = uint32_t4(impl::shift64ExtraRightJamming(mantissasPacked.xyz, 1), 0u);
                    ++exp;
                }

                return createPreserveBitPattern(impl::roundAndPackFloat64(sign, exp, mantissasPacked.xyz));
            }
            else
            {
                lhsMantissa |= 1ull << 52;
                rhsMantissa |= 1ull << 52;

                uint32_t2 lhsPacked = impl::packUint64(lhsMantissa);
                uint32_t2 rhsPacked = impl::packUint64(rhsMantissa);
                uint64_t lhsHigh = lhsPacked.x;
                uint64_t lhsLow = lhsPacked.y;
                uint64_t rhsHigh = rhsPacked.x;
                uint64_t rhsLow = rhsPacked.y;

                //((hi_lhs * hi_rhs) << 11) + ((hi_lhs * lo_rhs + lo_lhs * hi_rhs) >> 37)

                uint64_t newPseudoMantissa = ((lhsHigh * rhsHigh) << 11) + ((lhsHigh * rhsLow + lhsLow * rhsHigh) >> 37);
                newPseudoMantissa <<= 1;
                //newPseudoMantissa >>= 52;
                /*if (newPseudoMantissa >= (1ull << 52))
                {
                    newPseudoMantissa >>= 1;
                    ++exp;
                }*/

                return createPreserveBitPattern(impl::assembleFloat64(sign, uint64_t(exp) << ieee754::traits<float64_t>::mantissaBitCnt, newPseudoMantissa & ieee754::traits<float64_t>::mantissaMask));
            }

            
        }

        emulated_float64_t_impl operator/(const emulated_float64_t_impl rhs)
        {

            // TODO: maybe add function to extract real mantissa
            const uint64_t lhsRealMantissa = (ieee754::extractMantissa(data) | (1ull << ieee754::traits<float64_t>::mantissaBitCnt));
            const uint64_t rhsRealMantissa = ieee754::extractMantissa(rhs.data) | (1ull << ieee754::traits<float64_t>::mantissaBitCnt);

            
            const uint64_t sign = (data ^ rhs.data) & ieee754::traits<float64_t>::signMask;
            int exp = ieee754::extractExponent(data) - ieee754::extractExponent(rhs.data) + ieee754::traits<float64_t>::exponentBias;
            uint64_t mantissa = impl::divMantissas(lhsRealMantissa, rhsRealMantissa);

            if (mantissa & (1ULL << (ieee754::traits<float64_t>::mantissaBitCnt + 1)))
            {
                mantissa >>= 1;
                ++exp;
            }

            return createPreserveBitPattern(impl::assembleFloat64(sign, exp, mantissa & ieee754::traits<float64_t>::mantissaMask));
        }


        // relational operators
        bool operator==(emulated_float64_t_impl rhs)
        {
            if (FastMath && (isnan(data) || isnan(rhs.data)))
                return false;

            const emulated_float64_t_impl xored = emulated_float64_t_impl::createPreserveBitPattern(data ^ rhs.data);
            // TODO: check what fast math returns for -0 == 0
            if ((xored.data & 0x7FFFFFFFFFFFFFFFull) == 0ull)
                return true;

            return !(xored.data);
        }
        bool operator!=(emulated_float64_t_impl rhs)
        {
            if (FastMath && (isnan(data) || isnan(rhs.data)))
                return true;

            const emulated_float64_t_impl xored = emulated_float64_t_impl::createPreserveBitPattern(data ^ rhs.data);

            // TODO: check what fast math returns for -0 == 0
            if ((xored.data & 0x7FFFFFFFFFFFFFFFull) == 0ull)
                return false;

            return xored.data;
        }
        bool operator<(emulated_float64_t_impl rhs)
        {
            const uint64_t lhsSign = ieee754::extractSign(data);
            const uint64_t rhsSign = ieee754::extractSign(rhs.data);

            // flip bits of negative numbers and flip signs of all numbers
            uint64_t lhsFlipped = data ^ ((0x7FFFFFFFFFFFFFFFull * lhsSign) | ieee754::traits<float64_t>::signMask);
            uint64_t rhsFlipped = rhs.data ^ ((0x7FFFFFFFFFFFFFFFull * rhsSign) | ieee754::traits<float64_t>::signMask);

            uint64_t diffBits = lhsFlipped ^ rhsFlipped;

            return (lhsFlipped & diffBits) < (rhsFlipped & diffBits);
        }
        bool operator>(emulated_float64_t_impl rhs) 
        {
            const uint64_t lhsSign = ieee754::extractSign(data);
            const uint64_t rhsSign = ieee754::extractSign(rhs.data);

            // flip bits of negative numbers and flip signs of all numbers
            uint64_t lhsFlipped = data ^ ((0x7FFFFFFFFFFFFFFFull * lhsSign) | ieee754::traits<float64_t>::signMask);
            uint64_t rhsFlipped = rhs.data ^ ((0x7FFFFFFFFFFFFFFFull * rhsSign) | ieee754::traits<float64_t>::signMask);

            uint64_t diffBits = lhsFlipped ^ rhsFlipped;

            return (lhsFlipped & diffBits) > (rhsFlipped & diffBits);
        }
        bool operator<=(emulated_float64_t_impl rhs) { return !(emulated_float64_t_impl::createPreserveBitPattern(data) > emulated_float64_t_impl::createPreserveBitPattern(rhs.data)); }
        bool operator>=(emulated_float64_t_impl rhs) { return !(emulated_float64_t_impl::createPreserveBitPattern(data) < emulated_float64_t_impl::createPreserveBitPattern(rhs.data)); }

        //logical operators
        bool operator&&(emulated_float64_t_impl rhs) { return bool(data) && bool(rhs.data); }
        bool operator||(emulated_float64_t_impl rhs) { return bool(data) || bool(rhs.data); }
        bool operator!() { return !bool(data); }

        // OMITED OPERATORS
        //  - not implementing bitwise and modulo operators since floating point types doesn't support them
        //  - compound operator overload not supported in HLSL
        //  - access operators (dereference and addressof) not supported in HLSL
        
        // TODO: should modify self?
        emulated_float64_t_impl flipSign()
        {
            return createPreserveBitPattern(data ^ ieee754::traits<float64_t>::signMask);
        }
        
        bool isNaN()
        {
            return isnan(bit_cast<float64_t>(data));
        }
    };

    using emulated_float64_t = emulated_float64_t_impl<true, true>;

namespace ieee754
{
    template<>
    struct traits_base<emulated_float64_t>
    {
        NBL_CONSTEXPR_STATIC_INLINE int16_t exponentBitCnt = 11;
        NBL_CONSTEXPR_STATIC_INLINE int16_t mantissaBitCnt = 52;
    };

	template<>
	uint32_t extractBiasedExponent(emulated_float64_t x)
	{
        return extractBiasedExponent<uint64_t>(x.data);
	}

	template<>
	int extractExponent(emulated_float64_t x)
	{
		return extractExponent(x.data);
	}

	template<>
    emulated_float64_t replaceBiasedExponent(emulated_float64_t x, typename unsigned_integer_of_size<sizeof(emulated_float64_t)>::type biasedExp)
	{
        return emulated_float64_t(replaceBiasedExponent(x.data, biasedExp));
	}

	//// performs no overflow tests, returns x*exp2(n)
	template <>
    emulated_float64_t fastMulExp2(emulated_float64_t x, int n)
	{
        return emulated_float64_t(replaceBiasedExponent(x.data, extractBiasedExponent(x) + uint32_t(n)));
	}

	template <>
	unsigned_integer_of_size<sizeof(emulated_float64_t)>::type extractMantissa(emulated_float64_t x)
	{
        return extractMantissa(x.data);
	}
}

}
}

#undef FLOAT_ROUND_NEAREST_EVEN
#undef FLOAT_ROUND_TO_ZERO
#undef FLOAT_ROUND_DOWN
#undef FLOAT_ROUND_UP
#undef FLOAT_ROUNDING_MODE

#endif
