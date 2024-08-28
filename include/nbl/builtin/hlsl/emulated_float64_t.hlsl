#ifndef _NBL_BUILTIN_HLSL_EMULATED_FLOAT64_T_INCLUDED_
#define _NBL_BUILTIN_HLSL_EMULATED_FLOAT64_T_INCLUDED_

#include <nbl/builtin/hlsl/impl/emulated_float64_t_impl.hlsl>

namespace nbl
{
namespace hlsl
{
    template<bool FastMath = true, bool FlushDenormToZero = true>
    struct emulated_float64_t
    {
        using storage_t = uint64_t;
        using this_t = emulated_float64_t<FastMath, FlushDenormToZero>;

        storage_t data;

        // constructors
        /*static emulated_float64_t create(uint16_t val)
        {
            return emulated_float64_t(bit_cast<uint64_t>(float64_t(val)));
        }*/

        NBL_CONSTEXPR_STATIC_INLINE this_t create(this_t val)
        {
            return val;
        }

        NBL_CONSTEXPR_STATIC_INLINE this_t create(int32_t val)
        {
            return bit_cast<this_t >(impl::castToUint64WithFloat64BitPattern(int64_t(val)));
        }

        NBL_CONSTEXPR_STATIC_INLINE this_t create(int64_t val)
        {
            return bit_cast<this_t >(impl::castToUint64WithFloat64BitPattern(val));
        }

        NBL_CONSTEXPR_STATIC_INLINE this_t create(uint32_t val)
        {
            return bit_cast<this_t >(impl::castToUint64WithFloat64BitPattern(uint64_t(val)));
        }

        NBL_CONSTEXPR_STATIC_INLINE this_t create(uint64_t val)
        {
            return bit_cast<this_t >(impl::castToUint64WithFloat64BitPattern(val));
        }

        NBL_CONSTEXPR_STATIC_INLINE this_t create(float32_t val)
        {
            this_t output;
            output.data = impl::castFloat32ToStorageType<FlushDenormToZero>(val);
            return output;
        }

        NBL_CONSTEXPR_STATIC_INLINE this_t create(float64_t val)
        {
#ifdef __HLSL_VERSION
            emulated_float64_t retval;
            uint32_t lo, hi;
            asuint(val, lo, hi);
            retval.data = (uint64_t(hi) << 32) | lo;
            return retval;
#else
            return bit_cast<this_t >(reinterpret_cast<uint64_t&>(val));
#endif
        }
        
        // TODO: unresolved external symbol imath_half_to_float_table
        /*static emulated_float64_t create(float16_t val)
        {
            return emulated_float64_t(bit_cast<uint64_t>(float64_t(val)));
        }*/

        // TODO: remove
        emulated_float64_t addOld(const emulated_float64_t rhs) NBL_CONST_MEMBER_FUNC
        {
            if (FlushDenormToZero)
            {
                emulated_float64_t<FastMath, FlushDenormToZero> retval = emulated_float64_t<FastMath, FlushDenormToZero>::create(0ull);

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
                        if (lhsBiasedExp == ieee754::traits<float64_t>::specialValueExp)
                        {
                            bool propagate = (lhsMantissa | rhsMantissa) != 0u;
                            return bit_cast<emulated_float64_t<FastMath, FlushDenormToZero> >(glsl::mix(data, impl::propagateFloat64NaN(data, rhs.data), propagate));
                        }

                        mantissa = lhsMantissa + rhsMantissa;
                        if (lhsBiasedExp == 0)
                            return bit_cast<emulated_float64_t<FastMath, FlushDenormToZero> >(impl::assembleFloat64(lhsSign, 0, mantissa));
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

                        if (lhsBiasedExp == ieee754::traits<float64_t>::specialValueExp)
                        {
                            const bool propagate = (lhsMantissa) != 0u;
                            return bit_cast<emulated_float64_t<FastMath, FlushDenormToZero> >(glsl::mix(ieee754::traits<float64_t>::exponentMask | lhsSign, impl::propagateFloat64NaN(data, rhs.data), propagate));
                        }

                        expDiff = glsl::mix(abs(expDiff), abs(expDiff) - 1, rhsBiasedExp == 0);
                        rhsMantissa = glsl::mix(rhsMantissa | (1ull << 52), rhsMantissa, rhsBiasedExp == 0);
                        const uint32_t3 shifted = impl::shift64ExtraRightJamming(uint32_t3(impl::packUint64(rhsMantissa), 0u), expDiff);
                        rhsMantissa = impl::unpackUint64(shifted.xy);
                        mantissaExtended.z = shifted.z;
                        biasedExp = lhsBiasedExp;

                        lhsMantissa |= (1ull << 52);
                        mantissaExtended.xy = impl::packUint64(lhsMantissa + rhsMantissa);
                        --biasedExp;
                        if (!(mantissaExtended.x < 0x00200000u))
                        {
                            mantissaExtended = impl::shift64ExtraRightJamming(mantissaExtended, 1);
                            ++biasedExp;
                        }

                        return bit_cast<emulated_float64_t<FastMath, FlushDenormToZero> >(impl::roundAndPackFloat64(lhsSign, biasedExp, mantissaExtended.xyz));
                    }

                    // cannot happen but compiler cries about not every path returning value
                    return bit_cast<emulated_float64_t<FastMath, FlushDenormToZero> >(impl::roundAndPackFloat64(lhsSign, biasedExp, mantissaExtended));
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

                        if (lhsBiasedExp == ieee754::traits<float64_t>::specialValueExp)
                        {
                            bool propagate = lhsMantissa != 0u;
                            return bit_cast<emulated_float64_t<FastMath, FlushDenormToZero> >(glsl::mix(impl::assembleFloat64(lhsSign, ieee754::traits<float64_t>::exponentMask, 0ull), impl::propagateFloat64NaN(data, rhs.data), propagate));
                        }

                        expDiff = glsl::mix(abs(expDiff), abs(expDiff) - 1, rhsBiasedExp == 0);
                        rhsMantissa = glsl::mix(rhsMantissa | 0x4000000000000000ull, rhsMantissa, rhsBiasedExp == 0);
                        rhsMantissa = impl::unpackUint64(impl::shift64RightJamming(impl::packUint64(rhsMantissa), expDiff));
                        lhsMantissa |= 0x4000000000000000ull;
                        frac.xy = impl::packUint64(lhsMantissa - rhsMantissa);
                        biasedExp = lhsBiasedExp;
                        --biasedExp;
                        return bit_cast<emulated_float64_t<FastMath, FlushDenormToZero> >(impl::normalizeRoundAndPackFloat64(lhsSign, biasedExp - 10, frac.x, frac.y));
                    }
                    if (lhsBiasedExp == ieee754::traits<float64_t>::specialValueExp)
                    {
                        bool propagate = ((lhsMantissa) | (rhsMantissa)) != 0u;
                        return bit_cast<emulated_float64_t<FastMath, FlushDenormToZero> >(glsl::mix(ieee754::traits<float64_t>::quietNaN, impl::propagateFloat64NaN(data, rhs.data), propagate));
                    }
                    rhsBiasedExp = glsl::mix(rhsBiasedExp, 1, lhsBiasedExp == 0);
                    lhsBiasedExp = glsl::mix(lhsBiasedExp, 1, lhsBiasedExp == 0);


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

                    biasedExp = glsl::mix(rhsBiasedExp, lhsBiasedExp, signOfDifference == 0u);
                    lhsSign ^= signOfDifference;
                    uint64_t retval_0 = impl::packFloat64(uint32_t(FLOAT_ROUNDING_MODE == FLOAT_ROUND_DOWN) << 31, 0, 0u, 0u);
                    uint64_t retval_1 = impl::normalizeRoundAndPackFloat64(lhsSign, biasedExp - 11, frac.x, frac.y);
                    return bit_cast<emulated_float64_t<FastMath, FlushDenormToZero> >(glsl::mix(retval_0, retval_1, frac.x != 0u || frac.y != 0u));
                }
            }
            else
            {
                //static_assert(false, "not implemented yet");
                return bit_cast<emulated_float64_t<FastMath, FlushDenormToZero> >(0xdeadbeefbadcaffeull);
            }
        }

        // arithmetic operators
        this_t operator+(const emulated_float64_t rhs) NBL_CONST_MEMBER_FUNC
        {
            return addOld(rhs);

            if (FlushDenormToZero)
            {
                if (!FastMath && (tgmath::isnan(data) || tgmath::isnan(rhs.data)))
                    return bit_cast<this_t>(ieee754::traits<float64_t>::quietNaN);

                int lhsBiasedExp = ieee754::extractBiasedExponent(data);
                int rhsBiasedExp = ieee754::extractBiasedExponent(rhs.data);

                if (lhsBiasedExp == 0ull)
                    return bit_cast<this_t>(rhs.data);
                if (rhsBiasedExp == 0ull)
                    return bit_cast<this_t>(data);

                uint64_t lhsSign = ieee754::extractSign(data);
                uint64_t rhsSign = ieee754::extractSign(rhs.data);

                int64_t lhsNormMantissa = int64_t(ieee754::extractNormalizeMantissa(data));
                int64_t rhsNormMantissa = int64_t(ieee754::extractNormalizeMantissa(rhs.data));

                lhsNormMantissa <<= 9;
                rhsNormMantissa <<= 9;

                // TODO: branchless?
                if (lhsSign != rhsSign)
                {
                    if (lhsSign)
                        lhsNormMantissa *= -1;
                    if (rhsSign)
                        rhsNormMantissa *= -1;
                }

                int expDiff = lhsBiasedExp - rhsBiasedExp;

                int exp = max(lhsBiasedExp, rhsBiasedExp) - ieee754::traits<float64_t>::exponentBias;
                uint32_t shiftAmount = abs(expDiff);

                // so lhsNormMantissa always holds mantissa of number with greater exponent
                if (expDiff < 0)
                    swap<int64_t>(lhsNormMantissa, rhsNormMantissa);

                rhsNormMantissa >>= shiftAmount;

                int64_t resultMantissa = lhsNormMantissa + rhsNormMantissa;

                resultMantissa >>= 9;

                const uint64_t resultSign = uint64_t((lhsSign && rhsSign) || (bit_cast<uint64_t>(resultMantissa) & (lhsSign << 63))) << 63;
                uint64_t resultBiasedExp = uint64_t(exp) + ieee754::traits<float64_t>::exponentBias;

                resultMantissa = abs(resultMantissa);

                if (resultMantissa & 1ull << 53)
                {
                    ++resultBiasedExp;
                    resultMantissa >>= 1;
                }

                // TODO: better implementation with no loop
                while (resultMantissa < (1ull << 52))
                {
                    --resultBiasedExp;
                    resultMantissa <<= 1;
                }

                resultMantissa &= ieee754::traits<float64_t>::mantissaMask;
                uint64_t output = impl::assembleFloat64(resultSign, uint64_t(resultBiasedExp) << ieee754::traits<float64_t>::mantissaBitCnt, abs(resultMantissa));
                return bit_cast<this_t>(output);
            }

            // not implemented
            if (!FlushDenormToZero)
                return bit_cast<this_t>(0xdeadbeefbadcaffeull);
        }

        emulated_float64_t operator+(float rhs)
        {
            return bit_cast<this_t >(data) + create(rhs);
        }

        emulated_float64_t operator-(emulated_float64_t rhs) NBL_CONST_MEMBER_FUNC
        {
            emulated_float64_t lhs = bit_cast<this_t >(data);
            emulated_float64_t rhsFlipped = rhs.flipSign();
            
            return lhs + rhsFlipped;
        }

        emulated_float64_t operator-(float rhs) NBL_CONST_MEMBER_FUNC
        {
            return bit_cast<this_t >(data) - create(rhs);
        }

        emulated_float64_t operator*(emulated_float64_t rhs) NBL_CONST_MEMBER_FUNC
        {
            if(FlushDenormToZero)
            {
                emulated_float64_t retval = this_t::create(0ull);

                uint64_t lhsSign = data & ieee754::traits<float64_t>::signMask;
                uint64_t rhsSign = rhs.data & ieee754::traits<float64_t>::signMask;
                uint64_t lhsMantissa = ieee754::extractMantissa(data);
                uint64_t rhsMantissa = ieee754::extractMantissa(rhs.data);
                int lhsBiasedExp = ieee754::extractBiasedExponent(data);
                int rhsBiasedExp = ieee754::extractBiasedExponent(rhs.data);

                int exp = int(lhsBiasedExp + rhsBiasedExp) - ieee754::traits<float64_t>::exponentBias;
                uint64_t sign = (data ^ rhs.data) & ieee754::traits<float64_t>::signMask;
                if (!FastMath)
                {
                    if (lhsBiasedExp == ieee754::traits<float64_t>::specialValueExp)
                    {
                        if ((lhsMantissa != 0u) || ((rhsBiasedExp == ieee754::traits<float64_t>::specialValueExp) && (rhsMantissa != 0u)))
                            return bit_cast<this_t >(impl::propagateFloat64NaN(data, rhs.data));
                        if ((uint64_t(rhsBiasedExp) | rhsMantissa) == 0u)
                            return bit_cast<this_t >(ieee754::traits<float64_t>::quietNaN);

                        return bit_cast<this_t >(impl::assembleFloat64(sign, ieee754::traits<float64_t>::exponentMask, 0ull));
                    }
                    if (rhsBiasedExp == ieee754::traits<float64_t>::specialValueExp)
                    {
                        /* a cannot be NaN, but is b NaN? */
                        if (rhsMantissa != 0u)
#ifdef RELAXED_NAN_PROPAGATION
                            return rhs.data;
#else
                            return bit_cast<this_t >(impl::propagateFloat64NaN(data, rhs.data));
#endif
                        if ((uint64_t(lhsBiasedExp) | lhsMantissa) == 0u)
                            return bit_cast<this_t >(ieee754::traits<float64_t>::quietNaN);

                        return bit_cast<this_t >(sign | ieee754::traits<float64_t>::exponentMask);
                    }
                    if (lhsBiasedExp == 0)
                    {
                        if (lhsMantissa == 0u)
                            return bit_cast<this_t >(sign);
                        impl::normalizeFloat64Subnormal(lhsMantissa, lhsBiasedExp, lhsMantissa);
                    }
                    if (rhsBiasedExp == 0)
                    {
                        if (rhsMantissa == 0u)
                            return bit_cast<this_t >(sign);
                        impl::normalizeFloat64Subnormal(rhsMantissa, rhsBiasedExp, rhsMantissa);
                    }
                }

                const uint64_t hi_l = (lhsMantissa >> 21) | (1ull << 31);
                const uint64_t lo_l = lhsMantissa & ((1ull << 21) - 1);
                const uint64_t hi_r = (rhsMantissa >> 21) | (1ull << 31);
                const uint64_t lo_r = rhsMantissa & ((1ull << 21) - 1);

                //const uint64_t RoundToNearest = (1ull << 31) - 1;
                uint64_t newPseudoMantissa = ((hi_l * hi_r) >> 10) + ((hi_l * lo_r + lo_l * hi_r/* + RoundToNearest*/) >> 31);

                if (newPseudoMantissa & (0x1ull << 53))
                {
                    newPseudoMantissa >>= 1;
                    ++exp;
                }
                newPseudoMantissa &= (ieee754::traits<float64_t>::mantissaMask);

                return bit_cast<this_t >(impl::assembleFloat64(sign, uint64_t(exp) << ieee754::traits<float64_t>::mantissaBitCnt, newPseudoMantissa));
            }
            else
            {
                //static_assert(false, "not implemented yet");
                return bit_cast<this_t >(0xdeadbeefbadcaffeull);
            }
        }

        emulated_float64_t operator*(float rhs)
        {
            return bit_cast<this_t >(data) * create(rhs);
        }

        /*this_t reciprocal(uint64_t x)
        {
            using ThisType = this_t;
            ThisType output = ThisType::bit_cast<this_t >((0xbfcdd6a18f6a6f52ULL - x) >> 1);
            output = output * output;
            return output;
        }*/

        emulated_float64_t operator/(const emulated_float64_t rhs) NBL_CONST_MEMBER_FUNC
        {
            if (FlushDenormToZero)
            {
                //return this_t::bit_cast<this_t >(data) * reciprocal(rhs.data);

                if (!FastMath && (tgmath::isnan<uint64_t>(data) || tgmath::isnan<uint64_t>(rhs.data)))
                    return bit_cast<this_t>(ieee754::traits<float64_t>::quietNaN);

                const uint64_t sign = (data ^ rhs.data) & ieee754::traits<float64_t>::signMask;

                if (!FastMath && impl::isZero(rhs.data))
                    return bit_cast<this_t>(ieee754::traits<float64_t>::inf | sign);

                if (!FastMath && impl::areBothInfinity(data, rhs.data))
                    return bit_cast<this_t>(ieee754::traits<float64_t>::quietNaN);

                if (!FastMath && tgmath::isInf(data))
                    return bit_cast<this_t>(ieee754::traits<float64_t>::inf | sign);

                if (!FastMath && tgmath::isInf(rhs.data))
                    return bit_cast<this_t>(0ull | sign);


                const uint64_t lhsRealMantissa = (ieee754::extractMantissa(data) | (1ull << ieee754::traits<float64_t>::mantissaBitCnt));
                const uint64_t rhsRealMantissa = ieee754::extractMantissa(rhs.data) | (1ull << ieee754::traits<float64_t>::mantissaBitCnt);

                int exp = ieee754::extractExponent(data) - ieee754::extractExponent(rhs.data) + ieee754::traits<float64_t>::exponentBias;

                uint64_t2 lhsMantissaShifted = impl::shiftMantissaLeftBy53(lhsRealMantissa);
                uint64_t mantissa = impl::divmod128by64(lhsMantissaShifted.x, lhsMantissaShifted.y, rhsRealMantissa);

                while (mantissa < (1ull << 52))
                {
                    mantissa <<= 1;
                    exp--;
                }

                mantissa &= ieee754::traits<float64_t>::mantissaMask;

                return bit_cast<this_t >(impl::assembleFloat64(sign, uint64_t(exp) << ieee754::traits<float64_t>::mantissaBitCnt, mantissa));
            }
            else
            {
                //static_assert(false, "not implemented yet");
                return bit_cast<this_t >(0xdeadbeefbadcaffeull);
            }
        }

        // relational operators
        // TODO: should `FlushDenormToZero` affect relational operators?
        bool operator==(this_t rhs) NBL_CONST_MEMBER_FUNC
        {
            if (!FastMath && (tgmath::isnan<uint64_t>(data) || tgmath::isnan<uint64_t>(rhs.data)))
                return false;
            // TODO: i'm not sure about this one
            if (!FastMath && impl::areBothZero(data, rhs.data))
                return true;

            const emulated_float64_t xored = bit_cast<this_t >(data ^ rhs.data);
            // TODO: check what fast math returns for -0 == 0
            if ((xored.data & 0x7FFFFFFFFFFFFFFFull) == 0ull)
                return true;

            return !(xored.data);
        }
        bool operator!=(emulated_float64_t rhs) NBL_CONST_MEMBER_FUNC
        {
            if (!FastMath && (tgmath::isnan<uint64_t>(data) || tgmath::isnan<uint64_t>(rhs.data)))
                return false;

            return !(bit_cast<this_t >(data) == rhs);
        }
        bool operator<(emulated_float64_t rhs) NBL_CONST_MEMBER_FUNC
        {
            if (!FastMath && (tgmath::isnan<uint64_t>(data) || tgmath::isnan<uint64_t>(rhs.data)))
                return false;
            if (!FastMath && impl::areBothSameSignInfinity(data, rhs.data))
                return false;
            if (!FastMath && impl::areBothZero(data, rhs.data))
                return false;

            const uint64_t lhsSign = ieee754::extractSign(data);
            const uint64_t rhsSign = ieee754::extractSign(rhs.data);

            // flip bits of negative numbers and flip signs of all numbers
            uint64_t lhsFlipped = data ^ ((0x7FFFFFFFFFFFFFFFull * lhsSign) | ieee754::traits<float64_t>::signMask);
            uint64_t rhsFlipped = rhs.data ^ ((0x7FFFFFFFFFFFFFFFull * rhsSign) | ieee754::traits<float64_t>::signMask);

            uint64_t diffBits = lhsFlipped ^ rhsFlipped;

            return (lhsFlipped & diffBits) < (rhsFlipped & diffBits);
        }
        bool operator>(emulated_float64_t rhs) NBL_CONST_MEMBER_FUNC
        {
            if (!FastMath && (tgmath::isnan<uint64_t>(data) || tgmath::isnan<uint64_t>(rhs.data)))
                return false;
            if (!FastMath && impl::areBothSameSignInfinity(data, rhs.data))
                return false;
            if (!FastMath && impl::areBothZero(data, rhs.data))
                return false;

            const uint64_t lhsSign = ieee754::extractSign(data);
            const uint64_t rhsSign = ieee754::extractSign(rhs.data);

            // flip bits of negative numbers and flip signs of all numbers
            uint64_t lhsFlipped = data ^ ((0x7FFFFFFFFFFFFFFFull * lhsSign) | ieee754::traits<float64_t>::signMask);
            uint64_t rhsFlipped = rhs.data ^ ((0x7FFFFFFFFFFFFFFFull * rhsSign) | ieee754::traits<float64_t>::signMask);

            uint64_t diffBits = lhsFlipped ^ rhsFlipped;

            return (lhsFlipped & diffBits) > (rhsFlipped & diffBits);
        }
        bool operator<=(emulated_float64_t rhs) NBL_CONST_MEMBER_FUNC 
        { 
            if (!FastMath && (tgmath::isnan<uint64_t>(data) || tgmath::isnan<uint64_t>(rhs.data)))
                return false;

            return !(bit_cast<this_t >(data) > bit_cast<this_t >(rhs.data));
        }
        bool operator>=(emulated_float64_t rhs)
        {
            if (!FastMath && (tgmath::isnan<uint64_t>(data) || tgmath::isnan<uint64_t>(rhs.data)))
                return false;

            return !(bit_cast<this_t >(data) < bit_cast<this_t >(rhs.data));
        }

        //logical operators
        bool operator&&(emulated_float64_t rhs) NBL_CONST_MEMBER_FUNC { return bool(data) && bool(rhs.data); }
        bool operator||(emulated_float64_t rhs) NBL_CONST_MEMBER_FUNC { return bool(data) || bool(rhs.data); }
        bool operator!() NBL_CONST_MEMBER_FUNC { return !bool(data); }

        emulated_float64_t flipSign()
        {
            return bit_cast<this_t >(data ^ ieee754::traits<float64_t>::signMask);
        }
        
        bool isNaN()
        {
            return tgmath::isnan(data);
        }

        NBL_CONSTEXPR_STATIC_INLINE bool supportsFastMath()
        {
            return FastMath;
        }
    };

#define IMPLEMENT_IEEE754_FUNC_SPEC_FOR_EMULATED_F64_TYPE(...) \
template<>\
struct traits_base<__VA_ARGS__ >\
{\
    NBL_CONSTEXPR_STATIC_INLINE int16_t exponentBitCnt = 11;\
    NBL_CONSTEXPR_STATIC_INLINE int16_t mantissaBitCnt = 52;\
};\
template<>\
inline uint32_t extractBiasedExponent(__VA_ARGS__ x)\
{\
    return extractBiasedExponent<uint64_t>(x.data);\
}\
\
template<>\
inline int extractExponent(__VA_ARGS__ x)\
{\
    return extractExponent(x.data);\
}\
\
template<>\
NBL_CONSTEXPR_INLINE_FUNC __VA_ARGS__ replaceBiasedExponent(__VA_ARGS__ x, typename unsigned_integer_of_size<sizeof(__VA_ARGS__)>::type biasedExp)\
{\
    return __VA_ARGS__(replaceBiasedExponent(x.data, biasedExp));\
}\
\
template <>\
NBL_CONSTEXPR_INLINE_FUNC __VA_ARGS__ fastMulExp2(__VA_ARGS__ x, int n)\
{\
    return __VA_ARGS__(replaceBiasedExponent(x.data, extractBiasedExponent(x) + uint32_t(n)));\
}\
\
template <>\
NBL_CONSTEXPR_INLINE_FUNC unsigned_integer_of_size<sizeof(__VA_ARGS__)>::type extractMantissa(__VA_ARGS__ x)\
{\
    return extractMantissa(x.data);\
}\
\
template <>\
NBL_CONSTEXPR_INLINE_FUNC uint64_t extractNormalizeMantissa(__VA_ARGS__ x)\
{\
    return extractNormalizeMantissa(x.data);\
}\
\

#define DEFINE_BIT_CAST_SPEC(...)\
template<>\
NBL_CONSTEXPR_FUNC __VA_ARGS__ bit_cast<__VA_ARGS__, uint64_t>(NBL_CONST_REF_ARG(uint64_t) val)\
{\
__VA_ARGS__ output; \
output.data = val; \
\
return output; \
}\
\

namespace impl
{

template<typename To, bool FastMath, bool FlushDenormToZero>
struct static_cast_helper<To,emulated_float64_t<FastMath,FlushDenormToZero>,void>
{
    // TODO:
    // static_assert(is_arithmetic<To>::value);

    using From = emulated_float64_t<FastMath,FlushDenormToZero>;

    // TODO: test
    static inline To cast(From v)
    {
        using ToAsFloat = typename float_of_size<sizeof(To)>::type;
        using ToAsUint = typename unsigned_integer_of_size<sizeof(To)>::type;


        if (is_same_v<To, float64_t>)
            return To(bit_cast<float64_t>(v.data));

        if (is_floating_point<To>::value)
        {

            const int exponent = ieee754::extractExponent(v.data);
            if (!From::supportsFastMath())
            {
                if (exponent > ieee754::traits<ToAsFloat>::exponentMax)
                    return bit_cast<To>(ieee754::traits<ToAsFloat>::inf);
                if (exponent < ieee754::traits<ToAsFloat>::exponentMin)
                    return -bit_cast<To>(ieee754::traits<ToAsFloat>::inf);
                if (tgmath::isnan(v.data))
                    return bit_cast<To>(ieee754::traits<ToAsFloat>::quietNaN);
            }


            const uint32_t toBitSize = sizeof(To) * 8;
            const ToAsUint sign = ToAsUint(ieee754::extractSign(v.data) << (toBitSize - 1));
            const ToAsUint biasedExponent = ToAsUint(exponent + ieee754::traits<ToAsFloat>::exponentBias) << ieee754::traits<ToAsFloat>::mantissaBitCnt;
            const ToAsUint mantissa = ToAsUint(v.data >> (ieee754::traits<float64_t>::mantissaBitCnt - ieee754::traits<ToAsFloat>::mantissaBitCnt)) & ieee754::traits<ToAsFloat>::mantissaMask;

            return bit_cast<ToAsFloat>(sign | biasedExponent | mantissa);
        }

        // NOTE: casting from negative float to unsigned int is an UB, function will return abs value in this case
        if (is_integral<To>::value)
        {
            const int exponent = ieee754::extractExponent(v.data);
            if (exponent < 0)
                return 0;

            uint64_t unsignedOutput = ieee754::extractMantissa(v.data) & 1ull << ieee754::traits<float64_t>::mantissaBitCnt;
            const int shiftAmount = exponent - int(ieee754::traits<float64_t>::mantissaBitCnt);

            if (shiftAmount < 0)
                unsignedOutput <<= -shiftAmount;
            else
                unsignedOutput >>= shiftAmount;

            if (is_signed<To>::value)
            {
                int64_t signedOutput64 = unsignedOutput & ((1ull << 63) - 1);
                To signedOutput = To(signedOutput64);
                if (ieee754::extractSignPreserveBitPattern(v.data) != 0)
                    signedOutput = -signedOutput;

                return signedOutput;
            }

            return To(unsignedOutput);
        }

        // assert(false);
        return To(0xdeadbeefbadcaffeull);
    }
};

template<bool FastMath, bool FlushDenormToZero>
struct static_cast_helper<emulated_float64_t<FastMath, FlushDenormToZero>, float32_t, void>
{
    using To = emulated_float64_t<FastMath, FlushDenormToZero>;

    static inline To cast(float32_t v)
    {
        return To::create(v);
    }
};

template<bool FastMath, bool FlushDenormToZero>
struct static_cast_helper<emulated_float64_t<FastMath, FlushDenormToZero>, float64_t, void>
{
    using To = emulated_float64_t<FastMath, FlushDenormToZero>;

    static inline To cast(float64_t v)
    {
        return To::create(v);
    }
};

template<bool FastMath, bool FlushDenormToZero>
struct static_cast_helper<emulated_float64_t<FastMath, FlushDenormToZero>, uint32_t, void>
{
    using To = emulated_float64_t<FastMath, FlushDenormToZero>;

    static inline To cast(uint32_t v)
    {
        return To::create(v);
    }
};

template<bool FastMath, bool FlushDenormToZero>
struct static_cast_helper<emulated_float64_t<FastMath, FlushDenormToZero>, uint64_t, void>
{
    using To = emulated_float64_t<FastMath, FlushDenormToZero>;

    static inline To cast(uint64_t v)
    {
        return To::create(v);
    }
};

template<bool FastMath, bool FlushDenormToZero>
struct static_cast_helper<emulated_float64_t<FastMath, FlushDenormToZero>, emulated_float64_t<FastMath, FlushDenormToZero>, void>
{
    static inline emulated_float64_t<FastMath, FlushDenormToZero> cast(emulated_float64_t<FastMath, FlushDenormToZero> v)
    {
        return v;
    }
};

}

DEFINE_BIT_CAST_SPEC(emulated_float64_t<true, true>);
DEFINE_BIT_CAST_SPEC(emulated_float64_t<false, false>);
DEFINE_BIT_CAST_SPEC(emulated_float64_t<true, false>);
DEFINE_BIT_CAST_SPEC(emulated_float64_t<false, true>);

//template<bool FastMath, bool FlushDenormToZero>
//struct is_floating_point<emulated_float64_t<FastMath, FlushDenormToZero> > : bool_constant<true> {};

namespace ieee754
{
IMPLEMENT_IEEE754_FUNC_SPEC_FOR_EMULATED_F64_TYPE(emulated_float64_t<true, true>);
IMPLEMENT_IEEE754_FUNC_SPEC_FOR_EMULATED_F64_TYPE(emulated_float64_t<false, false>);
IMPLEMENT_IEEE754_FUNC_SPEC_FOR_EMULATED_F64_TYPE(emulated_float64_t<true, false>);
IMPLEMENT_IEEE754_FUNC_SPEC_FOR_EMULATED_F64_TYPE(emulated_float64_t<false, true>);
}

}
}

#undef FLOAT_ROUND_NEAREST_EVEN
#undef FLOAT_ROUND_TO_ZERO
#undef FLOAT_ROUND_DOWN
#undef FLOAT_ROUND_UP
#undef FLOAT_ROUNDING_MODE

#undef IMPLEMENT_IEEE754_FUNC_SPEC_FOR_EMULATED_F64_TYPE
#undef DEFINE_BIT_CAST_SPEC

#endif
