#ifndef _NBL_BUILTIN_HLSL_EMULATED_FLOAT64_T_INCLUDED_
#define _NBL_BUILTIN_HLSL_EMULATED_FLOAT64_T_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/ieee754.hlsl>
#include <nbl/builtin/hlsl/algorithm.hlsl>
#include <nbl/builtin/hlsl/tgmath.hlsl>

// TODO: when it will be possible, use this unions wherever they fit:
/*
* union Mantissa
* {
*   struct
*   {
*       uint32_t highBits;
*       uint64_t lowBits;
*   };
*
*   uint32_t2 packed;
* };
*
*/

/*
* union Mantissa
* {
*   struct
*   {
*       uint64_t lhs;
*       uint64_t rhs;
*   };
*
*   uint32_t4 packed;
* };
*
*/
   
#define FLOAT_ROUND_NEAREST_EVEN    0
#define FLOAT_ROUND_TO_ZERO         1
#define FLOAT_ROUND_DOWN            2
#define FLOAT_ROUND_UP              3
#define FLOAT_ROUNDING_MODE         FLOAT_ROUND_NEAREST_EVEN

namespace nbl
{
namespace hlsl
{
namespace impl
{
    template <typename T>
    uint64_t promoteToUint64(T val)
    {
        using AsFloat = typename float_of_size<sizeof(T)>::type;
        uint64_t asUint = ieee754::impl::castToUintType(val);

        const uint64_t sign = (uint64_t(ieee754::traits<float64_t>::signMask) & asUint) << (sizeof(float64_t) - sizeof(T));
        const int64_t newExponent = ieee754::extractExponent(val) + ieee754::traits<float64_t>::exponentBias;

        const uint64_t exp = (uint64_t(ieee754::extractExponent(val)) + ieee754::traits<float64_t>::exponentBias) << (ieee754::traits<float64_t>::mantissaBitCnt);
        const uint64_t mantissa = (uint64_t(ieee754::traits<float64_t>::mantissaMask) & asUint) << (ieee754::traits<float64_t>::exponentBias - ieee754::traits<AsFloat>::mantissaBitCnt);

        return sign | exp | mantissa;
    };

    template<> uint64_t promoteToUint64(float64_t val) { return bit_cast<uint64_t, float64_t>(val); }

    nbl::hlsl::uint32_t2 umulExtended(uint32_t lhs, uint32_t rhs)
    {
        uint64_t product = uint64_t(lhs) * uint64_t(rhs);
        nbl::hlsl::uint32_t2 output;
        output.x = uint32_t((product & 0xFFFFFFFF00000000) >> 32);
        output.y = uint32_t(product & 0x00000000FFFFFFFFull);
        return output;
    }

    uint64_t propagateFloat64NaN(uint64_t lhs, uint64_t rhs)
    {
#if defined RELAXED_NAN_PROPAGATION
        return lhs | rhs;
#else
        const bool lhsIsNaN = isnan(bit_cast<float64_t>(lhs));
        const bool rhsIsNaN = isnan(bit_cast<float64_t>(rhs));
        lhs |= 0x0000000000080000ull;
        rhs |= 0x0000000000080000ull;

        return lerp(rhs, lerp(lhs, rhs, rhsIsNaN), lhsIsNaN);
#endif
    }

    uint64_t packFloat64(uint32_t zSign, int zExp, uint32_t zFrac0, uint32_t zFrac1)
    {
        nbl::hlsl::uint32_t2 z;

        z.x = zSign + (uint32_t(zExp) << 20) + zFrac0;
        z.y = zFrac1;

        uint64_t output = 0u;
        output |= (uint64_t(z.x) << 32) & 0xFFFFFFFF00000000ull;
        output |= uint64_t(z.y);
        return  output;
    }

    uint32_t2 packUint64(uint64_t val)
    {
        return uint32_t2((val & 0xFFFFFFFF00000000ull) >> 32, val & 0x00000000FFFFFFFFull);
    }

    uint64_t unpackUint64(uint32_t2 val)
    {
        return ((uint64_t(val.x) & 0x00000000FFFFFFFFull) << 32) | uint64_t(val.y);
    }

    nbl::hlsl::uint32_t2 add64(uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1)
    {
       nbl::hlsl::uint32_t2 output;
       output.y = a1 + b1;
       output.x = a0 + b0 + uint32_t(output.y < a1);

       return output;
    }


    nbl::hlsl::uint32_t2 sub64(uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1)
    {
        nbl::hlsl::uint32_t2 output;
        output.y = a1 - b1;
        output.x = a0 - b0 - uint32_t(a1 < b1);
        
        return output;
    }

    // TODO: test
    int countLeadingZeros32(uint32_t val)
    {
#ifndef __HLSL_VERSION
        return 31 - nbl::hlsl::findMSB(val);
#else
        return 31 - firstbithigh(val);
#endif
    }
    
    uint32_t2 shift64RightJamming(uint32_t2 val, int count)
    {
        uint32_t2 output;
        const int negCount = (-count) & 31;
    
        output.x = lerp(0u, val.x, count == 0);
        output.x = lerp(output.x, (val.x >> count), count < 32);
    
        output.y = uint32_t((val.x | val.y) != 0u); /* count >= 64 */
        uint32_t z1_lt64 = (val.x>>(count & 31)) | uint32_t(((val.x<<negCount) | val.y) != 0u);
        output.y = lerp(output.y, z1_lt64, count < 64);
        output.y = lerp(output.y, (val.x | uint32_t(val.y != 0u)), count == 32);
        uint32_t z1_lt32 = (val.x<<negCount) | (val.y>>count) | uint32_t((val.y<<negCount) != 0u);
        output.y = lerp(output.y, z1_lt32, count < 32);
        output.y = lerp(output.y, val.y, count == 0);
        
        return output;
    }
    
    
    uint32_t4 mul64to128(uint32_t4 mantissasPacked)
    {
        uint32_t4 output;
        uint32_t more1 = 0u;
        uint32_t more2 = 0u;
    
        // a0 = x
        // a1 = y
        // b0 = z
        // b1 = w

        uint32_t2 z2z3 = umulExtended(mantissasPacked.y, mantissasPacked.w);
        output.z = z2z3.x;
        output.w = z2z3.y;
        uint32_t2 z1more2 = umulExtended(mantissasPacked.y, mantissasPacked.z);
        output.y = z1more2.x;
        more2 = z1more2.y;
        uint32_t2 z1z2 = add64(output.y, more2, 0u, output.z);
        output.y = z1z2.x;
        output.z = z1z2.y;
        uint32_t2 z0more1 = umulExtended(mantissasPacked.x, mantissasPacked.z);
        output.x = z0more1.x;
        more1 = z0more1.y;
        uint32_t2 z0z1 = add64(output.x, more1, 0u, output.y);
        output.x = z0z1.x;
        output.y = z0z1.y;
        uint32_t2 more1more2 = umulExtended(mantissasPacked.x, mantissasPacked.w);
        more1 = more1more2.x;
        more2 = more1more2.y;
        uint32_t2 more1z2 = add64(more1, more2, 0u, output.z);
        more1 = more1z2.x;
        output.z = more1z2.y;
        uint32_t2 z0z12 = add64(output.x, output.y, 0u, more1);
        output.x = z0z12.x;
        output.y = z0z12.y;

        return output;
    }
    
    nbl::hlsl::uint32_t3 shift64ExtraRightJamming(uint32_t3 val, int count)
    {
        uint32_t3 output;
        output.x = 0u;
       
        int negCount = (-count) & 31;
    
        output.z = lerp(uint32_t(val.x != 0u), val.x, count == 64);
        output.z = lerp(output.z, val.x << negCount, count < 64);
        output.z = lerp(output.z, val.y << negCount, count < 32);
    
        output.y = lerp(0u, (val.x >> (count & 31)), count < 64);
        output.y = lerp(output.y, (val.x  << negCount) | (val.y >> count), count < 32);
    
        val.z = lerp(val.z | val.y, val.z, count < 32);
        output.x = lerp(output.x, val.x >> count, count < 32);
        output.z |= uint32_t(val.z != 0u);
    
        output.x = lerp(output.x, 0u, (count == 32));
        output.y = lerp(output.y, val.x, (count == 32));
        output.z = lerp(output.z, val.y, (count == 32));
        output.x = lerp(output.x, val.x, (count == 0));
        output.y = lerp(output.y, val.y, (count == 0));
        output.z = lerp(output.z, val.z, (count == 0));
       
        return output;
    }

    uint64_t shortShift64Left(uint64_t val, int count)
    {
        const uint32_t2 packed = packUint64(val);
        
        nbl::hlsl::uint32_t2 output;
        output.y = packed.y << count;
        // TODO: fix
        output.x = lerp((packed.x << count | (packed.y >> ((-count) & 31))), packed.x, count == 0);

        // y = 3092377600
        // x = 2119009566
        return unpackUint64(output);
    };

    uint64_t assembleFloat64(uint64_t signShifted, uint64_t expShifted, uint64_t mantissa)
    {
        return  signShifted + expShifted + mantissa;
    }
    
    uint64_t roundAndPackFloat64(uint64_t zSign, int zExp, uint32_t3 mantissaExtended)
    {
        bool roundNearestEven;
        bool increment;
    
        roundNearestEven = true;
        increment = int(mantissaExtended.z) < 0;
        if (!roundNearestEven) 
        {
            if (false) //(FLOAT_ROUNDING_MODE == FLOAT_ROUND_TO_ZERO)
            {
                increment = false;
            } 
            else
            {
                if (false) //(zSign != 0u)
                {
                    //increment = (FLOAT_ROUNDING_MODE == FLOAT_ROUND_DOWN) &&
                    //   (zFrac2 != 0u);
                }
                else
                {
                    //increment = (FLOAT_ROUNDING_MODE == FLOAT_ROUND_UP) &&
                    //   (zFrac2 != 0u);
                }
            }
        }
        if (0x7FD <= zExp)
        {
            if ((0x7FD < zExp) || ((zExp == 0x7FD) && (0x001FFFFFu == mantissaExtended.x && 0xFFFFFFFFu == mantissaExtended.y) && increment))
            {
                if (false) // ((FLOAT_ROUNDING_MODE == FLOAT_ROUND_TO_ZERO) ||
                // ((zSign != 0u) && (FLOAT_ROUNDING_MODE == FLOAT_ROUND_UP)) ||
                // ((zSign == 0u) && (FLOAT_ROUNDING_MODE == FLOAT_ROUND_DOWN)))
                {
                    return packFloat64(zSign, 0x7FE, 0x000FFFFFu, 0xFFFFFFFFu);
                }
                 
            return packFloat64(zSign, 0x7FF, 0u, 0u);
        }
        }
    
        if (zExp < 0)
        {
            mantissaExtended = shift64ExtraRightJamming(mantissaExtended, -zExp);
            zExp = 0;
              
            if (roundNearestEven)
            {
                increment = mantissaExtended.z < 0u;
            }
            else
            {
                if (zSign != 0u)
                {
                    increment = (FLOAT_ROUNDING_MODE == FLOAT_ROUND_DOWN) && (mantissaExtended.z != 0u);
                }
                else
                {
                    increment = (FLOAT_ROUNDING_MODE == FLOAT_ROUND_UP) && (mantissaExtended.z != 0u);
                }
            }
        }
    
        if (increment)
        {
            const uint64_t added = impl::unpackUint64(uint32_t2(mantissaExtended.xy)) + 1ull;
            mantissaExtended.xy = packUint64(added);
            mantissaExtended.y &= ~((mantissaExtended.z + uint32_t(mantissaExtended.z == 0u)) & uint32_t(roundNearestEven));
        }
        else
        {
            zExp = lerp(zExp, 0, (mantissaExtended.x | mantissaExtended.y) == 0u);
        }
       
        return assembleFloat64(zSign, uint64_t(zExp) << ieee754::traits<float64_t>::mantissaBitCnt, unpackUint64(mantissaExtended.xy));
    }
    
    uint64_t normalizeRoundAndPackFloat64(uint64_t sign, int exp, uint32_t frac0, uint32_t frac1)
    {
        int shiftCount;
        nbl::hlsl::uint32_t3 frac = nbl::hlsl::uint32_t3(frac0, frac1, 0u);
    
        if (frac.x == 0u)
        {
            exp -= 32;
            frac.x = frac.y;
            frac.y = 0u;
        }
    
        shiftCount = countLeadingZeros32(frac.x) - 11;
        if (0 <= shiftCount)
        {
            frac.xy = shortShift64Left(unpackUint64(frac.xy), shiftCount);
        }
        else
        {
            frac.xyz = shift64ExtraRightJamming(uint32_t3(frac.xy, 0), -shiftCount);
        }
        exp -= shiftCount;
        return roundAndPackFloat64(sign, exp, frac);
    }

    void normalizeFloat64Subnormal(uint64_t mantissa,
        NBL_REF_ARG(int) outExp,
        NBL_REF_ARG(uint64_t) outMantissa)
    {
        uint32_t2 mantissaPacked = packUint64(mantissa);
        int shiftCount;
        uint32_t2 temp;
        shiftCount = countLeadingZeros32(lerp(mantissaPacked.x, mantissaPacked.y, mantissaPacked.x == 0u)) - 11;
        outExp = lerp(1 - shiftCount, -shiftCount - 31, mantissaPacked.x == 0u);

        temp.x = lerp(mantissaPacked.y << shiftCount, mantissaPacked.y >> (-shiftCount), shiftCount < 0);
        temp.y = lerp(0u, mantissaPacked.y << (shiftCount & 31), shiftCount < 0);

        shortShift64Left(impl::unpackUint64(mantissaPacked), shiftCount);

        outMantissa = lerp(outMantissa, unpackUint64(temp), mantissaPacked.x == 0);
    }

    }

    struct emulated_float64_t
    {
        using storage_t = uint64_t;

        storage_t data;

        // constructors
        /*static emulated_float64_t create(uint16_t val)
        {
            return emulated_float64_t(bit_cast<uint64_t>(float64_t(val)));
        }*/

        static emulated_float64_t create(int32_t val)
        {
            return emulated_float64_t(bit_cast<uint64_t>(float64_t(val)));
        }

        static emulated_float64_t create(int64_t val)
        {
            return emulated_float64_t(bit_cast<uint64_t>(float64_t(val)));
        }

        static emulated_float64_t create(uint32_t val)
        {
            return emulated_float64_t(bit_cast<uint64_t>(float64_t(val)));
        }

        static emulated_float64_t create(uint64_t val)
        {
            return emulated_float64_t(bit_cast<uint64_t>(float64_t(val)));
        }

        static emulated_float64_t create(float64_t val)
        {
            return emulated_float64_t(bit_cast<uint64_t>(val));
        }
        
        // TODO: unresolved external symbol imath_half_to_float_table
        /*static emulated_float64_t create(float16_t val)
        {
            return emulated_float64_t(bit_cast<uint64_t>(float64_t(val)));
        }*/

        static emulated_float64_t create(float32_t val)
        {
            return emulated_float64_t(bit_cast<uint64_t>(float64_t(val)));
        }

        static emulated_float64_t createPreserveBitPattern(uint64_t val)
        {
            return emulated_float64_t(val);
        }

        // arithmetic operators
        emulated_float64_t operator+(const emulated_float64_t rhs)
        {
            emulated_float64_t retval = createPreserveBitPattern(0u);

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
                    nbl::hlsl::uint32_t2 frac;
                
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

        emulated_float64_t operator-(emulated_float64_t rhs)
        {
            emulated_float64_t lhs = createPreserveBitPattern(data);
            emulated_float64_t rhsFlipped = rhs.flipSign();
            
            return lhs + rhsFlipped;
        }

        emulated_float64_t operator*(emulated_float64_t rhs)
        {
            emulated_float64_t retval = emulated_float64_t::createPreserveBitPattern(0u);

            uint64_t lhsSign = data & ieee754::traits<float64_t>::signMask;
            uint64_t rhsSign = rhs.data & ieee754::traits<float64_t>::signMask;
            uint64_t lhsMantissa = ieee754::extractMantissa(data);
            uint64_t rhsMantissa = ieee754::extractMantissa(rhs.data);
            int lhsBiasedExp = ieee754::extractBiasedExponent(data);
            int rhsBiasedExp = ieee754::extractBiasedExponent(rhs.data);

            int exp = int(lhsBiasedExp + rhsBiasedExp) - 0x400;
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

            lhsMantissa |= 0x0010000000000000ull;
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

        // TODO
        emulated_float64_t operator/(const emulated_float64_t rhs)
        {
            return createPreserveBitPattern(0xdeadbeefbadcaffeull);
        }

        // relational operators
        bool operator==(emulated_float64_t rhs)
        {
            if (isnan(data) || isnan(rhs.data))
                return false;

            const emulated_float64_t xored = emulated_float64_t::createPreserveBitPattern(data ^ rhs.data);
            // TODO: check what fast math returns for -0 == 0
            if ((xored.data & 0x7FFFFFFFFFFFFFFFull) == 0ull)
                return true;

            return !(xored.data);
        }
        bool operator!=(emulated_float64_t rhs)
        {
            if (isnan(data) || isnan(rhs.data))
                return true;

            const emulated_float64_t xored = emulated_float64_t::createPreserveBitPattern(data ^ rhs.data);

            // TODO: check what fast math returns for -0 == 0
            if ((xored.data & 0x7FFFFFFFFFFFFFFFull) == 0ull)
                return false;

            return xored.data;
        }
        bool operator<(emulated_float64_t rhs)
        {
            const uint64_t lhsSign = ieee754::extractSign(data);
            const uint64_t rhsSign = ieee754::extractSign(rhs.data);

            // flip bits of negative numbers and flip signs of all numbers
            uint64_t lhsFlipped = data ^ ((0x7FFFFFFFFFFFFFFFull * lhsSign) | ieee754::traits<float64_t>::signMask);
            uint64_t rhsFlipped = rhs.data ^ ((0x7FFFFFFFFFFFFFFFull * rhsSign) | ieee754::traits<float64_t>::signMask);

            uint64_t diffBits = lhsFlipped ^ rhsFlipped;

            return (lhsFlipped & diffBits) < (rhsFlipped & diffBits);
        }
        bool operator>(emulated_float64_t rhs) 
        {
            const uint64_t lhsSign = ieee754::extractSign(data);
            const uint64_t rhsSign = ieee754::extractSign(rhs.data);

            // flip bits of negative numbers and flip signs of all numbers
            uint64_t lhsFlipped = data ^ ((0x7FFFFFFFFFFFFFFFull * lhsSign) | ieee754::traits<float64_t>::signMask);
            uint64_t rhsFlipped = rhs.data ^ ((0x7FFFFFFFFFFFFFFFull * rhsSign) | ieee754::traits<float64_t>::signMask);

            uint64_t diffBits = lhsFlipped ^ rhsFlipped;

            return (lhsFlipped & diffBits) > (rhsFlipped & diffBits);
        }
        bool operator<=(emulated_float64_t rhs) { return !(emulated_float64_t::createPreserveBitPattern(data) > emulated_float64_t::createPreserveBitPattern(rhs.data)); }
        bool operator>=(emulated_float64_t rhs) { return !(emulated_float64_t::createPreserveBitPattern(data) < emulated_float64_t::createPreserveBitPattern(rhs.data)); }

        //logical operators
        bool operator&&(emulated_float64_t rhs) { return bool(data) && bool(rhs.data); }
        bool operator||(emulated_float64_t rhs) { return bool(data) || bool(rhs.data); }
        bool operator!() { return !bool(data); }

        // OMITED OPERATORS
        //  - not implementing bitwise and modulo operators since floating point types doesn't support them
        //  - compound operator overload not supported in HLSL
        //  - access operators (dereference and addressof) not supported in HLSL
        
        // TODO: should modify self?
        emulated_float64_t flipSign()
        {
            return createPreserveBitPattern(data ^ ieee754::traits<float64_t>::signMask);
        }
        
        bool isNaN()
        {
            return isnan(bit_cast<float64_t>(data));
        }
    };

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
