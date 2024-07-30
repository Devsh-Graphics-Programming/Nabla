#ifndef _NBL_BUILTIN_HLSL_EMULATED_FLOAT64_T_IMPL_INCLUDED_
#define _NBL_BUILTIN_HLSL_EMULATED_FLOAT64_T_IMPL_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/ieee754.hlsl>
#include <nbl/builtin/hlsl/algorithm.hlsl>
#include <nbl/builtin/hlsl/tgmath.hlsl>

#define FLOAT_ROUND_NEAREST_EVEN    0
#define FLOAT_ROUND_TO_ZERO         1
#define FLOAT_ROUND_DOWN            2
#define FLOAT_ROUND_UP              3
#define FLOAT_ROUNDING_MODE         FLOAT_ROUND_NEAREST_EVEN

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

namespace nbl
{
namespace hlsl
{
namespace impl
{
    uint64_t2 shiftMantissaLeftBy52(uint64_t mantissa64)
    {
        uint64_t2 output;
        output.x = mantissa64 >> (64 - ieee754::traits<float64_t>::mantissaBitCnt);
        output.y = mantissa64 << ieee754::traits<float64_t>::mantissaBitCnt;

        return output;
    }

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

    uint32_t2 umulExtended(uint32_t lhs, uint32_t rhs)
    {
        uint64_t product = uint64_t(lhs) * uint64_t(rhs);
        uint32_t2 output;
        output.x = uint32_t((product & 0xFFFFFFFF00000000) >> 32);
        output.y = uint32_t(product & 0x00000000FFFFFFFFull);
        return output;
    }

    uint64_t propagateFloat64NaN(uint64_t lhs, uint64_t rhs)
    {
#if defined RELAXED_NAN_PROPAGATION
        return lhs | rhs;
#else

        lhs |= 0x0008000000000000ull;
        rhs |= 0x0008000000000000ull;
        return lerp(rhs, lerp(lhs, rhs, isnan(rhs)), isnan(lhs));
        return 0;
#endif
    }


    uint64_t packFloat64(uint32_t zSign, int zExp, uint32_t zFrac0, uint32_t zFrac1)
    {
        uint32_t2 z;

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

    uint32_t2 add64(uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1)
    {
       uint32_t2 output;
       output.y = a1 + b1;
       output.x = a0 + b0 + uint32_t(output.y < a1);

       return output;
    }


    uint32_t2 sub64(uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1)
    {
        uint32_t2 output;
        output.y = a1 - b1;
        output.x = a0 - b0 - uint32_t(a1 < b1);
        
        return output;
    }

    // TODO: test
    int countLeadingZeros32(uint32_t val)
    {
#ifndef __HLSL_VERSION
        return 31 - findMSB(val);
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
    
    uint32_t3 shift64ExtraRightJamming(uint32_t3 val, int count)
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
        
        uint32_t2 output;
        output.y = packed.y << count;
        // TODO: fix
        output.x = lerp((packed.x << count | (packed.y >> ((-count) & 31))), packed.x, count == 0);

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
        uint32_t3 frac = uint32_t3(frac0, frac1, 0u);
    
        if (frac.x == 0u)
        {
            exp -= 32;
            frac.x = frac.y;
            frac.y = 0u;
        }
    
        shiftCount = countLeadingZeros32(frac.x) - 11;
        if (0 <= shiftCount)
        {
            // TODO: this is packing and unpacking madness, fix it
            frac.xy = packUint64(shortShift64Left(unpackUint64(frac.xy), shiftCount));
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
    
    bool areBothSameSignInfinity(uint64_t lhs, uint64_t rhs)
    {
        lhs ^= ieee754::traits<float64_t>::signMask;
        rhs ^= ieee754::traits<float64_t>::signMask;
        
        bool output = lhs == rhs && ieee754::traits<float64_t>::inf;
        output = output && ((lhs & (~ieee754::traits<float64_t>::signMask)) == ieee754::traits<float64_t>::inf);

        return output;
    }

    bool areBothZero(uint64_t lhs, uint64_t rhs)
    {
        return ((lhs << 1) == 0ull) && ((rhs << 1) == 0ull);
    }

    bool areBothSameSignZero(uint64_t lhs, uint64_t rhs)
    {
        return ((lhs << 1) == 0ull) && (lhs == rhs);
    }

    // TODO: find more efficient algorithm
    uint64_t nlz64(uint64_t x)
    {
        static const uint64_t MASK = 1ull << 63;

        uint64_t counter = 0;

        while ((x & MASK) == 0)
        {
            x <<= 1;
            ++counter;
        }
        return counter;
    }

    uint64_t2 divmod128by64(const uint64_t u1, const uint64_t u0, uint64_t v)
    {
        const uint64_t b = 1ull << 32;
        uint64_t un1, un0, vn1, vn0, q1, q0, un32, un21, un10, rhat, left, right;
        uint64_t s;

        s = nlz64(v);
        v <<= s;
        vn1 = v >> 32;
        vn0 = v & 0xffffffff;

        if (s > 0)
        {
            un32 = (u1 << s) | (u0 >> (64 - s));
            un10 = u0 << s;
        }
        else
        {
            un32 = u1;
            un10 = u0;
        }

        un1 = un10 >> 32;
        un0 = un10 & 0xffffffff;

        q1 = un32 / vn1;
        rhat = un32 % vn1;

        left = q1 * vn0;
        right = (rhat << 32) + un1;
        while ((q1 >= b) || (left > right))
        {
            --q1;
            rhat += vn1;
            if (rhat < b)
            {
                left -= vn0;
                right = (rhat << 32) | un1;
            }
            break;
        }

        un21 = (un32 << 32) + (un1 - (q1 * v));

        q0 = un21 / vn1;
        rhat = un21 % vn1;

        left = q0 * vn0;
        right = (rhat << 32) | un0;
        while ((q0 >= b) || (left > right))
        {
            --q0;
            rhat += vn1;
            if (rhat < b)
            {
                left -= vn0;
                right = (rhat << 32) | un0;
                continue;
            }
            break;
        }

        uint64_t2 output;
        output.x = (q1 << 32) | q0; // quotient
        output.y = ((un21 << 32) + (un0 - (q0 * v))) >> s; // remainder

        return output;
    }
}
}
}
#endif