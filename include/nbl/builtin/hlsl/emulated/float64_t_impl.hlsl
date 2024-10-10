#ifndef _NBL_BUILTIN_HLSL_EMULATED_FLOAT64_T_IMPL_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_EMULATED_FLOAT64_T_IMPL_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/ieee754.hlsl>
#include <nbl/builtin/hlsl/algorithm.hlsl>
#include <nbl/builtin/hlsl/tgmath.hlsl>
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>

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
namespace emulated_float64_t_impl
{
NBL_CONSTEXPR_INLINE_FUNC uint64_t2 shiftMantissaLeftBy53(uint64_t mantissa64)
{
    uint64_t2 output;
    output.x = mantissa64 >> (64 - ieee754::traits<float64_t>::mantissaBitCnt);
    output.y = mantissa64 << (ieee754::traits<float64_t>::mantissaBitCnt);

    return output;
}

template<bool FlushDenormToZero>
inline uint64_t castFloat32ToStorageType(float32_t val)
{
    if (FlushDenormToZero)
    {
        const uint64_t sign = uint64_t(ieee754::extractSign(val)) << 63;
        if (tgmath::isInf(val))
            return ieee754::traits<float64_t>::inf | sign;
        uint32_t asUint = ieee754::impl::bitCastToUintType(val);
        const int f32BiasedExp = int(ieee754::extractBiasedExponent(val));
        if (f32BiasedExp == 0)
            return sign;
        const uint64_t biasedExp = uint64_t(f32BiasedExp - ieee754::traits<float32_t>::exponentBias + ieee754::traits<float64_t>::exponentBias) << (ieee754::traits<float64_t>::mantissaBitCnt);
        const uint64_t mantissa = (uint64_t(ieee754::traits<float32_t>::mantissaMask) & asUint) << (ieee754::traits<float64_t>::mantissaBitCnt - ieee754::traits<float32_t>::mantissaBitCnt);

        return sign | biasedExp | mantissa;
    }
    else
    {
        // static_assert(false);
        return 0xdeadbeefbadcaffeull;
    }
};

NBL_CONSTEXPR_INLINE_FUNC bool isZero(uint64_t val)
{
    return (val << 1) == 0;
}

// TODO: where do i move this function? also rename
template <typename Int>
inline int _findMSB(Int val)
{
    //static_assert(is_integral<Int>::value);
#ifndef __HLSL_VERSION
    return nbl::hlsl::findMSB(val);
#else
    return firstbithigh(val);
#endif
}

template <>
inline int _findMSB(uint64_t val)
{
#ifndef __HLSL_VERSION
    return nbl::hlsl::findMSB(val);
#else
    int msbHigh = firstbithigh(uint32_t(val >> 32));
    int msbLow = firstbithigh(uint32_t(val));
    return msbHigh != -1 ? msbHigh + 32 : msbLow;
#endif
}

inline uint64_t castToUint64WithFloat64BitPattern(uint64_t val)
{
    if (isZero(val))
        return val;

    int exp = _findMSB(val);
    uint64_t mantissa;

    int shiftCnt = 52 - exp;
    if (shiftCnt >= 0)
    {
        mantissa = val << shiftCnt;
    }
    else
    {
        const int shiftCntAbs = -shiftCnt;
        uint64_t roundingBit = 1ull << (shiftCnt - 1);
        uint64_t stickyBitMask = roundingBit - 1;
        uint64_t stickyBit = val & stickyBitMask;

        mantissa = val >> shiftCntAbs;

        if ((val & roundingBit) && (!stickyBit))
        {
            bool isEven = mantissa & 1;
            if (!isEven)
                mantissa++;
        }
        else if ((val & roundingBit) && (stickyBit || (mantissa & 1)))
            val += roundingBit;

        //val += (1ull << (shiftCnt)) - 1;
        //mantissa = val >> shiftCntAbs;

        if (mantissa & 1ull << 53)
        {
            mantissa >>= 1;
            exp++;
        }
    }
    mantissa &= ieee754::traits<float64_t>::mantissaMask;
    const uint64_t biasedExp = uint64_t(ieee754::traits<float64_t>::exponentBias + exp) << ieee754::traits<float64_t>::mantissaBitCnt;

    return biasedExp | mantissa;
};

inline uint64_t castToUint64WithFloat64BitPattern(int64_t val)
{
    const uint64_t sign = val & ieee754::traits<float64_t>::signMask;
    const uint64_t absVal = uint64_t(abs(val));
    return sign | castToUint64WithFloat64BitPattern(absVal);
};

NBL_CONSTEXPR_INLINE_FUNC uint32_t2 umulExtended(uint32_t lhs, uint32_t rhs)
{
    uint64_t product = uint64_t(lhs) * uint64_t(rhs);
    uint32_t2 output;
    output.x = uint32_t((product & 0xFFFFFFFF00000000) >> 32);
    output.y = uint32_t(product & 0x00000000FFFFFFFFull);
    return output;
}

NBL_CONSTEXPR_INLINE_FUNC uint64_t flushDenormToZero(uint64_t extractedBiasedExponent, uint64_t value)
{
    return extractedBiasedExponent ? value : ieee754::extractSignPreserveBitPattern(value);
}

NBL_CONSTEXPR_INLINE_FUNC uint64_t assembleFloat64(uint64_t signShifted, uint64_t expShifted, uint64_t mantissa)
{
    return  signShifted | expShifted | mantissa;
}

NBL_CONSTEXPR_INLINE_FUNC bool areBothInfinity(uint64_t lhs, uint64_t rhs)
{
    lhs &= ~ieee754::traits<float64_t>::signMask;
    rhs &= ~ieee754::traits<float64_t>::signMask;

    return lhs == rhs && lhs == ieee754::traits<float64_t>::inf;
}

NBL_CONSTEXPR_INLINE_FUNC bool areBothZero(uint64_t lhs, uint64_t rhs)
{
    return ((lhs << 1) == 0ull) && ((rhs << 1) == 0ull);
}

NBL_CONSTEXPR_INLINE_FUNC bool areBothSameSignZero(uint64_t lhs, uint64_t rhs)
{
    return ((lhs << 1) == 0ull) && (lhs == rhs);
}

// TODO: remove, use Newton-Raphson instead
// returns pair of quotient and remainder
inline uint64_t divmod128by64(const uint64_t dividentHigh, const uint64_t dividentLow, uint64_t divisor)
{
    const uint64_t b = 1ull << 32;
    uint64_t un1, un0, vn1, vn0, q1, q0, un32, un21, un10, rhat, left, right;
    uint64_t s;

    s = countl_zero(divisor);
    divisor <<= s;
    vn1 = divisor >> 32;
    vn0 = divisor & 0xFFFFFFFF;

    if (s > 0)
    {
        un32 = (dividentHigh << s) | (dividentLow >> (64 - s));
        un10 = dividentLow << s;
    }
    else
    {
        un32 = dividentHigh;
        un10 = dividentLow;
    }

    un1 = un10 >> 32;
    un0 = un10 & 0xFFFFFFFF;

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

    un21 = (un32 << 32) + (un1 - (q1 * divisor));

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

    return (q1 << 32) | q0;
}

struct uint128_t
{
    uint64_t highBits;
    uint64_t lowBits;
};

inline uint64_t subMantissas128NormalizeResult(const uint64_t greaterNumberMantissa, const uint64_t lesserNumberMantissaHigh, const uint64_t lesserNumberMantissaLow, NBL_REF_ARG(int) resultExp)
{
    uint64_t greaterHigh, greaterLow;
    greaterHigh = greaterNumberMantissa;
    greaterLow = 0ull;

    uint64_t diffHigh, diffLow;
    diffHigh = greaterHigh - lesserNumberMantissaHigh;
    diffLow = greaterLow - lesserNumberMantissaLow;

    if (lesserNumberMantissaLow > greaterLow)
        --diffHigh;

    int msbIdx = _findMSB(diffHigh);
    if (msbIdx == -1)
    {
        msbIdx = _findMSB(diffLow);
        if (msbIdx == -1)
            return 0ull;
    }
    else
    {
        msbIdx += 64;
    }

    static const int TargetMSB = 52 + 64;
    int shiftAmount = msbIdx - TargetMSB;
    resultExp += shiftAmount;

    if (shiftAmount < 0)
    {
        shiftAmount = -shiftAmount;
        diffHigh <<= shiftAmount;
        const uint64_t shiftedOutBits = diffLow >> (64 - shiftAmount);
        diffHigh |= shiftedOutBits;
    }

    return diffHigh;
}

}
}
}
#endif