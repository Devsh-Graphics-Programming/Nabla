#ifndef _NBL_BUILTIN_HLSL_EMULATED_FLOAT64_T_IMPL_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_EMULATED_FLOAT64_T_IMPL_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/ieee754.hlsl>
#include <nbl/builtin/hlsl/algorithm.hlsl>
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
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
        if (hlsl::isinf(val))
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
    return (val << 1) == 0ull;
}

template<typename T>
inline uint64_t reinterpretAsFloat64BitPattern(T);

template<>
inline uint64_t reinterpretAsFloat64BitPattern<uint64_t>(uint64_t val)
{
    if (isZero(val))
        return val;

    int exp = findMSB(val);
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

template<>
inline uint64_t reinterpretAsFloat64BitPattern<int64_t>(int64_t val)
{
    const uint64_t sign = val & ieee754::traits<float64_t>::signMask;
    const uint64_t absVal = uint64_t(abs(val));
    return sign | reinterpretAsFloat64BitPattern(absVal);
};

NBL_CONSTEXPR_INLINE_FUNC uint64_t flushDenormToZero(uint64_t value)
{
    const uint64_t biasBits = value & ieee754::traits<float64_t>::exponentMask;
    return biasBits ? value : (value & ieee754::traits<float64_t>::signMask);
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
    return !bool((lhs | rhs) << 1);
}

NBL_CONSTEXPR_INLINE_FUNC bool areBothSameSignZero(uint64_t lhs, uint64_t rhs)
{
    return !bool((lhs) << 1) && (lhs == rhs);
}

template<bool FastMath, typename Op>
NBL_CONSTEXPR_INLINE_FUNC bool operatorLessAndGreaterCommonImplementation(uint64_t lhs, uint64_t rhs)
{
    if (!FastMath)
    {
        if (cpp_compat_intrinsics_impl::isnan_uint_impl<uint64_t>(lhs) || cpp_compat_intrinsics_impl::isnan_uint_impl<uint64_t>(rhs))
            return false;
        if (emulated_float64_t_impl::areBothZero(lhs, rhs))
            return false;
    }

    const uint64_t lhsSign = ieee754::extractSignPreserveBitPattern(lhs);
    const uint64_t rhsSign = ieee754::extractSignPreserveBitPattern(rhs);

    // flip bits of negative numbers and flip signs of all numbers
    if (lhsSign)
        lhs ^= 0x7FFFFFFFFFFFFFFFull;
    if (rhsSign)
        rhs ^= 0x7FFFFFFFFFFFFFFFull;
    lhs ^= ieee754::traits<float64_t>::signMask;
    rhs ^= ieee754::traits<float64_t>::signMask;

    Op compare;
    return compare(lhs, rhs);
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

    int msbIdx = findMSB(diffHigh);
    if (msbIdx == -1)
    {
        msbIdx = findMSB(diffLow);
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