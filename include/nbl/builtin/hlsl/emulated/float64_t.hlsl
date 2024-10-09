#ifndef _NBL_BUILTIN_HLSL_EMULATED_FLOAT64_T_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_EMULATED_FLOAT64_T_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/emulated/float64_t_impl.hlsl>

namespace nbl
{
namespace hlsl
{
    /*enum E_ROUNDING_MODE
    {
        FLOAT_ROUND_NEAREST_EVEN,
        FLOAT_ROUND_TO_ZERO,
        FLOAT_ROUND_DOWN,
        FLOAT_ROUND_UP
    };*/

    // currently only FLOAT_ROUND_TO_ZERO is supported, cannot implement partial specialization in this case due to dxc bug https://github.com/microsoft/DirectXShaderCompiler/issues/5563
    // TODO: partial specializations with new template parameter `E_ROUNDING_MODE RoundingMode`
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
            return bit_cast<this_t>(emulated_float64_t_impl::castToUint64WithFloat64BitPattern(int64_t(val)));
        }

        NBL_CONSTEXPR_STATIC_INLINE this_t create(int64_t val)
        {
            return bit_cast<this_t>(emulated_float64_t_impl::castToUint64WithFloat64BitPattern(val));
        }

        NBL_CONSTEXPR_STATIC_INLINE this_t create(uint32_t val)
        {
            return bit_cast<this_t>(emulated_float64_t_impl::castToUint64WithFloat64BitPattern(uint64_t(val)));
        }

        NBL_CONSTEXPR_STATIC_INLINE this_t create(uint64_t val)
        {
            return bit_cast<this_t>(emulated_float64_t_impl::castToUint64WithFloat64BitPattern(val));
        }

        NBL_CONSTEXPR_STATIC_INLINE this_t create(float32_t val)
        {
            this_t output;
            output.data = emulated_float64_t_impl::castFloat32ToStorageType<FlushDenormToZero>(val);
            return output;
        }

        NBL_CONSTEXPR_STATIC_INLINE this_t create(float64_t val)
        {
#ifdef __HLSL_VERSION
            emulated_float64_t retval;
            uint32_t lo, hi;
            asuint(val, lo, hi);
            retval.data = (uint64_t(hi) << 32) | uint64_t(lo);
            return retval;
#else
            return bit_cast<this_t>(val);
#endif
        }

        // arithmetic operators
        this_t operator+(const emulated_float64_t rhs) NBL_CONST_MEMBER_FUNC
        {
            if (FlushDenormToZero)
            {
                if(FastMath)
                {
                    if (tgmath::isNaN(data) || tgmath::isNaN(rhs.data))
                        return bit_cast<this_t>(ieee754::traits<float64_t>::quietNaN);

                    if (emulated_float64_t_impl::areBothInfinity(data, rhs.data))
                    {
                        uint64_t lhsSign = data & ieee754::traits<float64_t>::signMask;
                        uint64_t rhsSign = rhs.data & ieee754::traits<float64_t>::signMask;

                        if (lhsSign == rhsSign)
                            return bit_cast<this_t>(ieee754::traits<float64_t>::inf | lhsSign);
                        else if (lhsSign || rhsSign)
                            return bit_cast<this_t>(ieee754::traits<float64_t>::quietNaN | ieee754::traits<float64_t>::signMask);
                    }

                    if (tgmath::isInf(data))
                        return bit_cast<this_t>(data);

                    if (tgmath::isInf(rhs.data))
                        return bit_cast<this_t>(rhs.data);
                }

                const int lhsBiasedExp = ieee754::extractBiasedExponent(data);
                const int rhsBiasedExp = ieee754::extractBiasedExponent(rhs.data);

                uint64_t lhsData = emulated_float64_t_impl::flushDenormToZero(lhsBiasedExp, data);
                uint64_t rhsData = emulated_float64_t_impl::flushDenormToZero(rhsBiasedExp, rhs.data);

                uint64_t lhsSign = ieee754::extractSignPreserveBitPattern(lhsData);
                uint64_t rhsSign = ieee754::extractSignPreserveBitPattern(rhsData);
                 
                if(FastMath)
                {
                    if (emulated_float64_t_impl::areBothZero(lhsData, rhsData))
                    {
                        if (lhsSign == rhsSign)
                            return bit_cast<this_t>(lhsSign);
                        else
                            return bit_cast<this_t>(0ull);
                    }
                    if(emulated_float64_t_impl::isZero(lhsData))
                        return bit_cast<this_t>(rhsData);
                    if (emulated_float64_t_impl::isZero(rhsData))
                        return bit_cast<this_t>(lhsData);
                    if (tgmath::isInf(lhsData))
                        return bit_cast<this_t>(ieee754::traits<float64_t>::inf | ieee754::extractSignPreserveBitPattern(max(lhsData, rhsData)));
                }

                uint64_t lhsNormMantissa = ieee754::extractNormalizeMantissa(lhsData);
                uint64_t rhsNormMantissa = ieee754::extractNormalizeMantissa(rhsData);

                const int expDiff = lhsBiasedExp - rhsBiasedExp;

                int exp = max(lhsBiasedExp, rhsBiasedExp) - ieee754::traits<float64_t>::exponentBias;
                const uint32_t shiftAmount = abs(expDiff);

                if (expDiff < 0)
                {
                    // so lhsNormMantissa always holds mantissa of number with greater exponent
                    swap<uint64_t>(lhsNormMantissa, rhsNormMantissa);
                    swap<uint64_t>(lhsSign, rhsSign);
                }

                uint64_t resultMantissa;
                if (lhsSign != rhsSign)
                {
                    uint64_t rhsNormMantissaHigh = shiftAmount >= 64 ? 0ull : rhsNormMantissa >> shiftAmount;
                    uint64_t rhsNormMantissaLow = 0ull;
                    if (shiftAmount < 128)
                    {
                        if (shiftAmount >= 64)
                            rhsNormMantissaLow = rhsNormMantissa >> (shiftAmount - 64);
                        else
                            rhsNormMantissaLow = rhsNormMantissa << (64 - shiftAmount);
                    }

                    const int64_t mantissaDiff = int64_t(lhsNormMantissa) - int64_t(rhsNormMantissaHigh);
                    // can only happen when shiftAmount == 0, so it is safe to swap only high bits of rhs mantissa
                    if (mantissaDiff < 0)
                    {
                        swap<uint64_t>(lhsNormMantissa, rhsNormMantissaHigh);
                        swap<uint64_t>(lhsSign, rhsSign);
                    }

                    resultMantissa = emulated_float64_t_impl::subMantissas128NormalizeResult(lhsNormMantissa, rhsNormMantissaHigh, rhsNormMantissaLow, exp);

                    if (resultMantissa == 0ull)
                        return _static_cast<this_t>(0ull);
                }
                else
                {
                    rhsNormMantissa >>= shiftAmount;
                    resultMantissa = lhsNormMantissa + rhsNormMantissa;

                    if (resultMantissa & 1ull << 53)
                    {
                        ++exp;
                        resultMantissa >>= 1;
                    }
                }

                uint64_t resultBiasedExp = uint64_t(exp) + ieee754::traits<float64_t>::exponentBias;
                resultMantissa &= ieee754::traits<float64_t>::mantissaMask;
                uint64_t output = emulated_float64_t_impl::assembleFloat64(lhsSign, resultBiasedExp << ieee754::traits<float64_t>::mantissaBitCnt, resultMantissa);
                return bit_cast<this_t>(output);
            }

            // not implemented
            if (!FlushDenormToZero)
                return bit_cast<this_t>(0xdeadbeefbadcaffeull);
        }

        emulated_float64_t operator+(float rhs)
        {
            return bit_cast<this_t>(data) + create(rhs);
        }

        emulated_float64_t operator-(emulated_float64_t rhs) NBL_CONST_MEMBER_FUNC
        {
            emulated_float64_t lhs = bit_cast<this_t>(data);
            emulated_float64_t rhsFlipped = rhs.flipSign();
            
            return lhs + rhsFlipped;
        }

        emulated_float64_t operator-(float rhs) NBL_CONST_MEMBER_FUNC
        {
            return bit_cast<this_t>(data) - create(rhs);
        }

        emulated_float64_t operator*(emulated_float64_t rhs) NBL_CONST_MEMBER_FUNC
        {
            // TODO: remove
            /*float64_t sum = bit_cast<float64_t>(data) * bit_cast<float64_t>(rhs.data);
            uint64_t sumAsUint = bit_cast<uint64_t>(sum);

            this_t output2;
            output2.data = sumAsUint;

            return output2;*/

            if(FlushDenormToZero)
            {
                emulated_float64_t retval = this_t::create(0ull);

                int lhsBiasedExp = ieee754::extractBiasedExponent(data);
                int rhsBiasedExp = ieee754::extractBiasedExponent(rhs.data);

                uint64_t lhsData = emulated_float64_t_impl::flushDenormToZero(lhsBiasedExp, data);
                uint64_t rhsData = emulated_float64_t_impl::flushDenormToZero(rhsBiasedExp, rhs.data);

                uint64_t lhsSign = lhsData & ieee754::traits<float64_t>::signMask;
                uint64_t rhsSign = rhsData & ieee754::traits<float64_t>::signMask;
                uint64_t sign = (lhsData ^ rhsData) & ieee754::traits<float64_t>::signMask;

                uint64_t lhsMantissa = ieee754::extractMantissa(lhsData);
                uint64_t rhsMantissa = ieee754::extractMantissa(rhsData);

                int exp = int(lhsBiasedExp + rhsBiasedExp) - ieee754::traits<float64_t>::exponentBias;
                if (FastMath)
                {
                    if (tgmath::isNaN(lhsData) || tgmath::isNaN(rhsData))
                        return bit_cast<this_t>(ieee754::traits<float64_t>::quietNaN | sign);
                    if (tgmath::isInf(lhsData) || tgmath::isInf(rhsData))
                        return bit_cast<this_t>(ieee754::traits<float64_t>::inf | sign);
                    if (emulated_float64_t_impl::areBothZero(lhsData, rhsData))
                        return bit_cast<this_t>(sign);
                }
                
                if (emulated_float64_t_impl::isZero(lhsData) || emulated_float64_t_impl::isZero(rhsData))
                    return bit_cast<this_t>(sign);

                const uint64_t hi_l = (lhsMantissa >> 21) | (1ull << 31);
                const uint64_t lo_l = lhsMantissa & ((1ull << 21) - 1);
                const uint64_t hi_r = (rhsMantissa >> 21) | (1ull << 31);
                const uint64_t lo_r = rhsMantissa & ((1ull << 21) - 1);

                //const uint64_t RoundToNearest = (1ull << 31) - 1;
                uint64_t newPseudoMantissa = ((hi_l * hi_r) >> 10) + ((hi_l * lo_r + lo_l * hi_r/* + RoundToNearest*/) >> 31);

                if (newPseudoMantissa == 0ull)
                    return _static_cast<this_t>(0ull);


                if (newPseudoMantissa & (0x1ull << 53))
                {
                    newPseudoMantissa >>= 1;
                    ++exp;
                }
                newPseudoMantissa &= (ieee754::traits<float64_t>::mantissaMask);

                return bit_cast<this_t>(emulated_float64_t_impl::assembleFloat64(sign, uint64_t(exp) << ieee754::traits<float64_t>::mantissaBitCnt, newPseudoMantissa));
            }
            else
            {
                //static_assert(false, "not implemented yet");
                return bit_cast<this_t>(0xdeadbeefbadcaffeull);
            }
        }

        emulated_float64_t operator*(float rhs)
        {
            return bit_cast<this_t>(data) * create(rhs);
        }

        emulated_float64_t operator/(const emulated_float64_t rhs) NBL_CONST_MEMBER_FUNC
        {
            // TODO: remove
            /*float64_t sum = bit_cast<float64_t>(data) / bit_cast<float64_t>(rhs.data);
            uint64_t sumAsUint = bit_cast<uint64_t>(sum);

            this_t output2;
            output2.data = sumAsUint;

            return output2;*/

            if (FlushDenormToZero)
            {
                const uint64_t sign = (data ^ rhs.data) & ieee754::traits<float64_t>::signMask;

                int lhsBiasedExp = ieee754::extractBiasedExponent(data);
                int rhsBiasedExp = ieee754::extractBiasedExponent(rhs.data);

                uint64_t lhsData = emulated_float64_t_impl::flushDenormToZero(lhsBiasedExp, data);
                uint64_t rhsData = emulated_float64_t_impl::flushDenormToZero(rhsBiasedExp, rhs.data);

                if(FastMath)
                {
                    if (tgmath::isNaN<uint64_t>(lhsData) || tgmath::isNaN<uint64_t>(rhsData))
                        return bit_cast<this_t>(ieee754::traits<float64_t>::quietNaN);
                    if (emulated_float64_t_impl::areBothZero(lhsData, rhsData))
                        return bit_cast<this_t>(ieee754::traits<float64_t>::quietNaN | sign);
                    if (emulated_float64_t_impl::isZero(rhsData))
                        return bit_cast<this_t>(ieee754::traits<float64_t>::inf | sign);
                    if (emulated_float64_t_impl::areBothInfinity(lhsData, rhsData))
                        return bit_cast<this_t>(ieee754::traits<float64_t>::quietNaN | ieee754::traits<float64_t>::signMask);
                    if (tgmath::isInf(lhsData))
                        return bit_cast<this_t>(ieee754::traits<float64_t>::inf | sign);
                    if (tgmath::isInf(rhsData))
                        return bit_cast<this_t>(sign);
                }

                if (emulated_float64_t_impl::isZero(lhsData))
                    return bit_cast<this_t>(sign);

                const uint64_t lhsRealMantissa = (ieee754::extractMantissa(lhsData) | (1ull << ieee754::traits<float64_t>::mantissaBitCnt));
                const uint64_t rhsRealMantissa = ieee754::extractMantissa(rhsData) | (1ull << ieee754::traits<float64_t>::mantissaBitCnt);

                int exp = lhsBiasedExp - rhsBiasedExp + int(ieee754::traits<float64_t>::exponentBias);

                uint64_t2 lhsMantissaShifted = emulated_float64_t_impl::shiftMantissaLeftBy53(lhsRealMantissa);
                uint64_t mantissa = emulated_float64_t_impl::divmod128by64(lhsMantissaShifted.x, lhsMantissaShifted.y, rhsRealMantissa);

                const int msb = emulated_float64_t_impl::_findMSB(mantissa);
                if(msb != -1)
                {
                    const int shiftAmount = 52 - msb;
                    assert(shiftAmount >= 0);
                    mantissa <<= shiftAmount;
                    exp -= shiftAmount;
                }

                mantissa &= ieee754::traits<float64_t>::mantissaMask;

                return bit_cast<this_t>(emulated_float64_t_impl::assembleFloat64(sign, uint64_t(exp) << ieee754::traits<float64_t>::mantissaBitCnt, mantissa));
            }
            else
            {
                //static_assert(false, "not implemented yet");
                return bit_cast<this_t>(0xdeadbeefbadcaffeull);
            }
        }

        emulated_float64_t operator/(const float rhs) NBL_CONST_MEMBER_FUNC
        {
            return bit_cast<this_t>(data) / create(rhs);
        }

        // relational operators
        bool operator==(this_t rhs) NBL_CONST_MEMBER_FUNC
        {
            if (FastMath)
            {
                if (tgmath::isNaN<uint64_t>(data) || tgmath::isNaN<uint64_t>(rhs.data))
                    return false;
                if (emulated_float64_t_impl::areBothZero(data, rhs.data))
                    return true;
            }

            const emulated_float64_t xored = bit_cast<this_t>(data ^ rhs.data);
            if ((xored.data & 0x7FFFFFFFFFFFFFFFull) == 0ull)
                return true;

            return !(xored.data);
        }
        bool operator!=(emulated_float64_t rhs) NBL_CONST_MEMBER_FUNC
        {
            if (FastMath && (tgmath::isNaN<uint64_t>(data) || tgmath::isNaN<uint64_t>(rhs.data)))
                return false;

            return !(bit_cast<this_t>(data) == rhs);
        }
        bool operator<(emulated_float64_t rhs) NBL_CONST_MEMBER_FUNC
        {
            if (FastMath)
            {
                if (tgmath::isNaN<uint64_t>(data) || tgmath::isNaN<uint64_t>(rhs.data))
                    return false;
                if (emulated_float64_t_impl::areBothInfinity(data, rhs.data))
                    return false;
                if (emulated_float64_t_impl::areBothZero(data, rhs.data))
                    return false;
            }

            const uint64_t lhsSign = ieee754::extractSign(data);
            const uint64_t rhsSign = ieee754::extractSign(rhs.data);

            // flip bits of negative numbers and flip signs of all numbers
            uint64_t lhsFlipped = data ^ ((0x7FFFFFFFFFFFFFFFull * lhsSign) | ieee754::traits<float64_t>::signMask);
            uint64_t rhsFlipped = rhs.data ^ ((0x7FFFFFFFFFFFFFFFull * rhsSign) | ieee754::traits<float64_t>::signMask);

            return lhsFlipped < rhsFlipped;
        }
        bool operator>(emulated_float64_t rhs) NBL_CONST_MEMBER_FUNC
        {
            if (FastMath)
            {
                if (tgmath::isNaN<uint64_t>(data) || tgmath::isNaN<uint64_t>(rhs.data))
                    return false;
                if (emulated_float64_t_impl::areBothInfinity(data, rhs.data))
                    return false;
                if (emulated_float64_t_impl::areBothZero(data, rhs.data))
                    return false;
            }

            const uint64_t lhsSign = ieee754::extractSign(data);
            const uint64_t rhsSign = ieee754::extractSign(rhs.data);

            // flip bits of negative numbers and flip signs of all numbers
            uint64_t lhsFlipped = data ^ ((0x7FFFFFFFFFFFFFFFull * lhsSign) | ieee754::traits<float64_t>::signMask);
            uint64_t rhsFlipped = rhs.data ^ ((0x7FFFFFFFFFFFFFFFull * rhsSign) | ieee754::traits<float64_t>::signMask);

            return lhsFlipped > rhsFlipped;
        }
        bool operator<=(emulated_float64_t rhs) NBL_CONST_MEMBER_FUNC 
        { 
            if (FastMath && (tgmath::isNaN<uint64_t>(data) || tgmath::isNaN<uint64_t>(rhs.data)))
                return false;

            return !(bit_cast<this_t>(data) > bit_cast<this_t>(rhs.data));
        }
        bool operator>=(emulated_float64_t rhs)
        {
            if (FastMath && (tgmath::isNaN<uint64_t>(data) || tgmath::isNaN<uint64_t>(rhs.data)))
                return false;

            return !(bit_cast<this_t>(data) < bit_cast<this_t>(rhs.data));
        }

        emulated_float64_t flipSign()
        {
            return bit_cast<this_t>(data ^ ieee754::traits<float64_t>::signMask);
        }

        NBL_CONSTEXPR_STATIC bool isFastMathSupported = FastMath;
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
__VA_ARGS__ output;\
output.data = val;\
\
return output;\
}\
\
template<>\
NBL_CONSTEXPR_FUNC __VA_ARGS__ bit_cast<__VA_ARGS__, float64_t>(NBL_CONST_REF_ARG(float64_t) val)\
{\
__VA_ARGS__ output;\
output.data = bit_cast<uint64_t>(val);\
\
return output;\
}\
\
template<>\
NBL_CONSTEXPR_FUNC uint64_t bit_cast<uint64_t, __VA_ARGS__ >(NBL_CONST_REF_ARG( __VA_ARGS__ ) val)\
{\
return val.data;\
}\
\
template<>\
NBL_CONSTEXPR_FUNC float64_t bit_cast<float64_t, __VA_ARGS__ >(NBL_CONST_REF_ARG( __VA_ARGS__ ) val)\
{\
return bit_cast<float64_t>(val.data);\
}\
\

namespace impl
{

template<typename To, bool FastMath, bool FlushDenormToZero>
struct static_cast_helper<To,emulated_float64_t<FastMath,FlushDenormToZero>,void>
{
    static_assert(is_scalar<To>::value);

    using From = emulated_float64_t<FastMath,FlushDenormToZero>;

    static inline To cast(From v)
    {
        using ToAsFloat = typename float_of_size<sizeof(To)>::type;
        using ToAsUint = typename unsigned_integer_of_size<sizeof(To)>::type;

        if (is_same_v<To, float64_t>)
            return To(bit_cast<float64_t>(v.data));

        if (is_floating_point<To>::value)
        {

            const int exponent = ieee754::extractExponent(v.data);
            if (!From::isFastMathSupported)
            {
                if (exponent > ieee754::traits<ToAsFloat>::exponentMax)
                    return bit_cast<To>(ieee754::traits<ToAsFloat>::inf);
                if (exponent < ieee754::traits<ToAsFloat>::exponentMin)
                    return bit_cast<To>(-ieee754::traits<ToAsFloat>::inf);
                if (tgmath::isNaN(v.data))
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

template<typename From, bool FastMath, bool FlushDenormToZero>
struct static_cast_helper<emulated_float64_t<FastMath, FlushDenormToZero>, From, void>
{
    using To = emulated_float64_t<FastMath, FlushDenormToZero>;

    static inline To cast(From v)
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

#undef IMPLEMENT_IEEE754_FUNC_SPEC_FOR_EMULATED_F64_TYPE
#undef DEFINE_BIT_CAST_SPEC

#endif
