#ifndef _NBL_BUILTIN_HLSL_IEE754_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_IEE754_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/ieee754/impl.hlsl>

namespace nbl
{
namespace hlsl
{
namespace ieee754
{

template<typename Float>
struct traits_base
{
	static_assert(is_same<Float, float16_t>::value || is_same<Float, float32_t>::value || is_same<Float, float64_t>::value);
	NBL_CONSTEXPR_STATIC_INLINE int16_t exponentBitCnt = int16_t(0xbeef);
	NBL_CONSTEXPR_STATIC_INLINE int16_t mantissaBitCnt = int16_t(0xbeef);
};

template<>
struct traits_base<float16_t>
{
	NBL_CONSTEXPR_STATIC_INLINE int16_t exponentBitCnt = 5;
	NBL_CONSTEXPR_STATIC_INLINE int16_t mantissaBitCnt = 10;
};

template<>
struct traits_base<float32_t>
{
	NBL_CONSTEXPR_STATIC_INLINE int16_t exponentBitCnt = 8;
	NBL_CONSTEXPR_STATIC_INLINE int16_t mantissaBitCnt = 23;
};

template<>
struct traits_base<float64_t>
{
	NBL_CONSTEXPR_STATIC_INLINE int16_t exponentBitCnt = 11;
	NBL_CONSTEXPR_STATIC_INLINE int16_t mantissaBitCnt = 52;
};

template<typename Float>
struct traits : traits_base<Float>
{
	//static_assert(is_same_v<Float, float16_t> || is_same_v<Float, float32_t> || is_same_v<Float, float64_t>);

	using bit_rep_t = typename unsigned_integer_of_size<sizeof(Float)>::type;
	using base_t = traits_base<Float>;

	NBL_CONSTEXPR_STATIC_INLINE bit_rep_t signMask = bit_rep_t(0x1ull) << (sizeof(Float) * 8 - 1);
	NBL_CONSTEXPR_STATIC_INLINE bit_rep_t exponentMask = ((~bit_rep_t(0)) << base_t::mantissaBitCnt) ^ signMask;
	NBL_CONSTEXPR_STATIC_INLINE bit_rep_t mantissaMask = (bit_rep_t(0x1u) << base_t::mantissaBitCnt) - 1;
	NBL_CONSTEXPR_STATIC_INLINE int exponentBias = (int(0x1) << (base_t::exponentBitCnt - 1)) - 1;
	NBL_CONSTEXPR_STATIC_INLINE bit_rep_t inf = exponentMask;
	NBL_CONSTEXPR_STATIC_INLINE bit_rep_t specialValueExp = (1ull << base_t::exponentBitCnt) - 1;
	NBL_CONSTEXPR_STATIC_INLINE bit_rep_t quietNaN = exponentMask | (1ull << (base_t::mantissaBitCnt - 1));
	NBL_CONSTEXPR_STATIC_INLINE bit_rep_t max = ((1ull << (sizeof(Float) * 8 - 1)) - 1) & (~(1ull << base_t::mantissaBitCnt));
	NBL_CONSTEXPR_STATIC_INLINE bit_rep_t min = 1ull << base_t::mantissaBitCnt;
	NBL_CONSTEXPR_STATIC_INLINE int exponentMax = exponentBias;
	NBL_CONSTEXPR_STATIC_INLINE int exponentMin = -(exponentBias - 1);
};

template <typename T>
inline uint32_t extractBiasedExponent(T x)
{
	using AsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
	return glsl::bitfieldExtract<AsUint>(ieee754::impl::bitCastToUintType(x), traits<typename float_of_size<sizeof(T)>::type>::mantissaBitCnt, traits<typename float_of_size<sizeof(T)>::type>::exponentBitCnt);
}

template<>
inline uint32_t extractBiasedExponent(uint64_t x)
{
	uint64_t output = (x >> traits<float64_t>::mantissaBitCnt) & (traits<float64_t>::exponentMask >> traits<float64_t>::mantissaBitCnt);
	return uint32_t(output);
}

template<>
inline uint32_t extractBiasedExponent(float64_t x)
{
	return extractBiasedExponent<uint64_t>(ieee754::impl::bitCastToUintType(x));
}

template <typename T>
inline int extractExponent(T x)
{
	using AsFloat = typename float_of_size<sizeof(T)>::type;
	return int(extractBiasedExponent(x)) - traits<AsFloat>::exponentBias;
}

template <typename T>
NBL_CONSTEXPR_INLINE_FUNC T replaceBiasedExponent(T x, typename unsigned_integer_of_size<sizeof(T)>::type biasedExp)
{
	// TODO:
	//staticAssertTmp(impl::isTypeAllowed<T>(), "Invalid type! Only floating point or unsigned integer types are allowed.");
	using AsFloat = typename float_of_size<sizeof(T)>::type;
	return impl::castBackToFloatType<T>(glsl::bitfieldInsert(ieee754::impl::bitCastToUintType(x), biasedExp, traits<AsFloat>::mantissaBitCnt, traits<AsFloat>::exponentBitCnt));
}

// performs no overflow tests, returns x*exp2(n)
template <typename T>
NBL_CONSTEXPR_INLINE_FUNC T fastMulExp2(T x, int n)
{
	return replaceBiasedExponent(x, extractBiasedExponent(x) + uint32_t(n));
}

template <typename T>
NBL_CONSTEXPR_INLINE_FUNC typename unsigned_integer_of_size<sizeof(T)>::type extractMantissa(T x)
{
	using AsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
	return ieee754::impl::bitCastToUintType(x) & traits<typename float_of_size<sizeof(T)>::type>::mantissaMask;
}

template <typename T>
NBL_CONSTEXPR_INLINE_FUNC typename unsigned_integer_of_size<sizeof(T)>::type extractNormalizeMantissa(T x)
{
	using AsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
	using AsFloat = typename float_of_size<sizeof(T)>::type;
	return extractMantissa(x) | (AsUint(1) << traits<AsFloat>::mantissaBitCnt);
}

template <typename T>
NBL_CONSTEXPR_INLINE_FUNC typename unsigned_integer_of_size<sizeof(T)>::type extractSign(T x)
{
	using AsFloat = typename float_of_size<sizeof(T)>::type;
	return (ieee754::impl::bitCastToUintType(x) & traits<AsFloat>::signMask) >> ((sizeof(T) * 8) - 1);
}

template <typename T>
NBL_CONSTEXPR_INLINE_FUNC typename unsigned_integer_of_size<sizeof(T)>::type extractSignPreserveBitPattern(T x)
{
	using AsFloat = typename float_of_size<sizeof(T)>::type;
	return ieee754::impl::bitCastToUintType(x) & traits<AsFloat>::signMask;
}

}
}
}

#endif