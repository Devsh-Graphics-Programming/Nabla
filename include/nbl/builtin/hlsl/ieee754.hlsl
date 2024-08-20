#ifndef _NBL_BUILTIN_HLSL_IEE754_H_INCLUDED_
#define _NBL_BUILTIN_HLSL_IEE754_H_INCLUDED_

#include <nbl/builtin/hlsl/type_traits.hlsl>
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/bit.hlsl>

namespace nbl
{
namespace hlsl
{
namespace ieee754
{

// TODO: move to builtin/hlsl/impl/ieee754_impl.hlsl?
namespace impl
{
	template <typename T>
	NBL_CONSTEXPR_INLINE_FUNC bool isTypeAllowed()
	{
		return is_same<T, uint16_t>::value ||
			is_same<T, uint32_t>::value ||
			is_same<T, uint64_t>::value ||
			is_same<T, float16_t>::value ||
			is_same<T, float32_t>::value ||
			is_same<T, float64_t>::value;
	}

	template <typename T>
	NBL_CONSTEXPR_INLINE_FUNC typename unsigned_integer_of_size<sizeof(T)>::type castToUintType(T x)
	{
		using AsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
		return bit_cast<AsUint, T>(x);
	}
	// to avoid bit cast from uintN_t to uintN_t
	template <> NBL_CONSTEXPR_INLINE_FUNC unsigned_integer_of_size<2>::type castToUintType(uint16_t x) { return x; }
	template <> NBL_CONSTEXPR_INLINE_FUNC unsigned_integer_of_size<4>::type castToUintType(uint32_t x) { return x; }
	template <> NBL_CONSTEXPR_INLINE_FUNC unsigned_integer_of_size<8>::type castToUintType(uint64_t x) { return x; }

	template <typename T>
	NBL_CONSTEXPR_INLINE_FUNC T castBackToFloatType(T x)
	{
		using AsFloat = typename float_of_size<sizeof(T)>::type;
		return bit_cast<AsFloat, T>(x);
	}
	template<> NBL_CONSTEXPR_INLINE_FUNC uint16_t castBackToFloatType(uint16_t x) { return x; }
	template<> NBL_CONSTEXPR_INLINE_FUNC uint32_t castBackToFloatType(uint32_t x) { return x; }
	template<> NBL_CONSTEXPR_INLINE_FUNC uint64_t castBackToFloatType(uint64_t x) { return x; }
}

template<typename Float>
struct traits_base
{
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

	NBL_CONSTEXPR_STATIC_INLINE bit_rep_t signMask = bit_rep_t(0x1u) << (sizeof(Float) * 8 - 1);
	NBL_CONSTEXPR_STATIC_INLINE bit_rep_t exponentMask = ((~bit_rep_t(0)) << base_t::mantissaBitCnt) ^ signMask;
	NBL_CONSTEXPR_STATIC_INLINE bit_rep_t mantissaMask = (bit_rep_t(0x1u) << base_t::mantissaBitCnt) - 1;
	NBL_CONSTEXPR_STATIC_INLINE int exponentBias = (int(0x1) << (base_t::exponentBitCnt - 1)) - 1;
	NBL_CONSTEXPR_STATIC_INLINE bit_rep_t inf = exponentMask;
	NBL_CONSTEXPR_STATIC_INLINE bit_rep_t specialValueExp = (1ull << base_t::exponentBitCnt) - 1;
	NBL_CONSTEXPR_STATIC_INLINE bit_rep_t quietNaN = exponentMask | (1ull << (base_t::mantissaBitCnt - 1));
	NBL_CONSTEXPR_STATIC_INLINE bit_rep_t max = ((1ull << (sizeof(Float) * 8 - 1)) - 1) & (~(1ull << base_t::mantissaBitCnt));
	NBL_CONSTEXPR_STATIC_INLINE bit_rep_t min = 1ull << base_t::mantissaBitCnt;
};

template <typename T>
inline uint32_t extractBiasedExponent(T x)
{
	using AsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
	return glsl::bitfieldExtract<AsUint>(impl::castToUintType(x), traits<typename float_of_size<sizeof(T)>::type>::mantissaBitCnt, traits<typename float_of_size<sizeof(T)>::type>::exponentBitCnt);
}

template<>
inline uint32_t extractBiasedExponent(uint64_t x)
{
	const uint32_t highBits = uint32_t(x >> 32);
	return glsl::bitfieldExtract<uint32_t>(highBits, traits<float64_t>::mantissaBitCnt - 32, traits<float64_t>::exponentBitCnt);
}

template<>
inline uint32_t extractBiasedExponent(float64_t x)
{
	return extractBiasedExponent<uint64_t>(impl::castToUintType(x));
}

template <typename T>
inline int extractExponent(T x)
{
	using AsFloat = typename float_of_size<sizeof(T)>::type;
	return int(extractBiasedExponent(x)) - int(traits<AsFloat>::exponentBias);
}

template <typename T>
NBL_CONSTEXPR_INLINE_FUNC T replaceBiasedExponent(T x, typename unsigned_integer_of_size<sizeof(T)>::type biasedExp)
{
	// TODO:
	//staticAssertTmp(impl::isTypeAllowed<T>(), "Invalid type! Only floating point or unsigned integer types are allowed.");
	using AsFloat = typename float_of_size<sizeof(T)>::type;
	return impl::castBackToFloatType<T>(glsl::bitfieldInsert(impl::castToUintType(x), biasedExp, traits<AsFloat>::mantissaBitCnt, traits<AsFloat>::exponentBitCnt));
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
	return impl::castToUintType(x) & traits<typename float_of_size<sizeof(T)>::type>::mantissaMask;
}

template <typename T>
NBL_CONSTEXPR_INLINE_FUNC typename unsigned_integer_of_size<sizeof(T)>::type extractSign(T x)
{
	return (impl::castToUintType(x) & traits<T>::signMask) >> ((sizeof(T) * 8) - 1);
}

template <typename T>
NBL_CONSTEXPR_INLINE_FUNC typename unsigned_integer_of_size<sizeof(T)>::type extractSignPreserveBitPattern(T x)
{
	return impl::castToUintType(x) & traits<T>::signMask;
}

}
}
}

#endif