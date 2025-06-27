#ifndef _NBL_BUILTIN_HLSL_IEE754_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_IEE754_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/ieee754/impl.hlsl>
#include <nbl/builtin/hlsl/concepts/core.hlsl>

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

template <typename FloatingPoint NBL_FUNC_REQUIRES(concepts::FloatingPointLikeScalar<FloatingPoint>)
NBL_CONSTEXPR_INLINE_FUNC FloatingPoint copySign(FloatingPoint to, FloatingPoint from)
{
	using AsUint = typename unsigned_integer_of_size<sizeof(FloatingPoint)>::type;

	const AsUint toAsUint = ieee754::impl::bitCastToUintType(to) & (~ieee754::traits<FloatingPoint>::signMask);
	const AsUint fromAsUint = ieee754::impl::bitCastToUintType(from);

	return bit_cast<FloatingPoint>(toAsUint | extractSignPreserveBitPattern(from));
}

namespace impl
{
template <typename T, typename U NBL_STRUCT_CONSTRAINABLE>
struct flipSign_helper;

template <typename FloatingPoint, typename Bool>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointLikeScalar<FloatingPoint> && concepts::BooleanScalar<Bool>)
struct flipSign_helper<FloatingPoint, Bool NBL_PARTIAL_REQ_BOT(concepts::FloatingPointLikeScalar<FloatingPoint> && concepts::BooleanScalar<Bool>) >
{
	static FloatingPoint __call(FloatingPoint val, Bool flip)
	{
		using AsFloat = typename float_of_size<sizeof(FloatingPoint)>::type;
		using AsUint = typename unsigned_integer_of_size<sizeof(FloatingPoint)>::type;
		const AsUint asUint = ieee754::impl::bitCastToUintType(val);
		return bit_cast<FloatingPoint>(asUint ^ (flip ? ieee754::traits<AsFloat>::signMask : AsUint(0ull)));
	}
};

template <typename Vectorial, typename Bool>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointLikeVectorial<Vectorial> && concepts::BooleanScalar<Bool>)
struct flipSign_helper<Vectorial, Bool NBL_PARTIAL_REQ_BOT(concepts::FloatingPointLikeVectorial<Vectorial> && concepts::BooleanScalar<Bool>) >
{
	static Vectorial __call(Vectorial val, Bool flip)
	{
		using traits = hlsl::vector_traits<Vectorial>;
		array_get<Vectorial, typename traits::scalar_type> getter;
		array_set<Vectorial, typename traits::scalar_type> setter;

		Vectorial output;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			setter(output, i, flipSign_helper<typename traits::scalar_type, Bool>::__call(getter(val, i), flip));

		return output;
	}
};

template <typename Vectorial, typename BoolVector>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointLikeVectorial<Vectorial> && concepts::Boolean<BoolVector> && !concepts::Scalar<BoolVector> && vector_traits<Vectorial>::Dimension==vector_traits<BoolVector>::Dimension)
struct flipSign_helper<Vectorial, BoolVector NBL_PARTIAL_REQ_BOT(concepts::FloatingPointLikeVectorial<Vectorial> && concepts::Boolean<BoolVector> && !concepts::Scalar<BoolVector> && vector_traits<Vectorial>::Dimension==vector_traits<BoolVector>::Dimension) >
{
	static Vectorial __call(Vectorial val, BoolVector flip)
	{
		using traits_v = hlsl::vector_traits<Vectorial>;
		using traits_f = hlsl::vector_traits<BoolVector>;
		array_get<Vectorial, typename traits_v::scalar_type> getter_v;
		array_get<BoolVector, typename traits_f::scalar_type> getter_f;
		array_set<Vectorial, typename traits_v::scalar_type> setter;

		Vectorial output;
		for (uint32_t i = 0; i < traits_v::Dimension; ++i)
			setter(output, i, flipSign_helper<typename traits_v::scalar_type, typename traits_f::scalar_type>::__call(getter_v(val, i), getter_f(flip, i)));

		return output;
	}
};
}

template <typename T, typename U>
NBL_CONSTEXPR_INLINE_FUNC T flipSign(T val, U flip)
{
	return impl::flipSign_helper<T, U>::__call(val, flip);
}

}
}
}

#endif