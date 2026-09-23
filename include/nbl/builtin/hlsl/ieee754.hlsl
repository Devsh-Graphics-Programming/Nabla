#ifndef _NBL_BUILTIN_HLSL_IEE754_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_IEE754_HLSL_INCLUDED_

// TODO: make every function in this header follow the scheme `replaceBiasedExponent` uses:
// - the implementation works on the `uintN_t` bit pattern (using `traits` masks) and returns integer types,
// - the floating point version `impl::bitCastToUintType`s its argument, calls the bit pattern version and,
//   if it returns a value of the input type, `impl::castBackToFloatType`s the result.
// So uint in gives uint out, float in gives float out, and there's a single implementation.
// Candidates: `extractBiasedExponent` (explicit `uint64_t`/`float64_t` specializations), `extractExponent`, `extractMantissa`,
// `extractNormalizeMantissa`, `extractSign`, `extractSignPreserveBitPattern`, `copySign`, `flipSign`, `flipSignIfRHSNegative`,
// `isSubnormal`, `isZero`, `nextDown`, `nextTowardZero`.
// Dispatch with `impl::` helper structs partially specialized on `concepts::UnsignedIntegralScalar`/`concepts::FloatingPointScalar`,
// NOT explicit function specializations: those aren't templates, so MSVC requires a `constexpr` one to be constant-evaluable (C3615).
// Also enforce with a concept that `impl::castBackToFloatType<T>` gets an unsigned integral `T`.

#include <nbl/builtin/hlsl/type_traits.hlsl>
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/bit.hlsl>
#include <nbl/builtin/hlsl/concepts/core.hlsl>
#include <nbl/builtin/hlsl/spirv_intrinsics/core.hlsl>

namespace nbl
{
namespace hlsl
{
namespace ieee754
{

namespace impl
{
template <typename T>
NBL_CONSTEXPR_FUNC unsigned_integer_of_size_t<sizeof(T)> bitCastToUintType(T x)
{
	using AsUint = unsigned_integer_of_size_t<sizeof(T)>;
	return bit_cast<AsUint, T>(x);
}
// to avoid bit cast from uintN_t to uintN_t
template <> NBL_CONSTEXPR_FUNC unsigned_integer_of_size_t<2> bitCastToUintType(uint16_t x) { return x; }
template <> NBL_CONSTEXPR_FUNC unsigned_integer_of_size_t<4> bitCastToUintType(uint32_t x) { return x; }
template <> NBL_CONSTEXPR_FUNC unsigned_integer_of_size_t<8> bitCastToUintType(uint64_t x) { return x; }

// Inverse of `bitCastToUintType`: reinterprets integer bits as the float of the same size.
// `T` MUST be the unsigned integer type of the bits (e.g. `castBackToFloatType<uint32_t>(bits)`), NOT the float type.
// Passing a float `T` makes the call site convert the integer bits to that float BY VALUE before the (then same-type) bit cast,
// which silently returns garbage (`nextDown(1.0f)` used to return ~1.07e9 because of exactly that).
// TODO: enforce `T` being an unsigned integral scalar with a concept (see the TODO at the top of this header).
template <typename T>
NBL_CONSTEXPR_FUNC typename float_of_size<sizeof(T)>::type castBackToFloatType(T x)
{
	using AsFloat = typename float_of_size<sizeof(T)>::type;
	return bit_cast<AsFloat, T>(x);
}
}

template<typename Float>
struct traits_base
{
	static_assert(is_same<Float, float16_t>::value || is_same<Float, float32_t>::value || is_same<Float, float64_t>::value);
	NBL_CONSTEXPR_STATIC_INLINE int16_t exponentBitCnt = (int16_t)0xbeef;
	NBL_CONSTEXPR_STATIC_INLINE int16_t mantissaBitCnt = (int16_t)0xbeef;
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
	return _static_cast<uint32_t>(output);
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

namespace impl
{
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct replaceBiasedExponent_helper;

// the implementation, works on the bit pattern and returns the bit pattern
template<typename UnsignedIntegral>
NBL_PARTIAL_REQ_TOP(concepts::UnsignedIntegralScalar<UnsignedIntegral>)
struct replaceBiasedExponent_helper<UnsignedIntegral NBL_PARTIAL_REQ_BOT(concepts::UnsignedIntegralScalar<UnsignedIntegral>) >
{
	static UnsignedIntegral __call(const UnsignedIntegral bits, const UnsignedIntegral biasedExp)
	{
		using traits_t = traits<typename float_of_size<sizeof(UnsignedIntegral)>::type>;
		// bits of `biasedExp` that don't fit in the exponent are dropped
		return UnsignedIntegral((bits & UnsignedIntegral(~traits_t::exponentMask)) | ((biasedExp << traits_t::mantissaBitCnt) & traits_t::exponentMask));
	}
};

// floats go through the bit pattern version and come back as floats
template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<FloatingPoint>)
struct replaceBiasedExponent_helper<FloatingPoint NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<FloatingPoint>) >
{
	using AsUint = typename unsigned_integer_of_size<sizeof(FloatingPoint)>::type;

	static FloatingPoint __call(const FloatingPoint x, const AsUint biasedExp)
	{
		return castBackToFloatType<AsUint>(replaceBiasedExponent_helper<AsUint>::__call(bitCastToUintType(x), biasedExp));
	}
};
}

// `T` can be a native float, returning a float, or the `uintN_t` bit pattern of one (as used by `emulated_float64_t`), returning a bit pattern
template <typename T>
NBL_CONSTEXPR_FUNC T replaceBiasedExponent(T x, typename unsigned_integer_of_size<sizeof(T)>::type biasedExp)
{
	return impl::replaceBiasedExponent_helper<T>::__call(x, biasedExp);
}

// performs no overflow tests, returns x*exp2(n)
template <typename T>
NBL_CONSTEXPR_FUNC T fastMulExp2(T x, int n)
{
	return replaceBiasedExponent(x, extractBiasedExponent(x) + _static_cast<uint32_t>(n));
}

template <typename T>
NBL_CONSTEXPR_FUNC typename unsigned_integer_of_size<sizeof(T)>::type extractMantissa(T x)
{
	using AsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
	return ieee754::impl::bitCastToUintType(x) & traits<typename float_of_size<sizeof(T)>::type>::mantissaMask;
}

template <typename T>
NBL_CONSTEXPR_FUNC typename unsigned_integer_of_size<sizeof(T)>::type extractNormalizeMantissa(T x)
{
	using AsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
	using AsFloat = typename float_of_size<sizeof(T)>::type;
	return extractMantissa(x) | (AsUint(1) << traits<AsFloat>::mantissaBitCnt);
}

template <typename T>
NBL_CONSTEXPR_FUNC typename unsigned_integer_of_size<sizeof(T)>::type extractSign(T x)
{
	using AsFloat = typename float_of_size<sizeof(T)>::type;
	return (ieee754::impl::bitCastToUintType(x) & traits<AsFloat>::signMask) >> ((sizeof(T) * 8) - 1);
}

template <typename T>
NBL_CONSTEXPR_FUNC typename unsigned_integer_of_size<sizeof(T)>::type extractSignPreserveBitPattern(T x)
{
	using AsFloat = typename float_of_size<sizeof(T)>::type;
	return ieee754::impl::bitCastToUintType(x) & traits<AsFloat>::signMask;
}

template <typename FloatingPoint NBL_FUNC_REQUIRES(concepts::FloatingPointLikeScalar<FloatingPoint>)
NBL_CONSTEXPR_FUNC FloatingPoint copySign(FloatingPoint to, FloatingPoint from)
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
		// can't use mix_helper because circular dep
#ifdef __HLSL_VERSION
		return bit_cast<FloatingPoint>(asUint ^ spirv::select(flip, ieee754::traits<AsFloat>::signMask, AsUint(0ull)));
#else
		return bit_cast<FloatingPoint>(asUint ^ (flip ? ieee754::traits<AsFloat>::signMask : AsUint(0ull)));
#endif
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

template <typename T, typename U NBL_STRUCT_CONSTRAINABLE>
struct flipSignIfRHSNegative_helper;

template <typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointLikeScalar<FloatingPoint>)
struct flipSignIfRHSNegative_helper<FloatingPoint, FloatingPoint NBL_PARTIAL_REQ_BOT(concepts::FloatingPointLikeScalar<FloatingPoint>) >
{
	static FloatingPoint __call(FloatingPoint val, FloatingPoint flip)
	{
		using AsFloat = typename float_of_size<sizeof(FloatingPoint)>::type;
		using AsUint = typename unsigned_integer_of_size<sizeof(FloatingPoint)>::type;
		const AsUint asUint = ieee754::impl::bitCastToUintType(val);
		return bit_cast<FloatingPoint>(asUint ^ (ieee754::traits<AsFloat>::signMask & ieee754::impl::bitCastToUintType(flip)));
	}
};

template <typename Vectorial>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointLikeVectorial<Vectorial>)
struct flipSignIfRHSNegative_helper<Vectorial, Vectorial NBL_PARTIAL_REQ_BOT(concepts::FloatingPointLikeVectorial<Vectorial>) >
{
	static Vectorial __call(Vectorial val, Vectorial flip)
	{
		using traits_v = hlsl::vector_traits<Vectorial>;
		array_get<Vectorial, typename traits_v::scalar_type> getter_v;
		array_set<Vectorial, typename traits_v::scalar_type> setter;

		Vectorial output;
		for (uint32_t i = 0; i < traits_v::Dimension; ++i)
			setter(output, i, flipSignIfRHSNegative_helper<typename traits_v::scalar_type,typename traits_v::scalar_type>::__call(getter_v(val, i), getter_v(flip, i)));

		return output;
	}
};

template <typename Vectorial, typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointLikeVectorial<Vectorial> && concepts::FloatingPointLikeScalar<FloatingPoint>)
struct flipSignIfRHSNegative_helper<Vectorial, FloatingPoint NBL_PARTIAL_REQ_BOT(concepts::FloatingPointLikeVectorial<Vectorial> && concepts::FloatingPointLikeScalar<FloatingPoint>) >
{
	static Vectorial __call(Vectorial val, FloatingPoint flip)
	{
		using traits_v = hlsl::vector_traits<Vectorial>;
		array_get<Vectorial, typename traits_v::scalar_type> getter_v;
		array_set<Vectorial, typename traits_v::scalar_type> setter;

		using AsFloat = typename float_of_size<sizeof(FloatingPoint)>::type;
		using AsUint = typename unsigned_integer_of_size<sizeof(FloatingPoint)>::type;
		const AsUint signBitFlip = ieee754::traits<AsFloat>::signMask & ieee754::impl::bitCastToUintType(flip);

		Vectorial output;
		for (uint32_t i = 0; i < traits_v::Dimension; ++i)
			setter(output, i, bit_cast<FloatingPoint>(ieee754::impl::bitCastToUintType(getter_v(val, i)) ^ signBitFlip));

		return output;
	}
};
}

template <typename T, typename U = bool>
NBL_CONSTEXPR_FUNC T flipSign(T val, U flip = true)
{
	return impl::flipSign_helper<T, U>::__call(val, flip);
}

template <typename T, typename U=T>
NBL_CONSTEXPR_FUNC T flipSignIfRHSNegative(T val, U flip)
{
	return impl::flipSignIfRHSNegative_helper<T, U>::__call(val, flip);
}

template <typename T NBL_FUNC_REQUIRES(hlsl::is_floating_point_v<T>)
NBL_CONSTEXPR_FUNC bool isSubnormal(T val)
{
	const uint32_t biasedExponent = extractBiasedExponent(val);
	const typename unsigned_integer_of_size<sizeof(T)>::type mantissa = extractMantissa(val);
	return biasedExponent == 0 && mantissa != 0u;
}

template <typename T NBL_FUNC_REQUIRES(hlsl::is_floating_point_v<T>)
NBL_CONSTEXPR_FUNC bool isZero(T val)
{
	using traits_t = traits<T>;
	using AsUint = typename unsigned_integer_of_size<sizeof(T)>::type;

	const AsUint exponentAndMantissaMask = ~traits_t::signMask;
	return !(ieee754::impl::bitCastToUintType(val) & exponentAndMantissaMask);
}

// Returns the largest representable value less than `val`.
// For positive values this decrements the bit representation; for negative values it increments.
// Caller must guarantee val is finite and non-zero.
template <bool CanBeNeg = true, typename T NBL_FUNC_REQUIRES(hlsl::is_floating_point_v<T>)
NBL_CONSTEXPR_FUNC T nextDown(T val)
{
	using AsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
	using traits_t = traits<T>;

	const AsUint bits = ieee754::impl::bitCastToUintType(val);

	// positive: decrement; negative: increment
	AsUint result;
	if (CanBeNeg)
	{
		const bool isNegative = (bits & traits_t::signMask) != AsUint(0);
		result = isNegative ? (bits + AsUint(1)) : (bits - AsUint(1));
	}
	else
		result = bits - AsUint(1);
	return impl::castBackToFloatType<AsUint>(result);
}

// Returns the representable value nearest to `val` in the direction of zero.
// For positive values this decrements the bit representation; for negative values it decrements (moving toward zero).
// Caller must guarantee val is finite and non-zero.
template <typename T NBL_FUNC_REQUIRES(hlsl::is_floating_point_v<T>)
NBL_CONSTEXPR_FUNC T nextTowardZero(T val)
{
	using AsUint = typename unsigned_integer_of_size<sizeof(T)>::type;

	const AsUint bits = ieee754::impl::bitCastToUintType(val);
	return impl::castBackToFloatType<AsUint>(bits - AsUint(1));
}

// Number of representable values (ULPs) between `lhs` and `rhs`, measured on the bit patterns.
// Crossing zero adds up both magnitudes, so `+0` and `-0` are 0 ULPs apart.
// NaN and infinity are not special cased, this is the raw bit pattern distance
// (two NaNs, or the largest finite value and infinity, can be a single ULP apart).
template <typename T NBL_FUNC_REQUIRES(hlsl::is_floating_point_v<T>)
NBL_CONSTEXPR_FUNC typename unsigned_integer_of_size<sizeof(T)>::type ulpDistance(T lhs, T rhs)
{
	using AsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
	const AsUint signMask = traits<T>::signMask;

	const AsUint lhsBits = ieee754::impl::bitCastToUintType(lhs);
	const AsUint rhsBits = ieee754::impl::bitCastToUintType(rhs);
	const AsUint lhsMagnitude = AsUint(lhsBits & AsUint(~signMask));
	const AsUint rhsMagnitude = AsUint(rhsBits & AsUint(~signMask));
	// opposite signs, the distance goes through zero (can't overflow, both magnitudes have their top bit clear)
	if ((lhsBits ^ rhsBits) & signMask)
		return AsUint(lhsMagnitude + rhsMagnitude);
	return lhsMagnitude > rhsMagnitude ? AsUint(lhsMagnitude - rhsMagnitude) : AsUint(rhsMagnitude - lhsMagnitude);
}

}
}
}

#endif