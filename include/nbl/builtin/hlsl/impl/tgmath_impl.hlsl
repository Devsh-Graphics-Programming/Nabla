#ifndef _NBL_BUILTIN_HLSL_TGMATH_IMPL_INCLUDED_
#define _NBL_BUILTIN_HLSL_TGMATH_IMPL_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/basic.h>
#include <nbl/builtin/hlsl/concepts.hlsl>
#include <nbl/builtin/hlsl/concepts/vector.hlsl>
#include <nbl/builtin/hlsl/spirv_intrinsics/core.hlsl>
#include <nbl/builtin/hlsl/spirv_intrinsics/glsl.std.450.hlsl>
#include <nbl/builtin/hlsl/ieee754.hlsl>

namespace nbl
{
namespace hlsl
{
namespace tgmath_impl
{

template<typename T, typename U NBL_STRUCT_CONSTRAINABLE>
struct lerp_helper;

#ifdef __HLSL_VERSION
#define MIX_FUNCTION spirv::fMix
#else
#define MIX_FUNCTION glm::mix
#endif

#define DEFINE_LERP_HELPER_COMMON_SPECIALIZATION(TYPE)\
template<>\
struct lerp_helper<TYPE, TYPE>\
{\
	static inline TYPE __call(NBL_CONST_REF_ARG(TYPE) x, NBL_CONST_REF_ARG(TYPE) y, NBL_CONST_REF_ARG(TYPE) a)\
	{\
		return MIX_FUNCTION(x, y, a);\
	}\
};\
\
template<int N>\
struct lerp_helper<vector<TYPE, N>, vector<TYPE, N> >\
{\
	static inline vector<TYPE, N> __call(NBL_CONST_REF_ARG(vector<TYPE, N>) x, NBL_CONST_REF_ARG(vector<TYPE, N>) y, NBL_CONST_REF_ARG(vector<TYPE, N>) a)\
	{\
		return MIX_FUNCTION(x, y, a);\
	}\
};\
\
template<int N>\
struct lerp_helper<vector<TYPE, N>, TYPE>\
{\
	static inline vector<TYPE, N> __call(NBL_CONST_REF_ARG(vector<TYPE, N>) x, NBL_CONST_REF_ARG(vector<TYPE, N>) y, NBL_CONST_REF_ARG(TYPE) a)\
	{\
		return MIX_FUNCTION(x, y, a);\
	}\
};\

DEFINE_LERP_HELPER_COMMON_SPECIALIZATION(float32_t)
DEFINE_LERP_HELPER_COMMON_SPECIALIZATION(float64_t)

#undef DEFINE_LERP_HELPER_COMMON_SPECIALIZATION
#undef MIX_FUNCTION

// LERP

template<typename T>
struct lerp_helper<T, bool>
{
	static inline T __call(NBL_CONST_REF_ARG(T) x, NBL_CONST_REF_ARG(T) y, NBL_CONST_REF_ARG(bool) a)
	{
		if (a)
			return y;
		else
			return x;
	}
};

template<typename T, int N>
struct lerp_helper<vector<T, N>, vector<bool, N> >
{
	using output_vec_t = vector<T, N>;

	static inline output_vec_t __call(NBL_CONST_REF_ARG(output_vec_t) x, NBL_CONST_REF_ARG(output_vec_t) y, NBL_CONST_REF_ARG(vector<bool, N>) a)
	{
		output_vec_t retval;
		for (uint32_t i = 0; i < vector_traits<output_vec_t>::Dimension; i++)
			retval[i] = a[i] ? y[i] : x[i];
		return retval;
	}
};

// ISNAN

template<typename UnsignedInteger NBL_FUNC_REQUIRES(hlsl::is_integral_v<UnsignedInteger> && hlsl::is_unsigned_v<UnsignedInteger>)
inline bool isnan_uint_impl(UnsignedInteger val)
{
	using AsFloat = typename float_of_size<sizeof(UnsignedInteger)>::type;
	UnsignedInteger absVal = val & (hlsl::numeric_limits<UnsignedInteger>::max >> 1);
	return absVal > (ieee754::traits<AsFloat>::specialValueExp << ieee754::traits<AsFloat>::mantissaBitCnt);
}

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct isnan_helper;

template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<FloatingPoint>)
struct isnan_helper<FloatingPoint NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<FloatingPoint>) >
{
	static bool __call(NBL_CONST_REF_ARG(FloatingPoint) x)
	{
#ifdef __HLSL_VERSION
		return spirv::isNan<FloatingPoint>(x);
#else
		// GCC and Clang will always return false with call to std::isnan when fast math is enabled,
		// this implementation will always return appropriate output regardless is fas math is enabled or not
		using AsUint = typename unsigned_integer_of_size<sizeof(FloatingPoint)>::type;
		return tgmath_impl::isnan_uint_impl(reinterpret_cast<const AsUint&>(x));
#endif
	}
};

template<typename V>
NBL_PARTIAL_REQ_TOP(concepts::Vectorial<V>)
struct isnan_helper<V NBL_PARTIAL_REQ_BOT(concepts::Vectorial<V>) >
{
	using output_t = vector<bool, hlsl::vector_traits<V>::Dimension>;

	static output_t __call(NBL_CONST_REF_ARG(V) vec)
	{
		using traits = hlsl::vector_traits<V>;
		array_get<V, typename traits::scalar_type> getter;
		array_set<output_t, typename traits::scalar_type> setter;

		output_t output;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			setter(output, i, isnan_helper<typename traits::scalar_type>::__call(getter(vec, i)));

		return output;
	}
};

// ISINF

template<typename UnsignedInteger NBL_FUNC_REQUIRES(hlsl::is_integral_v<UnsignedInteger>&& hlsl::is_unsigned_v<UnsignedInteger>)
inline bool isinf_uint_impl(UnsignedInteger val)
{
	using AsFloat = typename float_of_size<sizeof(UnsignedInteger)>::type;
	return (val & (~ieee754::traits<AsFloat>::signMask)) == ieee754::traits<AsFloat>::inf;
}

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct isinf_helper;

template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<FloatingPoint>)
struct isinf_helper<FloatingPoint NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<FloatingPoint>) >
{
	static bool __call(NBL_CONST_REF_ARG(FloatingPoint) x)
	{
#ifdef __HLSL_VERSION
		return spirv::isInf<FloatingPoint>(x);
#else
		// GCC and Clang will always return false with call to std::isinf when fast math is enabled,
		// this implementation will always return appropriate output regardless is fas math is enabled or not
		using AsUint = typename unsigned_integer_of_size<sizeof(FloatingPoint)>::type;
		return tgmath_impl::isinf_uint_impl(reinterpret_cast<const AsUint&>(x));
#endif
	}
};

template<typename V>
NBL_PARTIAL_REQ_TOP(concepts::Vectorial<V>)
struct isinf_helper<V NBL_PARTIAL_REQ_BOT(concepts::Vectorial<V>) >
{
	using output_t = vector<bool, hlsl::vector_traits<V>::Dimension>;

	static output_t __call(NBL_CONST_REF_ARG(V) x)
	{
		using traits = hlsl::vector_traits<V>;
		array_get<V, typename traits::scalar_type> getter;
		array_set<output_t, typename traits::scalar_type> setter;

		output_t output;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			setter(output, i, isinf_helper<typename traits::scalar_type>::__call(getter(x, i)));

		return output;
	}
};

// FLOOR

template<typename V NBL_STRUCT_CONSTRAINABLE>
struct floor_helper;

template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(hlsl::is_floating_point_v<FloatingPoint> && hlsl::is_scalar_v<FloatingPoint>)
struct floor_helper<FloatingPoint NBL_PARTIAL_REQ_BOT(hlsl::is_floating_point_v<FloatingPoint> && hlsl::is_scalar_v<FloatingPoint>) >
{
	static FloatingPoint __call(NBL_CONST_REF_ARG(FloatingPoint) val)
	{
#ifdef __HLSL_VERSION
		return spirv::floor(val);
#else
		return std::floor(val);
#endif
	}
};

template<typename V>
NBL_PARTIAL_REQ_TOP(concepts::Vectorial<V>)
struct floor_helper<V NBL_PARTIAL_REQ_BOT(concepts::Vectorial<V>) >
{
	static V __call(NBL_CONST_REF_ARG(V) vec)
	{
		using traits = hlsl::vector_traits<V>;
		array_get<V, typename traits::scalar_type> getter;
		array_set<V, typename traits::scalar_type> setter;

		V output;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			setter(output, i, floor_helper<typename traits::scalar_type>::__call(getter(vec, i)));

		return output;
	}
};

// ERF

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct erf_helper;

template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<FloatingPoint>)
struct erf_helper<FloatingPoint NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<FloatingPoint>) >
{
	static FloatingPoint __call(NBL_CONST_REF_ARG(FloatingPoint) _x)
	{
#ifdef __HLSL_VERSION
		const FloatingPoint a1 = 0.254829592;
		const FloatingPoint a2 = -0.284496736;
		const FloatingPoint a3 = 1.421413741;
		const FloatingPoint a4 = -1.453152027;
		const FloatingPoint a5 = 1.061405429;
		const FloatingPoint p = 0.3275911;

		FloatingPoint sign = sign(_x);
		FloatingPoint x = abs(_x);

		FloatingPoint t = 1.0 / (1.0 + p * x);
		FloatingPoint y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

		return sign * y;
#else
		return std::erf(_x);
#endif
	}
};

// ERFINV

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct erfInv_helper;

template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<FloatingPoint>)
struct erfInv_helper<FloatingPoint NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<FloatingPoint>) >
{
	static FloatingPoint __call(NBL_CONST_REF_ARG(FloatingPoint) _x)
	{
		FloatingPoint x = clamp<FloatingPoint>(_x, -0.99999, 0.99999);
#ifdef __HLSL_VERSION
		FloatingPoint w = -log((1.0 - x) * (1.0 + x));
#else
		FloatingPoint w = -std::log((1.0 - x) * (1.0 + x));
#endif
		FloatingPoint p;
		if (w < 5.0)
		{
			w -= 2.5;
			p = 2.81022636e-08;
			p = 3.43273939e-07 + p * w;
			p = -3.5233877e-06 + p * w;
			p = -4.39150654e-06 + p * w;
			p = 0.00021858087 + p * w;
			p = -0.00125372503 + p * w;
			p = -0.00417768164 + p * w;
			p = 0.246640727 + p * w;
			p = 1.50140941 + p * w;
		}
		else
		{
#ifdef __HLSL_VERSION
			w = sqrt(w) - 3.0;
#else
			w = std::sqrt(w) - 3.0;
#endif
			p = -0.000200214257;
			p = 0.000100950558 + p * w;
			p = 0.00134934322 + p * w;
			p = -0.00367342844 + p * w;
			p = 0.00573950773 + p * w;
			p = -0.0076224613 + p * w;
			p = 0.00943887047 + p * w;
			p = 1.00167406 + p * w;
			p = 2.83297682 + p * w;
		}
		return p * x;
	}
};

// POW

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct pow_helper;

template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<FloatingPoint> && (sizeof(FloatingPoint) <= 4))
struct pow_helper<FloatingPoint NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<FloatingPoint> && (sizeof(FloatingPoint) <= 4)) >
{
	static FloatingPoint __call(NBL_CONST_REF_ARG(FloatingPoint) x, NBL_CONST_REF_ARG(FloatingPoint) y)
	{
#ifdef __HLSL_VERSION
		return spirv::pow<FloatingPoint>(x, y);
#else
		return std::pow(x, y);
#endif
	}
};


#ifndef __HLSL_VERSION
template<typename Float64>
NBL_PARTIAL_REQ_TOP(is_same_v<Float64, float64_t>)
struct pow_helper<Float64 NBL_PARTIAL_REQ_BOT(is_same_v<Float64, float64_t>) >
{
	static Float64 __call(NBL_CONST_REF_ARG(Float64) x)
	{
		return std::pow(x);
	}
};
#endif

template<typename V>
NBL_PARTIAL_REQ_TOP(concepts::Vectorial<V>)
struct pow_helper<V NBL_PARTIAL_REQ_BOT(concepts::Vectorial<V>) >
{
	static V __call(NBL_CONST_REF_ARG(V) x, NBL_CONST_REF_ARG(V) y)
	{
		using traits = hlsl::vector_traits<V>;
		array_get<V, typename traits::scalar_type> getter;
		array_set<V, typename traits::scalar_type> setter;

		V output;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			setter(output, i, pow_helper<typename traits::scalar_type>::__call(getter(x, i), getter(y, i)));

		return output;
	}
};

// EXP

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct exp_helper;

template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<FloatingPoint> && (sizeof(FloatingPoint) <= 4))
struct exp_helper<FloatingPoint NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<FloatingPoint> && (sizeof(FloatingPoint) <= 4)) >
{
	static FloatingPoint __call(NBL_CONST_REF_ARG(FloatingPoint) x)
	{
#ifdef __HLSL_VERSION
		return spirv::exp<FloatingPoint>(x);
#else
		return std::exp(x);
#endif
	}
};


#ifndef __HLSL_VERSION
template<typename Float64>
NBL_PARTIAL_REQ_TOP(is_same_v<Float64, float64_t>)
struct exp_helper<Float64 NBL_PARTIAL_REQ_BOT(is_same_v<Float64, float64_t>) >
{
	static Float64 __call(NBL_CONST_REF_ARG(Float64) x)
	{
		return std::exp(x);
	}
};
#endif

template<typename V>
NBL_PARTIAL_REQ_TOP(concepts::Vectorial<V>)
struct exp_helper<V NBL_PARTIAL_REQ_BOT(concepts::Vectorial<V>) >
{
	static V __call(NBL_CONST_REF_ARG(V) x)
	{
		using traits = hlsl::vector_traits<V>;
		array_get<V, typename traits::scalar_type> getter;
		array_set<V, typename traits::scalar_type> setter;

		V output;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			setter(output, i, exp_helper<typename traits::scalar_type>::__call(getter(x, i)));

		return output;
	}
};

// EXP2

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct exp2_helper;

template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<FloatingPoint> && (sizeof(FloatingPoint) <= 4))
struct exp2_helper<FloatingPoint NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<FloatingPoint> && (sizeof(FloatingPoint) <= 4)) >
{
	static FloatingPoint __call(NBL_CONST_REF_ARG(FloatingPoint) x)
	{
#ifdef __HLSL_VERSION
		return spirv::exp2<FloatingPoint>(x);
#else
		return std::exp2(x);
#endif
	}
};

#ifndef __HLSL_VERSION
template<typename Float64>
NBL_PARTIAL_REQ_TOP(is_same_v<Float64, float64_t>)
struct exp2_helper<Float64 NBL_PARTIAL_REQ_BOT(is_same_v<Float64, float64_t>) >
{
	static Float64 __call(NBL_CONST_REF_ARG(Float64) x)
	{
		return std::exp2(x);
	}
};
#endif

template<typename Integral>
NBL_PARTIAL_REQ_TOP(hlsl::is_integral_v<Integral> && hlsl::is_scalar_v<Integral>)
struct exp2_helper<Integral NBL_PARTIAL_REQ_BOT(hlsl::is_integral_v<Integral> && hlsl::is_scalar_v<Integral>) >
{
	static Integral __call(NBL_CONST_REF_ARG(Integral) x)
	{
		return _static_cast<Integral>(1ull << x);
	}
};

template<typename V>
NBL_PARTIAL_REQ_TOP(concepts::Vectorial<V>)
struct exp2_helper<V NBL_PARTIAL_REQ_BOT(concepts::Vectorial<V>) >
{
	static V __call(NBL_CONST_REF_ARG(V) vec)
	{
		using traits = hlsl::vector_traits<V>;
		array_get<V, typename traits::scalar_type> getter;
		array_set<V, typename traits::scalar_type> setter;

		V output;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			setter(output, i, exp2_helper<typename traits::scalar_type>::__call(getter(vec, i)));

		return output;
	}
};

// LOG

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct log_helper;

template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<FloatingPoint> && (sizeof(FloatingPoint) <= 4))
struct log_helper<FloatingPoint NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<FloatingPoint> && (sizeof(FloatingPoint) <= 4)) >
{
	static FloatingPoint __call(NBL_CONST_REF_ARG(FloatingPoint) x)
	{
#ifdef __HLSL_VERSION
		return spirv::log<FloatingPoint>(x);
#else
		return std::log(x);
#endif
	}
};

#ifndef __HLSL_VERSION
template<typename Float64>
NBL_PARTIAL_REQ_TOP(is_same_v<Float64, float64_t>)
struct log_helper<Float64 NBL_PARTIAL_REQ_BOT(is_same_v<Float64, float64_t>) >
{
	static Float64 __call(NBL_CONST_REF_ARG(Float64) x)
	{
		return std::log(x);
	}
};
#endif

template<typename V>
NBL_PARTIAL_REQ_TOP(concepts::Vectorial<V>)
struct log_helper<V NBL_PARTIAL_REQ_BOT(concepts::Vectorial<V>) >
{
	static V __call(NBL_CONST_REF_ARG(V) x)
	{
		using traits = hlsl::vector_traits<V>;
		array_get<V, typename traits::scalar_type> getter;
		array_set<V, typename traits::scalar_type> setter;

		V output;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			setter(output, i, log_helper<typename traits::scalar_type>::__call(getter(x, i)));

		return output;
	}
};

// ABS

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct abs_helper;

template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<FloatingPoint>)
struct abs_helper<FloatingPoint NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<FloatingPoint>) >
{
	static FloatingPoint __call(NBL_CONST_REF_ARG(FloatingPoint) x)
	{
#ifdef __HLSL_VERSION
		return spirv::fAbs<FloatingPoint>(x);
#else
		return std::abs(x);
#endif
	}
};

template<typename SignedInteger>
NBL_PARTIAL_REQ_TOP(is_signed_v<SignedInteger> && is_integral_v<SignedInteger> && is_scalar_v<SignedInteger>)
struct abs_helper<SignedInteger NBL_PARTIAL_REQ_BOT(is_signed_v<SignedInteger> && is_integral_v<SignedInteger> && is_scalar_v<SignedInteger>) >
{
	static SignedInteger __call(NBL_CONST_REF_ARG(SignedInteger) x)
	{
#ifdef __HLSL_VERSION
		return spirv::sAbs<SignedInteger>(x);
#else
		return std::abs(x);
#endif
	}
};

template<typename V>
NBL_PARTIAL_REQ_TOP(concepts::Vectorial<V>)
struct abs_helper<V NBL_PARTIAL_REQ_BOT(concepts::Vectorial<V>) >
{
	static V __call(NBL_CONST_REF_ARG(V) vec)
	{
		using traits = hlsl::vector_traits<V>;
		array_get<V, typename traits::scalar_type> getter;
		array_set<V, typename traits::scalar_type> setter;

		V output;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			setter(output, i, abs_helper<typename traits::scalar_type>::__call(getter(vec, i)));

		return output;
	}
};

// SQRT

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct sqrt_helper;

template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<FloatingPoint>)
struct sqrt_helper<FloatingPoint NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<FloatingPoint>) >
{
	static FloatingPoint __call(NBL_CONST_REF_ARG(FloatingPoint) x)
	{
#ifdef __HLSL_VERSION
		return spirv::sqrt(x);
#else
		return std::sqrt(x);
#endif
	}
};

template<typename V>
NBL_PARTIAL_REQ_TOP(concepts::Vectorial<V>)
struct sqrt_helper<V NBL_PARTIAL_REQ_BOT(concepts::Vectorial<V>) >
{
	static V __call(NBL_CONST_REF_ARG(V) vec)
	{
		using traits = hlsl::vector_traits<V>;
		array_get<V, typename traits::scalar_type> getter;
		array_set<V, typename traits::scalar_type> setter;

		V output;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			setter(output, i, sqrt_helper<typename traits::scalar_type>::__call(getter(vec, i)));

		return output;
	}
};

// SIN

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct sin_helper;

template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<FloatingPoint> && (sizeof(FloatingPoint) <= 4))
struct sin_helper<FloatingPoint NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<FloatingPoint> && (sizeof(FloatingPoint) <= 4)) >
{
	static FloatingPoint __call(NBL_CONST_REF_ARG(FloatingPoint) x)
	{
#ifdef __HLSL_VERSION
		return spirv::sin(x);
#else
		return std::sin(x);
#endif
	}
};

template<typename V>
NBL_PARTIAL_REQ_TOP(concepts::Vectorial<V>)
struct sin_helper<V NBL_PARTIAL_REQ_BOT(concepts::Vectorial<V>) >
{
	static V __call(NBL_CONST_REF_ARG(V) vec)
	{
		using traits = hlsl::vector_traits<V>;
		array_get<V, typename traits::scalar_type> getter;
		array_set<V, typename traits::scalar_type> setter;

		V output;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			setter(output, i, sin_helper<typename traits::scalar_type>::__call(getter(vec, i)));

		return output;
	}
};


#ifndef __HLSL_VERSION
template<typename Float64>
NBL_PARTIAL_REQ_TOP(is_same_v<Float64, float64_t>)
struct sin_helper<Float64 NBL_PARTIAL_REQ_BOT(is_same_v<Float64, float64_t>) >
{
	static Float64 __call(NBL_CONST_REF_ARG(Float64) x)
	{
		return std::sin(x);
	}
};
#endif

// COS

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct cos_helper;

template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<FloatingPoint> && (sizeof(FloatingPoint) <= 4))
struct cos_helper<FloatingPoint NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<FloatingPoint> && (sizeof(FloatingPoint) <= 4)) >
{
	static FloatingPoint __call(NBL_CONST_REF_ARG(FloatingPoint) x)
	{
#ifdef __HLSL_VERSION
		return spirv::cos(x);
#else
		return std::cos(x);
#endif
	}
};


#ifndef __HLSL_VERSION
template<typename Float64>
NBL_PARTIAL_REQ_TOP(is_same_v<Float64, float64_t>)
struct cos_helper<Float64 NBL_PARTIAL_REQ_BOT(is_same_v<Float64, float64_t>) >
{
	static Float64 __call(NBL_CONST_REF_ARG(Float64) x)
	{
		return std::cos(x);
	}
};
#endif

template<typename V>
NBL_PARTIAL_REQ_TOP(concepts::Vectorial<V>)
struct cos_helper<V NBL_PARTIAL_REQ_BOT(concepts::Vectorial<V>) >
{
	static V __call(NBL_CONST_REF_ARG(V) vec)
	{
		using traits = hlsl::vector_traits<V>;
		array_get<V, typename traits::scalar_type> getter;
		array_set<V, typename traits::scalar_type> setter;

		V output;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			setter(output, i, cos_helper<typename traits::scalar_type>::__call(getter(vec, i)));

		return output;
	}
};

// ACOS

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct acos_helper;

template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<FloatingPoint> && (sizeof(FloatingPoint) <= 4))
struct acos_helper<FloatingPoint NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<FloatingPoint> && (sizeof(FloatingPoint) <= 4)) >
{
	static FloatingPoint __call(NBL_CONST_REF_ARG(FloatingPoint) x)
	{
#ifdef __HLSL_VERSION
		return spirv::acos(x);
#else
		return std::acos(x);
#endif
	}
};

#ifndef __HLSL_VERSION
template<typename Float64>
NBL_PARTIAL_REQ_TOP(is_same_v<Float64, float64_t>)
struct acos_helper<Float64 NBL_PARTIAL_REQ_BOT(is_same_v<Float64, float64_t>) >
{
	static Float64 __call(NBL_CONST_REF_ARG(Float64) x)
	{
		return std::acos(x);
	}
};
#endif

template<typename V>
NBL_PARTIAL_REQ_TOP(concepts::Vectorial<V>)
struct acos_helper<V NBL_PARTIAL_REQ_BOT(concepts::Vectorial<V>) >
{
	static V __call(NBL_CONST_REF_ARG(V) vec)
	{
		using traits = hlsl::vector_traits<V>;
		array_get<V, typename traits::scalar_type> getter;
		array_set<V, typename traits::scalar_type> setter;

		V output;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			setter(output, i, acos_helper<typename traits::scalar_type>::__call(getter(vec, i)));

		return output;
	}
};

}
}
}

#endif