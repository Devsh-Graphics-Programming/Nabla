#ifndef _NBL_BUILTIN_HLSL_TGMATH_IMPL_INCLUDED_
#define _NBL_BUILTIN_HLSL_TGMATH_IMPL_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/basic.h>
#include <nbl/builtin/hlsl/concepts.hlsl>
#include <nbl/builtin/hlsl/concepts/vector.hlsl>
#include <nbl/builtin/hlsl/spirv_intrinsics/core.hlsl>
#include <nbl/builtin/hlsl/spirv_intrinsics/glsl.std.450.hlsl>
#include <nbl/builtin/hlsl/ieee754.hlsl>

// C++ includes
#ifndef __HLSL_VERSION
#include <cmath>
#include <tgmath.h>
#endif

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

template<typename UnsignedInteger NBL_FUNC_REQUIRES(hlsl::is_integral_v<UnsignedInteger> && hlsl::is_unsigned_v<UnsignedInteger>)
inline bool isnan_uint_impl(UnsignedInteger val)
{
	using AsFloat = typename float_of_size<sizeof(UnsignedInteger)>::type;
	UnsignedInteger absVal = val & (hlsl::numeric_limits<UnsignedInteger>::max >> 1);
	return absVal > (ieee754::traits<AsFloat>::specialValueExp << ieee754::traits<AsFloat>::mantissaBitCnt);
}
template<typename UnsignedInteger NBL_FUNC_REQUIRES(hlsl::is_integral_v<UnsignedInteger>&& hlsl::is_unsigned_v<UnsignedInteger>)
inline bool isinf_uint_impl(UnsignedInteger val)
{
	using AsFloat = typename float_of_size<sizeof(UnsignedInteger)>::type;
	return (val & (~ieee754::traits<AsFloat>::signMask)) == ieee754::traits<AsFloat>::inf;
}

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct erf_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct erfInv_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct isnan_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct isinf_helper;
template<typename V NBL_STRUCT_CONSTRAINABLE>
struct floor_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct pow_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct exp_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct exp2_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct log_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct abs_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct cos_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct sin_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct acos_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct sqrt_helper;

#ifdef __HLSL_VERSION
template<typename T> NBL_PARTIAL_REQ_TOP(always_true<decltype(spirv::cos<T>(experimental::declval<T>()))>)
struct cos_helper<T NBL_PARTIAL_REQ_BOT(always_true<decltype(spirv::cos<T>(experimental::declval<T>()))>) >
{
	static inline T __call(const T arg)
	{
		return spirv::cos<T>(arg);
	}
};
template<typename T> NBL_PARTIAL_REQ_TOP(always_true<decltype(spirv::sin<T>(experimental::declval<T>()))>)
struct sin_helper<T NBL_PARTIAL_REQ_BOT(always_true<decltype(spirv::sin<T>(experimental::declval<T>()))>) >
{
	static inline T __call(const T arg)
	{
		return spirv::sin<T>(arg);
	}
};
template<typename T>
NBL_PARTIAL_REQ_TOP(always_true<decltype(spirv::sAbs<T>(experimental::declval<T>()))>)
struct abs_helper<T NBL_PARTIAL_REQ_BOT(always_true<decltype(spirv::sAbs<T>(experimental::declval<T>()))>) >
{
	static T __call(NBL_CONST_REF_ARG(T) x)
	{
		return spirv::sAbs<T>(x);
	}
};
template<typename T> NBL_PARTIAL_REQ_TOP(always_true<decltype(spirv::fAbs<T>(experimental::declval<T>()))>)
struct abs_helper<T NBL_PARTIAL_REQ_BOT(always_true<decltype(spirv::fAbs<T>(experimental::declval<T>()))>) >
{
	static T __call(NBL_CONST_REF_ARG(T) x)
	{
		return spirv::fAbs<T>(x);
	}
};
template<typename T> NBL_PARTIAL_REQ_TOP(always_true<decltype(spirv::sqrt<T>(experimental::declval<T>()))>)
struct sqrt_helper<T NBL_PARTIAL_REQ_BOT(always_true<decltype(spirv::sqrt<T>(experimental::declval<T>()))>) >
{
	static inline T __call(const T arg)
	{
		return spirv::sqrt<T>(arg);
	}
};
template<typename T> NBL_PARTIAL_REQ_TOP(always_true<decltype(spirv::log<T>(experimental::declval<T>()))>)
struct log_helper<T NBL_PARTIAL_REQ_BOT(always_true<decltype(spirv::log<T>(experimental::declval<T>()))>) >
{
	static inline T __call(const T arg)
	{
		return spirv::log<T>(arg);
	}
};
template<typename T> NBL_PARTIAL_REQ_TOP(always_true<decltype(spirv::exp2<T>(experimental::declval<T>()))>)
struct exp2_helper<T NBL_PARTIAL_REQ_BOT(always_true<decltype(spirv::exp2<T>(experimental::declval<T>()))>) >
{
	static inline T __call(const T arg)
	{
		return spirv::exp2<T>(arg);
	}
};
template<typename T> NBL_PARTIAL_REQ_TOP(always_true<decltype(spirv::exp<T>(experimental::declval<T>()))>)
struct exp_helper<T NBL_PARTIAL_REQ_BOT(always_true<decltype(spirv::exp<T>(experimental::declval<T>()))>) >
{
	static inline T __call(const T arg)
	{
		return spirv::exp<T>(arg);
	}
};
template<typename T> NBL_PARTIAL_REQ_TOP(always_true<decltype(spirv::pow<T>(experimental::declval<T>(), experimental::declval<T>()))>)
struct pow_helper<T NBL_PARTIAL_REQ_BOT(always_true<decltype(spirv::pow<T>(experimental::declval<T>(), experimental::declval<T>()))>) >
{
	static inline T __call(const T x, const T y)
	{
		return spirv::pow<T>(x, y);
	}
};
template<typename T> NBL_PARTIAL_REQ_TOP(always_true<decltype(spirv::floor<T>(experimental::declval<T>()))>)
struct floor_helper<T NBL_PARTIAL_REQ_BOT(always_true<decltype(spirv::floor<T>(experimental::declval<T>()))>) >
{
	static inline T __call(const T arg)
	{
		return spirv::floor<T>(arg);
	}
};
template<typename T> NBL_PARTIAL_REQ_TOP(always_true<decltype(spirv::isInf<T>(experimental::declval<T>()))>)
struct isinf_helper<T NBL_PARTIAL_REQ_BOT(always_true<decltype(spirv::isInf<T>(experimental::declval<T>()))>) >
{
	static inline T __call(const T arg)
	{
		return spirv::isInf<T>(arg);
	}
};
template<typename T> NBL_PARTIAL_REQ_TOP(always_true<decltype(spirv::isNan<T>(experimental::declval<T>()))>)
struct isnan_helper<T NBL_PARTIAL_REQ_BOT(always_true<decltype(spirv::isNan<T>(experimental::declval<T>()))>) >
{
	static inline T __call(const T arg)
	{
		return spirv::isNan<T>(arg);
	}
};

#else // C++
template<typename FloatingPoint>
requires concepts::FloatingPointScalar<FloatingPoint>
struct cos_helper<FloatingPoint>
{
	static inline FloatingPoint __call(const FloatingPoint arg)
	{
		return std::cos(arg);
	}
};

template<typename FloatingPoint>
requires concepts::FloatingPointScalar<FloatingPoint>
struct sin_helper<FloatingPoint>
{
    static inline FloatingPoint __call(const FloatingPoint arg)
    {
        return std::sin(arg);
    }
};

template<typename FloatingPoint>
requires concepts::FloatingPointScalar<FloatingPoint>
struct acos_helper<FloatingPoint>
{
	static inline FloatingPoint __call(const FloatingPoint arg)
	{
		return std::acos(arg);
	}
};

template<typename FloatingPoint>
requires concepts::FloatingPointScalar<FloatingPoint>
struct sqrt_helper<FloatingPoint>
{
	static inline FloatingPoint __call(const FloatingPoint arg)
	{
		return std::sqrt(arg);
	}
};

template<typename T>
requires concepts::Scalar<T>
struct abs_helper<T>
{
	static inline T __call(const T arg)
	{
		return std::abs(arg);
	}
};

template<typename T>
requires concepts::Scalar<T>
struct log_helper<T>
{
	static inline T __call(const T arg)
	{
		return std::log(arg);
	}
};

template<typename T>
requires concepts::Scalar<T>
struct exp2_helper<T>
{
	static inline T __call(const T arg)
	{
		return std::exp2(arg);
	}
};

template<typename T>
requires concepts::Scalar<T>
struct exp_helper<T>
{
	static inline T __call(const T arg)
	{
		return std::exp(arg);
	}
};

template<typename T>
requires concepts::FloatingPointScalar<T>
struct pow_helper<T>
{
	static inline T __call(const T x, const T y)
	{
		return std::pow(x, y);
	}
};

template<typename T>
requires concepts::FloatingPointScalar<T>
struct floor_helper<T>
{
	static inline T __call(const T arg)
	{
		return std::floor(arg);
	}
};

template<typename T>
requires concepts::FloatingPointScalar<T>
struct isinf_helper<T>
{
	static inline T __call(const T arg)
	{
		// GCC and Clang will always return false with call to std::isinf when fast math is enabled,
		// this implementation will always return appropriate output regardless is fas math is enabled or not
		using AsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
		return tgmath_impl::isinf_uint_impl(reinterpret_cast<const AsUint&>(arg));
	}
};

template<typename T>
requires concepts::FloatingPointScalar<T>
struct isnan_helper<T>
{
	static inline T __call(const T arg)
	{
		// GCC and Clang will always return false with call to std::isnan when fast math is enabled,
		// this implementation will always return appropriate output regardless is fas math is enabled or not
		using AsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
		return tgmath_impl::isnan_uint_impl(reinterpret_cast<const AsUint&>(arg));
	}
};

#endif

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
template<typename V>
NBL_PARTIAL_REQ_TOP(concepts::Vectorial<V>)
struct sin_helper<V NBL_PARTIAL_REQ_BOT(concepts::Vectorial< V>) >
{
    static V __call(NBL_CONST_REF_ARG( V) vec)
	{
		using traits = hlsl::vector_traits < V >;
		array_get<V, typename traits::scalar_type> getter;
		array_set<V, typename traits::scalar_type> setter;

		V output;
		for(uint32_t i = 0;i < traits::Dimension;++i)
			setter(output, i, sin_helper<typename traits::scalar_type>::__call(getter(vec, i)));

		return output;
	}
};
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

}
}
}

#endif