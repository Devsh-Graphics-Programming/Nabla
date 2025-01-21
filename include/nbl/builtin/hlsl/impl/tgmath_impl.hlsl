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
template<typename T, typename U NBL_STRUCT_CONSTRAINABLE>
struct lerp_helper;

#ifdef __HLSL_VERSION

#define AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(HELPER_NAME, SPIRV_FUNCTION_NAME, RETURN_TYPE)\
template<typename T> NBL_PARTIAL_REQ_TOP(always_true<decltype(spirv::SPIRV_FUNCTION_NAME<T>(experimental::declval<T>()))>)\
struct HELPER_NAME<T NBL_PARTIAL_REQ_BOT(always_true<decltype(spirv::SPIRV_FUNCTION_NAME<T>(experimental::declval<T>()))>) >\
{\
	using return_t = RETURN_TYPE;\
	static inline return_t __call(const T arg)\
	{\
		return spirv::SPIRV_FUNCTION_NAME<T>(arg);\
	}\
};

AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(cos_helper, cos, T)
AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(sin_helper, sin, T)
AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(abs_helper, sAbs, T)
AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(abs_helper, fAbs, T)
AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(sqrt_helper, sqrt, T)
AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(log_helper, log, T)
AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(exp2_helper, exp2, T)
AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(exp_helper, exp, T)
AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(pow_helper, pow, T)
AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(floor_helper, floor, T)
#define ISINF_AND_ISNAN_RETURN_TYPE conditional_t<is_vector_v<T>, vector<bool, vector_traits<T>::Dimension>, bool>
AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(isinf_helper, isInf, ISINF_AND_ISNAN_RETURN_TYPE)
AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(isnan_helper, isNan, ISINF_AND_ISNAN_RETURN_TYPE)
#undef ISINF_AND_ISNAN_RETURN_TYPE 
#undef AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER

template<typename T, typename U> NBL_PARTIAL_REQ_TOP(always_true<decltype(spirv::fMix<T>(experimental::declval<T>(), experimental::declval<T>(), experimental::declval<U>()))>)
struct lerp_helper<T, U NBL_PARTIAL_REQ_BOT(always_true<decltype(spirv::fMix<T>(experimental::declval<T>(), experimental::declval<T>(), experimental::declval<U>()))>) >
{
	using return_t = conditional_t<is_vector_v<T>, vector<bool, vector_traits<T>::Dimension>, bool>;
	static inline return_t __call(const T x, const T y, const U a)
	{
		return spirv::fMix<T>(x, y, a);
	}
};

#else // C++ only specializations

#define AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(HELPER_NAME, REQUIREMENT, STD_FUNCTION_NAME, RETURN_TYPE)\
template<typename T>\
requires REQUIREMENT \
struct HELPER_NAME<T>\
{\
	using return_t = RETURN_TYPE;\
	static inline return_t __call(const T arg)\
	{\
		return std::STD_FUNCTION_NAME(arg);\
	}\
};

AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(cos_helper, concepts::FloatingPointScalar<T>, cos, T)
AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(sin_helper, concepts::FloatingPointScalar<T>, sin, T)
AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(acos_helper, concepts::FloatingPointScalar<T>, acos, T)
AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(sqrt_helper, concepts::FloatingPointScalar<T>, sqrt, T)
AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(abs_helper, concepts::Scalar<T>, abs, T)
AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(log_helper, concepts::Scalar<T>, log, T)
AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(exp2_helper, concepts::Scalar<T>, exp2, T)
AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(exp_helper, concepts::Scalar<T>, exp, T)
AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(pow_helper, concepts::FloatingPointScalar<T>, pow, T)
AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(floor_helper, concepts::FloatingPointScalar<T>, floor, T)
#undef AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER

template<typename T>
requires concepts::FloatingPointScalar<T>
struct isinf_helper<T>
{
	using return_t = bool;
	static inline return_t __call(const T arg)
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
	using return_t = bool;
	static inline return_t __call(const T arg)
	{
		// GCC and Clang will always return false with call to std::isnan when fast math is enabled,
		// this implementation will always return appropriate output regardless is fas math is enabled or not
		using AsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
		return tgmath_impl::isnan_uint_impl(reinterpret_cast<const AsUint&>(arg));
	}
};

template<typename T, typename U>
requires concepts::FloatingPoint<T> && (concepts::FloatingPoint<T> || concepts::Boolean<T>)
struct lerp_helper<T, U>
{
	using return_t = T;
	static inline return_t __call(const T x, const T y, const U a)
	{
		return glm::mix(x, y ,a);
	}
};

#endif

// C++ and HLSL specializations

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

#define AUTO_SPECIALIZE_HELPER_FOR_VECTOR(HELPER_NAME)\
template<typename Vectorial>\
NBL_PARTIAL_REQ_TOP(concepts::Vectorial<Vectorial>)\
struct HELPER_NAME<Vectorial NBL_PARTIAL_REQ_BOT(concepts::Vectorial<Vectorial>) >\
{\
	static Vectorial __call(NBL_CONST_REF_ARG(Vectorial) vec)\
	{\
		using traits = hlsl::vector_traits<Vectorial>;\
		array_get<Vectorial, typename traits::scalar_type> getter;\
		array_set<Vectorial, typename traits::scalar_type> setter;\
\
		Vectorial output;\
		for (uint32_t i = 0; i < traits::Dimension; ++i)\
			setter(output, i, HELPER_NAME<typename traits::scalar_type>::__call(getter(vec, i)));\
\
		return output;\
	}\
};

AUTO_SPECIALIZE_HELPER_FOR_VECTOR(cos_helper)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(sin_helper)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(acos_helper)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(sqrt_helper)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(abs_helper)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(log_helper)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(exp2_helper)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(exp_helper)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(pow_helper)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(floor_helper)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(isinf_helper)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(isnan_helper)
#undef AUTO_SPECIALIZE_HELPER_FOR_VECTOR

}
}
}

#endif