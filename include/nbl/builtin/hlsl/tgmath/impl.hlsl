#ifndef _NBL_BUILTIN_HLSL_TGMATH_IMPL_INCLUDED_
#define _NBL_BUILTIN_HLSL_TGMATH_IMPL_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/basic.h>
#include <nbl/builtin/hlsl/concepts.hlsl>
#include <nbl/builtin/hlsl/concepts/vector.hlsl>
#include <nbl/builtin/hlsl/spirv_intrinsics/core.hlsl>
#include <nbl/builtin/hlsl/spirv_intrinsics/glsl.std.450.hlsl>
#include <nbl/builtin/hlsl/tgmath/output_structs.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/intrinsics.hlsl>

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
struct log2_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct abs_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct cos_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct sin_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct acos_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct tan_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct asin_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct atan_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct sinh_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct cosh_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct tanh_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct asinh_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct acosh_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct atanh_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct atan2_helper;

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct sqrt_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct modf_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct round_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct roundEven_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct trunc_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct ceil_helper;
template<typename T, typename U NBL_STRUCT_CONSTRAINABLE>
struct ldexp_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct modfStruct_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct frexpStruct_helper;

#ifdef __HLSL_VERSION

#define DECLVAL(r,data,i,_T) BOOST_PP_COMMA_IF(BOOST_PP_NOT_EQUAL(i,0)) experimental::declval<_T>()
#define DECL_ARG(r,data,i,_T) BOOST_PP_COMMA_IF(BOOST_PP_NOT_EQUAL(i,0)) const _T arg##i
#define WRAP(r,data,i,_T) BOOST_PP_COMMA_IF(BOOST_PP_NOT_EQUAL(i,0)) _T
#define ARG(r,data,i,_T) BOOST_PP_COMMA_IF(BOOST_PP_NOT_EQUAL(i,0)) arg##i

// the template<> needs to be written ourselves
// return type is __VA_ARGS__ to protect against `,` in templated return types
#define AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(HELPER_NAME, SPIRV_FUNCTION_NAME, ARG_TYPE_LIST, ARG_TYPE_SET, ...)\
NBL_PARTIAL_REQ_TOP(is_same_v<decltype(spirv::SPIRV_FUNCTION_NAME<T>(BOOST_PP_SEQ_FOR_EACH_I(DECLVAL, _, ARG_TYPE_SET))), __VA_ARGS__ >) \
struct HELPER_NAME<BOOST_PP_SEQ_FOR_EACH_I(WRAP, _, ARG_TYPE_LIST) NBL_PARTIAL_REQ_BOT(is_same_v<decltype(spirv::SPIRV_FUNCTION_NAME<T>(BOOST_PP_SEQ_FOR_EACH_I(DECLVAL, _, ARG_TYPE_SET))), __VA_ARGS__ >) >\
{\
	using return_t = __VA_ARGS__;\
	static inline return_t __call( BOOST_PP_SEQ_FOR_EACH_I(DECL_ARG, _, ARG_TYPE_SET) )\
	{\
		return spirv::SPIRV_FUNCTION_NAME<T>( BOOST_PP_SEQ_FOR_EACH_I(ARG, _, ARG_TYPE_SET) );\
	}\
};

template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(sin_helper, sin, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(cos_helper, cos, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(acos_helper, acos, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(tan_helper, tan, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(asin_helper, asin, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(atan_helper, atan, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(sinh_helper, sinh, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(cosh_helper, cosh, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(tanh_helper, tanh, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(asinh_helper, asinh, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(acosh_helper, acosh, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(atanh_helper, atanh, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(atan2_helper, atan2, (T), (T)(T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(abs_helper, sAbs, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(abs_helper, fAbs, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(sqrt_helper, sqrt, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(log_helper, log, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(log2_helper, log2, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(exp2_helper, exp2, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(exp_helper, exp, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(floor_helper, floor, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(round_helper, round, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(roundEven_helper, roundEven, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(trunc_helper, trunc, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(ceil_helper, ceil, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(pow_helper, pow, (T), (T)(T), T)
template<typename T, typename U> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(ldexp_helper, ldexp, (T)(U), (T)(U), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(modfStruct_helper, modfStruct, (T), (T), ModfOutput<T>)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(frexpStruct_helper, frexpStruct, (T), (T), FrexpOutput<T>)

#define ISINF_AND_ISNAN_RETURN_TYPE conditional_t<is_vector_v<T>, vector<bool, vector_traits<T>::Dimension>, bool>
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(isinf_helper, isInf, (T), (T), ISINF_AND_ISNAN_RETURN_TYPE)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(isnan_helper, isNan, (T), (T), ISINF_AND_ISNAN_RETURN_TYPE)
#undef ISINF_AND_ISNAN_RETURN_TYPE 

#undef DECLVAL
#undef DECL_ARG
#undef WRAP
#undef ARG
#undef AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER

template<typename T> NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<T>)
struct modf_helper<T NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<T>) >
{
	using return_t = T;
	static inline return_t __call(const T x)
	{
		ModfOutput<T> output = modfStruct_helper<T>::__call(x);
		return output.fractionalPart;
	}
};

template<typename T> NBL_PARTIAL_REQ_TOP(concepts::FloatingPoint<T> && is_vector_v<T>)
struct modf_helper<T NBL_PARTIAL_REQ_BOT(concepts::FloatingPoint<T> && is_vector_v<T>) >
{
	using return_t = T;
	static inline return_t __call(const T x)
	{
		using traits = hlsl::vector_traits<T>;
		array_get<T, typename traits::scalar_type> getter;
		array_set<T, typename traits::scalar_type> setter;

		return_t output;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			setter(output, i, modf_helper<typename traits::scalar_type>::__call(getter(x, i)));

		return output;
	}
};

template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<FloatingPoint>)
struct erf_helper<FloatingPoint NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<FloatingPoint>) >
{
	static FloatingPoint __call(NBL_CONST_REF_ARG(FloatingPoint) _x)
	{
		const FloatingPoint a1 = FloatingPoint(NBL_FP64_LITERAL(0.254829592));
		const FloatingPoint a2 = FloatingPoint(NBL_FP64_LITERAL(-0.284496736));
		const FloatingPoint a3 = FloatingPoint(NBL_FP64_LITERAL(1.421413741));
		const FloatingPoint a4 = FloatingPoint(NBL_FP64_LITERAL(-1.453152027));
		const FloatingPoint a5 = FloatingPoint(NBL_FP64_LITERAL(1.061405429));
		const FloatingPoint p = FloatingPoint(NBL_FP64_LITERAL(0.3275911));

		FloatingPoint _sign = FloatingPoint(sign(_x));
		FloatingPoint x = abs(_x);

		FloatingPoint t = FloatingPoint(NBL_FP64_LITERAL(1.0)) / (FloatingPoint(NBL_FP64_LITERAL(1.0)) + p * x);
		FloatingPoint y = FloatingPoint(NBL_FP64_LITERAL(1.0)) - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

		return _sign * y;
	}
};

#else // C++ only specializations

#define DECL_ARG(r,data,i,_T) BOOST_PP_COMMA_IF(BOOST_PP_NOT_EQUAL(i,0)) const _T arg##i
#define WRAP(r,data,i,_T) BOOST_PP_COMMA_IF(BOOST_PP_NOT_EQUAL(i,0)) _T
#define ARG(r,data,i,_T) BOOST_PP_COMMA_IF(BOOST_PP_NOT_EQUAL(i,0)) arg##i

// not giving an explicit template parameter to std function below because not every function used here is templated
#define AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(HELPER_NAME, STD_FUNCTION_NAME, REQUIREMENT, ARG_TYPE_LIST, ARG_TYPE_SET, ...)\
requires REQUIREMENT \
struct HELPER_NAME<BOOST_PP_SEQ_FOR_EACH_I(WRAP, _, ARG_TYPE_LIST)>\
{\
	using return_t = __VA_ARGS__;\
	static inline return_t __call( BOOST_PP_SEQ_FOR_EACH_I(DECL_ARG, _, ARG_TYPE_SET) )\
	{\
		return std::STD_FUNCTION_NAME( BOOST_PP_SEQ_FOR_EACH_I(ARG, _, ARG_TYPE_SET) );\
	}\
};

template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(cos_helper, cos, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(sin_helper, sin, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(tan_helper, tan, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(asin_helper, asin, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(acos_helper, acos, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(atan_helper, atan, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(sinh_helper, sinh, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(cosh_helper, cosh, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(tanh_helper, tanh, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(asinh_helper, asinh, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(acosh_helper, acosh, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(atanh_helper, atanh, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(atan2_helper, atan2, concepts::FloatingPointScalar<T>, (T), (T)(T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(sqrt_helper, sqrt, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(abs_helper, abs, concepts::Scalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(log_helper, log, concepts::Scalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(log2_helper, log2, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(exp2_helper, exp2, concepts::Scalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(exp_helper, exp, concepts::Scalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(floor_helper, floor, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(round_helper, round, concepts::FloatingPointScalar<T>, (T), (T), T)
// TODO: uncomment when C++23
//template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(roundEven_helper, roundeven, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(trunc_helper, trunc, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(ceil_helper, ceil, concepts::FloatingPointScalar<T>, (T), (T), T)

#undef DECL_ARG
#undef WRAP
#undef ARG
#undef AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER

template<typename T>
requires concepts::FloatingPointScalar<T>
struct pow_helper<T>
{
	using return_t = T;
	static inline return_t __call(const T x, const T y)
	{
		return std::pow<T>(x, y);
	}
};

template<typename T>
requires concepts::FloatingPointScalar<T>
struct modf_helper<T>
{
	using return_t = T;
	static inline return_t __call(const T x)
	{
		T tmp;
		return std::modf(x, &tmp);
	}
};

template<typename T>
requires concepts::FloatingPointScalar<T>
struct isinf_helper<T>
{
	using return_t = bool;
	static inline return_t __call(const T arg)
	{
		// GCC and Clang will always return false with call to std::isinf when fast math is enabled,
		// this implementation will always return appropriate output regardless is fast math is enabled or not
		using AsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
		return cpp_compat_intrinsics_impl::isinf_uint_impl(reinterpret_cast<const AsUint&>(arg));
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
		// this implementation will always return appropriate output regardless is fast math is enabled or not
		using AsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
		return cpp_compat_intrinsics_impl::isnan_uint_impl(reinterpret_cast<const AsUint&>(arg));
	}
};

template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<FloatingPoint>)
struct erf_helper<FloatingPoint NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<FloatingPoint>) >
{
	static FloatingPoint __call(NBL_CONST_REF_ARG(FloatingPoint) x)
	{
		return std::erf(x);
	}
};

// TODO: remove when C++23
template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<FloatingPoint>)
struct roundEven_helper<FloatingPoint NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<FloatingPoint>) >
{
	static FloatingPoint __call(NBL_CONST_REF_ARG(FloatingPoint) x)
	{
		// TODO: no way this is optimal, find a better implementation
		float tmp;
		if (std::abs(std::modf(x, &tmp)) == 0.5f)
		{
			int32_t result = static_cast<int32_t>(x);
			if (result % 2 != 0)
				result >= 0 ? ++result : --result;
			return result;
		}

		return std::round(x);
	}
};

template<typename T, typename U>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<T> && concepts::IntegralScalar<U>)
struct ldexp_helper<T, U NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<T> && concepts::IntegralScalar<U>) >
{
	static T __call(NBL_CONST_REF_ARG(T) arg, NBL_CONST_REF_ARG(U) exp)
	{
		return std::ldexp(arg, exp);
	}
};

template<typename T>
requires concepts::FloatingPointScalar<T>
struct modfStruct_helper<T>
{
	using return_t = ModfOutput<T>;
	static inline return_t __call(const T val)
	{
		return_t output;
		output.fractionalPart = std::modf(val, &output.wholeNumberPart);

		return output;
	}
};

template<typename T>
requires concepts::FloatingPointScalar<T>
struct frexpStruct_helper<T>
{
	using return_t = FrexpOutput<T>;
	static inline return_t __call(const T val)
	{
		return_t output;
		output.significand = std::frexp(val, &output.exponent);

		return output;
	}
};

#endif // C++ only specializations

// C++ and HLSL specializations

template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<FloatingPoint>)
struct erfInv_helper<FloatingPoint NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<FloatingPoint>) >
{
	static FloatingPoint __call(NBL_CONST_REF_ARG(FloatingPoint) _x)
	{
		FloatingPoint x = clamp<FloatingPoint>(_x, FloatingPoint(NBL_FP64_LITERAL(-0.99999)), FloatingPoint(NBL_FP64_LITERAL(0.99999)));

		FloatingPoint w = -log_helper<FloatingPoint>::__call((FloatingPoint(NBL_FP64_LITERAL(1.0)) - x) * (FloatingPoint(NBL_FP64_LITERAL(1.0)) + x));
		FloatingPoint p;
		if (w < 5.0)
		{
			w -= FloatingPoint(NBL_FP64_LITERAL(2.5));
			p = FloatingPoint(NBL_FP64_LITERAL(2.81022636e-08));
			p = FloatingPoint(NBL_FP64_LITERAL(3.43273939e-07)) + p * w;
			p = FloatingPoint(NBL_FP64_LITERAL(-3.5233877e-06)) + p * w;
			p = FloatingPoint(NBL_FP64_LITERAL(-4.39150654e-06)) + p * w;
			p = FloatingPoint(NBL_FP64_LITERAL(0.00021858087)) + p * w;
			p = FloatingPoint(NBL_FP64_LITERAL(-0.00125372503)) + p * w;
			p = FloatingPoint(NBL_FP64_LITERAL(-0.00417768164)) + p * w;
			p = FloatingPoint(NBL_FP64_LITERAL(0.246640727)) + p * w;
			p = FloatingPoint(NBL_FP64_LITERAL(1.50140941)) + p * w;
		}
		else
		{
			w = sqrt_helper<FloatingPoint>::__call(w) - FloatingPoint(NBL_FP64_LITERAL(3.0));
			p = FloatingPoint(NBL_FP64_LITERAL(-0.000200214257));
			p = FloatingPoint(NBL_FP64_LITERAL(0.000100950558)) + p * w;
			p = FloatingPoint(NBL_FP64_LITERAL(0.00134934322)) + p * w;
			p = FloatingPoint(NBL_FP64_LITERAL(-0.00367342844)) + p * w;
			p = FloatingPoint(NBL_FP64_LITERAL(0.00573950773)) + p * w;
			p = FloatingPoint(NBL_FP64_LITERAL(-0.0076224613)) + p * w;
			p = FloatingPoint(NBL_FP64_LITERAL(0.00943887047)) + p * w;
			p = FloatingPoint(NBL_FP64_LITERAL(1.00167406)) + p * w;
			p = FloatingPoint(NBL_FP64_LITERAL(2.83297682)) + p * w;
		}
		return p * x;
	}
};

#ifdef __HLSL_VERSION
// SPIR-V already defines specializations for builtin vector types
#define VECTOR_SPECIALIZATION_CONCEPT concepts::Vectorial<T> && !is_vector_v<T>
#else
#define VECTOR_SPECIALIZATION_CONCEPT concepts::Vectorial<T>
#endif

#define AUTO_SPECIALIZE_HELPER_FOR_VECTOR(HELPER_NAME, CONCEPT, RETURN_TYPE)\
template<typename T>\
NBL_PARTIAL_REQ_TOP(CONCEPT)\
struct HELPER_NAME<T NBL_PARTIAL_REQ_BOT(CONCEPT) >\
{\
	using return_t = RETURN_TYPE;\
	static return_t __call(NBL_CONST_REF_ARG(T) vec)\
	{\
		using traits = hlsl::vector_traits<T>;\
		using return_t_traits = hlsl::vector_traits<return_t>;\
		array_get<T, typename traits::scalar_type> getter;\
		array_set<return_t, typename return_t_traits::scalar_type> setter;\
\
		return_t output;\
		for (uint32_t i = 0; i < traits::Dimension; ++i)\
			setter(output, i, HELPER_NAME<typename traits::scalar_type>::__call(getter(vec, i)));\
\
		return output;\
	}\
};

AUTO_SPECIALIZE_HELPER_FOR_VECTOR(sqrt_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(abs_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(log_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(log2_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(exp2_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(exp_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(floor_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
#define INT_VECTOR_RETURN_TYPE vector<int32_t, vector_traits<T>::Dimension>
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(isinf_helper, VECTOR_SPECIALIZATION_CONCEPT, INT_VECTOR_RETURN_TYPE)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(isnan_helper, VECTOR_SPECIALIZATION_CONCEPT, INT_VECTOR_RETURN_TYPE)
#undef INT_VECTOR_RETURN_TYPE
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(cos_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(sin_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(acos_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(tan_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(asin_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(atan_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(sinh_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(cosh_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(tanh_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(asinh_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(acosh_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(atanh_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(modf_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(round_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(roundEven_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(trunc_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(ceil_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(erf_helper, concepts::Vectorial<T>, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(erfInv_helper, concepts::Vectorial<T>, T)


#undef AUTO_SPECIALIZE_HELPER_FOR_VECTOR

template<typename T>
NBL_PARTIAL_REQ_TOP(VECTOR_SPECIALIZATION_CONCEPT)
struct pow_helper<T NBL_PARTIAL_REQ_BOT(VECTOR_SPECIALIZATION_CONCEPT) >
{
	using return_t = T;
	static return_t __call(NBL_CONST_REF_ARG(T) x, NBL_CONST_REF_ARG(T) y)
	{
		using traits = hlsl::vector_traits<T>;
		array_get<T, typename traits::scalar_type> getter;
		array_set<T, typename traits::scalar_type> setter;
		
		return_t output;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			setter(output, i, pow_helper<typename traits::scalar_type>::__call(getter(x, i), getter(y, i)));
	
		return output;
	}
};

template<typename T, typename U>
NBL_PARTIAL_REQ_TOP(VECTOR_SPECIALIZATION_CONCEPT && (vector_traits<T>::Dimension == vector_traits<U>::Dimension))
struct ldexp_helper<T, U NBL_PARTIAL_REQ_BOT(VECTOR_SPECIALIZATION_CONCEPT && (vector_traits<T>::Dimension == vector_traits<U>::Dimension)) >
{
	using return_t = T;
	static return_t __call(NBL_CONST_REF_ARG(T) arg, NBL_CONST_REF_ARG(U) exp)
	{
		using arg_traits = hlsl::vector_traits<T>;
		using exp_traits = hlsl::vector_traits<U>;
		array_get<T, typename arg_traits::scalar_type> argGetter;
		array_get<U, typename exp_traits::scalar_type> expGetter;
		array_set<T, typename arg_traits::scalar_type> setter;

		return_t output;
		for (uint32_t i = 0; i < arg_traits::Dimension; ++i)
			setter(output, i, ldexp_helper<typename arg_traits::scalar_type, typename exp_traits::scalar_type>::__call(argGetter(arg, i), expGetter(exp, i)));

		return output;
	}
};

template<typename T>
NBL_PARTIAL_REQ_TOP(VECTOR_SPECIALIZATION_CONCEPT)
struct modfStruct_helper<T NBL_PARTIAL_REQ_BOT(VECTOR_SPECIALIZATION_CONCEPT) >
{
	using return_t = ModfOutput<T>;
	static return_t __call(NBL_CONST_REF_ARG(T) x)
	{
		using traits = hlsl::vector_traits<T>;
		array_get<T, typename traits::scalar_type> getter;
		array_set<T, typename traits::scalar_type> setter;

		T fracPartOut;
		T intPartOut;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
		{
			using component_return_t = ModfOutput<typename vector_traits<T>::scalar_type>;
			component_return_t result = modfStruct_helper<typename traits::scalar_type>::__call(getter(x, i));

			setter(fracPartOut, i, result.fractionalPart);
			setter(intPartOut, i, result.wholeNumberPart);
		}

		return_t output;
		output.fractionalPart = fracPartOut;
		output.wholeNumberPart = intPartOut;

		return output;
	}
};

template<typename T>
NBL_PARTIAL_REQ_TOP(VECTOR_SPECIALIZATION_CONCEPT)
struct frexpStruct_helper<T NBL_PARTIAL_REQ_BOT(VECTOR_SPECIALIZATION_CONCEPT) >
{
	using return_t = FrexpOutput<T>;
	static return_t __call(NBL_CONST_REF_ARG(T) x)
	{
		using traits = hlsl::vector_traits<T>;
		array_get<T, typename traits::scalar_type> getter;
		array_set<T, typename traits::scalar_type> significandSetter;
		array_set<T, typename traits::scalar_type> exponentSetter;

		T significandOut;
		T exponentOut;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
		{
			using component_return_t = FrexpOutput<typename vector_traits<T>::scalar_type>;
			component_return_t result = frexpStruct_helper<typename traits::scalar_type>::__call(getter(x, i));

			significandSetter(significandOut, i, result.significand);
			exponentSetter(exponentOut, i, result.exponent);
		}

		return_t output;
		output.significand = significandOut;
		output.exponent = exponentOut;

		return output;
	}
};

template<typename T>
NBL_PARTIAL_REQ_TOP(VECTOR_SPECIALIZATION_CONCEPT)
struct atan2_helper<T NBL_PARTIAL_REQ_BOT(VECTOR_SPECIALIZATION_CONCEPT) >
{
	using return_t = T;
	static return_t __call(NBL_CONST_REF_ARG(T) y, NBL_CONST_REF_ARG(T) x)
	{
		using traits = hlsl::vector_traits<T>;
		array_get<T, typename traits::scalar_type> getter;
		array_set<T, typename traits::scalar_type> setter;

		return_t output;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			setter(output, i, atan2_helper<typename traits::scalar_type>::__call(getter(y, i), getter(x, i)));

		return output;
	}
};

#undef VECTOR_SPECIALIZATION_CONCEPT

}
}
}

#endif