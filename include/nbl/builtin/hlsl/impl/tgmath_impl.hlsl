#ifndef _NBL_BUILTIN_HLSL_TGMATH_IMPL_INCLUDED_
#define _NBL_BUILTIN_HLSL_TGMATH_IMPL_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/basic.h>
#include <nbl/builtin/hlsl/concepts.hlsl>
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

template<typename UnsignedInteger NBL_FUNC_REQUIRES(hlsl::is_integral_v<UnsignedInteger>&& hlsl::is_unsigned_v<UnsignedInteger>)
inline bool isnan_uint_impl(UnsignedInteger val)
{
	using AsFloat = typename float_of_size<sizeof(UnsignedInteger)>::type;
	return bool((ieee754::extractBiasedExponent<UnsignedInteger>(val) == ieee754::traits<AsFloat>::specialValueExp) && (val & ieee754::traits<AsFloat>::mantissaMask));
}

template<typename UnsignedInteger NBL_FUNC_REQUIRES(hlsl::is_integral_v<UnsignedInteger>&& hlsl::is_unsigned_v<UnsignedInteger>)
inline bool isinf_uint_impl(UnsignedInteger val)
{
	using AsFloat = typename float_of_size<sizeof(UnsignedInteger)>::type;
	return (val & (~ieee754::traits<AsFloat>::signMask)) == ieee754::traits<AsFloat>::inf;
}

template<typename V NBL_STRUCT_CONSTRAINABLE>
struct floor_helper;

template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(hlsl::is_floating_point_v<FloatingPoint>&& hlsl::is_scalar_v<FloatingPoint>)
struct floor_helper<FloatingPoint NBL_PARTIAL_REQ_BOT(hlsl::is_vector_v<FloatingPoint>) >
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

template<typename Vector>
NBL_PARTIAL_REQ_TOP(hlsl::is_vector_v<Vector>)
struct floor_helper<Vector NBL_PARTIAL_REQ_BOT(hlsl::is_vector_v<Vector>) >
{
	static Vector __call(NBL_CONST_REF_ARG(Vector) vec)
	{
#ifdef __HLSL_VERSION
		return spirv::floor(vec);
#else
		Vector output;
		using traits = hlsl::vector_traits<Vector>;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			output[i] = floor(vec[i]);
		return output;

		// TODO: cherry-pick array_getters
		/*array_get<Vector> getter;
		array_set<Vector> setter;
		Vector output;
		using traits = hlsl::vector_traits<Vector>;
		for (uint32_t i = 0; i < traits::Dimension; i++)
			setter(output, floor_helper<traits::scalar_type>::__call(getter(vec, i)), i);
		return output;*/
#endif
	}
};

}
}
}

#endif