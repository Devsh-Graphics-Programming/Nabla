#ifndef _NBL_BUILTIN_HLSL_CPP_COMPAT_INTRINSICS_INCLUDED_
#define _NBL_BUILTIN_HLSL_CPP_COMPAT_INTRINSICS_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/type_traits.hlsl>

#ifndef __HLSL_VERSION
#include <algorithm>
#include <cmath>
#include "nbl/core/util/bitflag.h"
#endif

namespace nbl
{
namespace hlsl
{

template<typename Integer>
int bitCount(Integer val)
{
#ifdef __HLSL_VERSION

#else
	return glm::bitCount(val);
#endif
}

template<typename T>
vector<T, 3> cross(NBL_CONST_REF_ARG(vector<T, 3>) lhs, NBL_CONST_REF_ARG(vector<T, 3>) rhs)
{
#ifdef __HLSL_VERSION

#else
	return glm::cross(val);
#endif
}

template<typename T>
int bitCount(T val)
{
#ifdef __HLSL_VERSION

#else
	return glm::clamp(val);
#endif
}

namespace dot_product_impl
{
	template<typename T>
	struct dot_helper
	{
		using scalar_type = typename vector_traits<T>::scalar_type;

		static inline scalar_type dot(NBL_CONST_REF_ARG(T) lhs, NBL_CONST_REF_ARG(T) rhs)
		{
			static array_get<T, scalar_type> getter;
			scalar_type retval = getter(lhs, 0) * getter(rhs, 0);

			static const uint32_t ArrayDim = sizeof(T) / sizeof(scalar_type);
			for (uint32_t i = 1; i < ArrayDim; ++i)
				retval = retval + getter(lhs, i) * getter(rhs, i);

			return retval;
		}
	};

#define DEFINE_BUILTIN_VECTOR_SPECIALIZATION(FLOAT_TYPE, RETURN_VALUE)\
template<uint32_t N>\
struct dot_helper<vector<FLOAT_TYPE, N> >\
{\
	using VectorType = vector<FLOAT_TYPE, N>;\
	using ScalarType = typename vector_traits<VectorType>::scalar_type;\
\
	static inline ScalarType dot(NBL_CONST_REF_ARG(VectorType) lhs, NBL_CONST_REF_ARG(VectorType) rhs)\
	{\
		return RETURN_VALUE;\
	}\
};\

#ifdef __HLSL_VERSION
#define BUILTIN_VECTOR_SPECIALIZATION_RET_VAL dot(lhs, rhs)
#else
#define BUILTIN_VECTOR_SPECIALIZATION_RET_VAL glm::dot(lhs, rhs)
#endif

	DEFINE_BUILTIN_VECTOR_SPECIALIZATION(float16_t, BUILTIN_VECTOR_SPECIALIZATION_RET_VAL)
		DEFINE_BUILTIN_VECTOR_SPECIALIZATION(float32_t, BUILTIN_VECTOR_SPECIALIZATION_RET_VAL)
		DEFINE_BUILTIN_VECTOR_SPECIALIZATION(float64_t, BUILTIN_VECTOR_SPECIALIZATION_RET_VAL)

#undef BUILTIN_VECTOR_SPECIALIZATION_RET_VAL
#undef DEFINE_BUILTIN_VECTOR_SPECIALIZATION

}

template<typename T>
typename vector_traits<T>::scalar_type dot(NBL_CONST_REF_ARG(T) lhs, NBL_CONST_REF_ARG(T) rhs)
{
	return dot_product_impl::dot_helper<T>::dot(lhs, rhs);
}

// determinant not defined cause its implemented via hidden friend
// https://stackoverflow.com/questions/67459950/why-is-a-friend-function-not-treated-as-a-member-of-a-namespace-of-a-class-it-wa
template<typename T, uint16_t N, uint16_t M>
inline T determinant(const matrix<T, N, M>& m)
{
#ifdef __HLSL_VERSION

#else
	return glm::determinant(reinterpret_cast<typename matrix<T, N, M>::Base const&>(m));
#endif
}

}
}

#endif
