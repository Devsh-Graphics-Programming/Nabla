#ifndef _NBL_BUILTIN_HLSL_CPP_COMPAT_INTRINSICS_INCLUDED_
#define _NBL_BUILTIN_HLSL_CPP_COMPAT_INTRINSICS_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/type_traits.hlsl>
#include <nbl/builtin/hlsl/vector_utils/vector_traits.hlsl>
#include <nbl/builtin/hlsl/array_accessors.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/impl/intrinsics_impl.hlsl>
#include <nbl/builtin/hlsl/matrix_utils/matrix_traits.hlsl>

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
inline int bitCount(NBL_CONST_REF_ARG(Integer) val)
{
#ifdef __HLSL_VERSION
	if (sizeof(Integer) == 8u)
	{
		uint32_t lowBits = uint32_t(val);
		uint32_t highBits = uint32_t(uint64_t(val) >> 32u);

		return countbits(lowBits) + countbits(highBits);
	}

	return countbits(val);

#else
	return glm::bitCount(val);
#endif
}

template<typename T>
vector<T, 3> cross(NBL_CONST_REF_ARG(vector<T, 3>) lhs, NBL_CONST_REF_ARG(vector<T, 3>) rhs)
{
#ifdef __HLSL_VERSION
	return spirv::cross(lhs, rhs);
#else
	return glm::cross(lhs, rhs);
#endif
}

template<typename T>
T clamp(NBL_CONST_REF_ARG(T) val, NBL_CONST_REF_ARG(T) min, NBL_CONST_REF_ARG(T) max)
{
#ifdef __HLSL_VERSION
	return clamp(val, min, max);
#else
	return glm::clamp(val, min, max);
#endif
}

template<typename T>
typename vector_traits<T>::scalar_type dot(NBL_CONST_REF_ARG(T) lhs, NBL_CONST_REF_ARG(T) rhs)
{
	return cpp_compat_intrinsics_impl::dot_helper<T>::__call(lhs, rhs);
}

// TODO: for clearer error messages, use concepts to ensure that input type is a square matrix
// determinant not defined cause its implemented via hidden friend
// https://stackoverflow.com/questions/67459950/why-is-a-friend-function-not-treated-as-a-member-of-a-namespace-of-a-class-it-wa
template<typename T, uint16_t N>
inline T determinant(NBL_CONST_REF_ARG(matrix<T, N, N>) m)
{
#ifdef __HLSL_VERSION
	spirv::determinant(m);
#else
	return glm::determinant(reinterpret_cast<typename matrix<T, N, N>::Base const&>(m));
#endif
}

template<typename Integer>
inline typename cpp_compat_intrinsics_impl::find_lsb_return_type<Integer>::type findLSB(NBL_CONST_REF_ARG(Integer) val)
{
	return cpp_compat_intrinsics_impl::find_lsb_helper<Integer>::__call(val);
}

template<typename Integer>
inline typename cpp_compat_intrinsics_impl::find_msb_return_type<Integer>::type findMSB(NBL_CONST_REF_ARG(Integer) val)
{
	return cpp_compat_intrinsics_impl::find_msb_helper<Integer>::__call(val);
}

// TODO: for clearer error messages, use concepts to ensure that input type is a square matrix
// inverse not defined cause its implemented via hidden friend
template<typename T, uint16_t N>
inline matrix<T, N, N> inverse(NBL_CONST_REF_ARG(matrix<T, N, N>) m)
{
#ifdef __HLSL_VERSION
	return spirv::matrixInverse(m);
#else
	return reinterpret_cast<matrix<T, N, N>&>(glm::inverse(reinterpret_cast<typename matrix<T, N, N>::Base const&>(m)));
#endif
}

// transpose not defined cause its implemented via hidden friend
template<typename Matrix>
inline typename matrix_traits<Matrix>::transposed_type transpose(NBL_CONST_REF_ARG(Matrix) m)
{
	return cpp_compat_intrinsics_impl::transpose_helper<Matrix>::__call(m);
}

// TODO: concepts, to ensure that MatT is a matrix and VecT is a vector type
template<typename LhsT, typename RhsT>
mul_output_t<LhsT, RhsT> mul(LhsT mat, RhsT vec)
{
	return cpp_compat_intrinsics_impl::mul_helper<LhsT, RhsT>::__call(mat, vec);
}

template<typename T>
inline T min(NBL_CONST_REF_ARG(T) a, NBL_CONST_REF_ARG(T) b)
{
#ifdef __HLSL_VERSION
	min(a, b);
#else
	return glm::min(a, b);
#endif
}

template<typename T>
inline T max(NBL_CONST_REF_ARG(T) a, NBL_CONST_REF_ARG(T) b)
{
#ifdef __HLSL_VERSION
	max(a, b);
#else
	return glm::max(a, b);
#endif
}

template<typename FloatingPoint>
inline FloatingPoint rsqrt(FloatingPoint x)
{
	// TODO: https://stackoverflow.com/a/62239778
#ifdef __HLSL_VERSION
	return spirv::inverseSqrt(x);
#else
	return 1.0f / std::sqrt(x);
#endif
}

template<typename Integer>
inline Integer bitReverse(Integer val)
{
	return cpp_compat_intrinsics_impl::bitReverse_helper<Integer>::__call(val);
}

template<typename FloatingPoint>
inline FloatingPoint negate(FloatingPoint val)
{
	return cpp_compat_intrinsics_impl::negate_helper<FloatingPoint>::__call(val);
}

}
}

#endif