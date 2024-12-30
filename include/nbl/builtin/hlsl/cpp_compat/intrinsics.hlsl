#ifndef _NBL_BUILTIN_HLSL_CPP_COMPAT_INTRINSICS_INCLUDED_
#define _NBL_BUILTIN_HLSL_CPP_COMPAT_INTRINSICS_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/type_traits.hlsl>
#include <nbl/builtin/hlsl/vector_utils/vector_traits.hlsl>
#include <nbl/builtin/hlsl/array_accessors.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/impl/intrinsics_impl.hlsl>
#include <nbl/builtin/hlsl/matrix_utils/matrix_traits.hlsl>
#include <nbl/builtin/hlsl/ieee754.hlsl>

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
inline cpp_compat_intrinsics_impl::bitcount_output_t<Integer> bitCount(NBL_CONST_REF_ARG(Integer) val)
{
	return cpp_compat_intrinsics_impl::bitCount_helper<Integer>::__call(val);
}

template<typename FloatingPointVector>
FloatingPointVector cross(NBL_CONST_REF_ARG(FloatingPointVector) lhs, NBL_CONST_REF_ARG(FloatingPointVector) rhs)
{
	return cpp_compat_intrinsics_impl::cross_helper<FloatingPointVector>::__call(lhs, rhs);
}

template<typename Scalar>
enable_if_t<!is_vector_v<Scalar>, Scalar> clamp(NBL_CONST_REF_ARG(Scalar) val, NBL_CONST_REF_ARG(Scalar) min, NBL_CONST_REF_ARG(Scalar) max)
{
	return cpp_compat_intrinsics_impl::clamp_helper<Scalar>::__call(val, min, max);
}

// TODO: is_vector_v<T> will be false for custom vector types, fix
template<typename Vector>
enable_if_t<is_vector_v<Vector>, Vector> clamp(NBL_CONST_REF_ARG(Vector) val, NBL_CONST_REF_ARG(typename vector_traits<Vector>::scalar_type) min, NBL_CONST_REF_ARG(typename vector_traits<Vector>::scalar_type) max)
{
	return cpp_compat_intrinsics_impl::clamp_helper<Vector>::__call(val, min, max);
}

template<typename Vector>
typename vector_traits<Vector>::scalar_type length(NBL_CONST_REF_ARG(Vector) vec)
{
	return cpp_compat_intrinsics_impl::length_helper<Vector>::__call(vec);
}

template<typename Vector>
Vector normalize(NBL_CONST_REF_ARG(Vector) vec)
{
	return cpp_compat_intrinsics_impl::normalize_helper<Vector>::__call(vec);
}

template<typename T>
typename vector_traits<T>::scalar_type dot(NBL_CONST_REF_ARG(T) lhs, NBL_CONST_REF_ARG(T) rhs)
{
	return cpp_compat_intrinsics_impl::dot_helper<T>::__call(lhs, rhs);
}

// determinant not defined cause its implemented via hidden friend
// https://stackoverflow.com/questions/67459950/why-is-a-friend-function-not-treated-as-a-member-of-a-namespace-of-a-class-it-wa
template<typename Matrix>
inline typename matrix_traits<Matrix>::scalar_type determinant(NBL_CONST_REF_ARG(Matrix) mat)
{
	return cpp_compat_intrinsics_impl::determinant_helper<Matrix>::__call(mat);
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

// inverse not defined cause its implemented via hidden friend
template<typename Matrix>
inline Matrix inverse(NBL_CONST_REF_ARG(Matrix) mat)
{
	return cpp_compat_intrinsics_impl::inverse_helper<Matrix>::__call(mat);
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
	return cpp_compat_intrinsics_impl::min_helper<T>::__call(a, b);
}

template<typename T>
inline T max(NBL_CONST_REF_ARG(T) a, NBL_CONST_REF_ARG(T) b)
{
	return cpp_compat_intrinsics_impl::max_helper<T>::__call(a, b);
}

template<typename FloatingPoint>
inline FloatingPoint rsqrt(FloatingPoint x)
{
	return cpp_compat_intrinsics_impl::rsqrt_helper<FloatingPoint>::__call(x);
}

template<typename Integer>
inline Integer bitReverse(Integer val)
{
	return cpp_compat_intrinsics_impl::bitReverse_helper<Integer>::__call(val);
}

template<typename Vector>
inline bool all(Vector vec)
{
	return cpp_compat_intrinsics_impl::all_helper<Vector>::__call(vec);
}

template<typename Vector>
inline bool any(Vector vec)
{
	return cpp_compat_intrinsics_impl::any_helper<Vector>::__call(vec);
}

}
}

#endif