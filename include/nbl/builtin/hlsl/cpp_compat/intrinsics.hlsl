#ifndef _NBL_BUILTIN_HLSL_CPP_COMPAT_INTRINSICS_INCLUDED_
#define _NBL_BUILTIN_HLSL_CPP_COMPAT_INTRINSICS_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/type_traits.hlsl>
#include <nbl/builtin/hlsl/vector_utils/vector_traits.hlsl>
#include <nbl/builtin/hlsl/array_accessors.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/impl/intrinsics_impl.hlsl>
#include <nbl/builtin/hlsl/matrix_utils/matrix_traits.hlsl>
#include <nbl/builtin/hlsl/ieee754.hlsl>
#include <nbl/builtin/hlsl/concepts/core.hlsl>
#include <nbl/builtin/hlsl/concepts/vector.hlsl>
#include <nbl/builtin/hlsl/concepts/matrix.hlsl>

#ifndef __HLSL_VERSION
#include <algorithm>
#include <cmath>
#include "nbl/core/util/bitflag.h"
#endif

namespace nbl
{
namespace hlsl
{

#ifdef __HLSL_VERSION
template<typename Integer NBL_FUNC_REQUIRES(concepts::Integral<Integer>)
#else
template<typename Integer NBL_FUNC_REQUIRES(concepts::Integral<Integer> || std::is_enum_v<Integer>)
#endif
inline cpp_compat_intrinsics_impl::bitcount_output_t<Integer> bitCount(NBL_CONST_REF_ARG(Integer) val)
{
	return cpp_compat_intrinsics_impl::bitCount_helper<Integer>::__call(val);
}

template<typename FloatingPointVectorial NBL_FUNC_REQUIRES(concepts::FloatingPointLikeVectorial<FloatingPointVectorial>)
FloatingPointVectorial cross(NBL_CONST_REF_ARG(FloatingPointVectorial) lhs, NBL_CONST_REF_ARG(FloatingPointVectorial) rhs)
{
	return cpp_compat_intrinsics_impl::cross_helper<FloatingPointVectorial>::__call(lhs, rhs);
}

template<typename Scalar NBL_FUNC_REQUIRES(concepts::Scalar<Scalar>)
Scalar clamp(NBL_CONST_REF_ARG(Scalar) val, NBL_CONST_REF_ARG(Scalar) min, NBL_CONST_REF_ARG(Scalar) max)
{
	return cpp_compat_intrinsics_impl::clamp_helper<Scalar>::__call(val, min, max);
}

template<typename Vectorial NBL_FUNC_REQUIRES(concepts::Vectorial<Vectorial>)
Vectorial clamp(NBL_CONST_REF_ARG(Vectorial) val, NBL_CONST_REF_ARG(typename vector_traits<Vectorial>::scalar_type) min, NBL_CONST_REF_ARG(typename vector_traits<Vectorial>::scalar_type) max)
{
	return cpp_compat_intrinsics_impl::clamp_helper<Vectorial>::__call(val, min, max);
}

template<typename FloatingPointVectorial NBL_FUNC_REQUIRES(concepts::FloatingPointLikeVectorial<FloatingPointVectorial>)
typename vector_traits<FloatingPointVectorial>::scalar_type length(NBL_CONST_REF_ARG(FloatingPointVectorial) vec)
{
	return cpp_compat_intrinsics_impl::length_helper<FloatingPointVectorial>::__call(vec);
}

template<typename FloatingPointVectorial NBL_FUNC_REQUIRES(concepts::FloatingPointLikeVectorial<FloatingPointVectorial>)
FloatingPointVectorial normalize(NBL_CONST_REF_ARG(FloatingPointVectorial) vec)
{
	return cpp_compat_intrinsics_impl::normalize_helper<FloatingPointVectorial>::__call(vec);
}

template<typename Vectorial NBL_FUNC_REQUIRES(concepts::Vectorial<Vectorial>)
typename vector_traits<Vectorial>::scalar_type dot(NBL_CONST_REF_ARG(Vectorial) lhs, NBL_CONST_REF_ARG(Vectorial) rhs)
{
	return cpp_compat_intrinsics_impl::dot_helper<Vectorial>::__call(lhs, rhs);
}

// determinant not defined cause its implemented via hidden friend
// https://stackoverflow.com/questions/67459950/why-is-a-friend-function-not-treated-as-a-member-of-a-namespace-of-a-class-it-wa
template<typename Matrix NBL_FUNC_REQUIRES(concepts::Matricial<Matrix> && matrix_traits<Matrix>::Square)
inline typename matrix_traits<Matrix>::scalar_type determinant(NBL_CONST_REF_ARG(Matrix) mat)
{
	return cpp_compat_intrinsics_impl::determinant_helper<Matrix>::__call(mat);
}

#ifdef __HLSL_VERSION
template<typename Integer>
inline typename cpp_compat_intrinsics_impl::find_lsb_return_type<Integer>::type findLSB(NBL_CONST_REF_ARG(Integer) val)
{
	return cpp_compat_intrinsics_impl::find_lsb_helper<Integer>::__call(val);
}
#else
// TODO: no concepts because then it wouldn't work for core::bitflag, find solution
template<typename Integer>
inline typename cpp_compat_intrinsics_impl::find_lsb_return_type<Integer>::type findLSB(NBL_CONST_REF_ARG(Integer) val)
{
	return cpp_compat_intrinsics_impl::find_lsb_helper<Integer>::__call(val);
}
#endif

#ifdef __HLSL_VERSION
template<typename Integer NBL_FUNC_REQUIRES(concepts::Integral<Integer>)
inline typename cpp_compat_intrinsics_impl::find_msb_return_type<Integer>::type findMSB(NBL_CONST_REF_ARG(Integer) val)
{
	return cpp_compat_intrinsics_impl::find_msb_helper<Integer>::__call(val);
}
#else
// TODO: no concepts because then it wouldn't work for core::bitflag, find solution
template<typename Integer>
inline typename cpp_compat_intrinsics_impl::find_msb_return_type<Integer>::type findMSB(NBL_CONST_REF_ARG(Integer) val)
{
	return cpp_compat_intrinsics_impl::find_msb_helper<Integer>::__call(val);
}
#endif
// inverse not defined cause its implemented via hidden friend
template<typename Matrix NBL_FUNC_REQUIRES(concepts::Matricial<Matrix> && matrix_traits<Matrix>::Square)
inline Matrix inverse(NBL_CONST_REF_ARG(Matrix) mat)
{
	return cpp_compat_intrinsics_impl::inverse_helper<Matrix>::__call(mat);
}

// transpose not defined cause its implemented via hidden friend
template<typename Matrix NBL_FUNC_REQUIRES(concepts::Matricial<Matrix>)
inline typename matrix_traits<Matrix>::transposed_type transpose(NBL_CONST_REF_ARG(Matrix) m)
{
	return cpp_compat_intrinsics_impl::transpose_helper<Matrix>::__call(m);
}

template<typename LhsT, typename RhsT NBL_FUNC_REQUIRES(concepts::Matricial<LhsT> && (concepts::Matricial<RhsT> || concepts::Vectorial<RhsT>))
mul_output_t<LhsT, RhsT> mul(LhsT lhs, RhsT rhs)
{
	return cpp_compat_intrinsics_impl::mul_helper<LhsT, RhsT>::__call(lhs, rhs);
}

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointLikeScalar<T> || concepts::Scalar<T> || concepts::Vectorial<T>)
inline T min(NBL_CONST_REF_ARG(T) a, NBL_CONST_REF_ARG(T) b)
{
	return cpp_compat_intrinsics_impl::min_helper<T>::__call(a, b);
}

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointLikeScalar<T> || concepts::Scalar<T> || concepts::Vectorial<T>)
inline T max(NBL_CONST_REF_ARG(T) a, NBL_CONST_REF_ARG(T) b)
{
	return cpp_compat_intrinsics_impl::max_helper<T>::__call(a, b);
}

template<typename FloatingPoint NBL_FUNC_REQUIRES(concepts::FloatingPointLikeScalar<FloatingPoint> || concepts::FloatingPointLikeVectorial<FloatingPoint>)
inline FloatingPoint rsqrt(FloatingPoint x)
{
	return cpp_compat_intrinsics_impl::rsqrt_helper<FloatingPoint>::__call(x);
}

template<typename Integer NBL_FUNC_REQUIRES(concepts::Integral<Integer>)
inline Integer bitReverse(Integer val)
{
	return cpp_compat_intrinsics_impl::bitReverse_helper<Integer>::__call(val);
}

template<typename Vector NBL_FUNC_REQUIRES(concepts::Vectorial<Vector>)
inline bool all(Vector vec)
{
	return cpp_compat_intrinsics_impl::all_helper<Vector>::__call(vec);
}

template<typename Vector NBL_FUNC_REQUIRES(concepts::Vectorial<Vector>)
inline bool any(Vector vec)
{
	return cpp_compat_intrinsics_impl::any_helper<Vector>::__call(vec);
}

}
}

#endif