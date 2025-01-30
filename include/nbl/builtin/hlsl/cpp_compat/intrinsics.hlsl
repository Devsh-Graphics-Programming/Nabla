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

template<typename T>
inline typename cpp_compat_intrinsics_impl::bitCount_helper<T>::return_t bitCount(NBL_CONST_REF_ARG(T) val)
{
	return cpp_compat_intrinsics_impl::bitCount_helper<T>::__call(val);
}

template<typename T>
T cross(NBL_CONST_REF_ARG(T) lhs, NBL_CONST_REF_ARG(T) rhs)
{
	return cpp_compat_intrinsics_impl::cross_helper<T>::__call(lhs, rhs);
}

template<typename T>
typename cpp_compat_intrinsics_impl::clamp_helper<T>::return_t clamp(NBL_CONST_REF_ARG(T) val, NBL_CONST_REF_ARG(T) _min, NBL_CONST_REF_ARG(T) _max)
{
	return cpp_compat_intrinsics_impl::clamp_helper<T>::__call(val, _min, _max);
}

template<typename FloatingPointVectorial>
typename vector_traits<FloatingPointVectorial>::scalar_type length(NBL_CONST_REF_ARG(FloatingPointVectorial) vec)
{
	return cpp_compat_intrinsics_impl::length_helper<FloatingPointVectorial>::__call(vec);
}

template<typename FloatingPointVectorial>
FloatingPointVectorial normalize(NBL_CONST_REF_ARG(FloatingPointVectorial) vec)
{
	return cpp_compat_intrinsics_impl::normalize_helper<FloatingPointVectorial>::__call(vec);
}

template<typename Vectorial>
typename vector_traits<Vectorial>::scalar_type dot(NBL_CONST_REF_ARG(Vectorial) lhs, NBL_CONST_REF_ARG(Vectorial) rhs)
{
	return cpp_compat_intrinsics_impl::dot_helper<Vectorial>::__call(lhs, rhs);
}

// determinant not defined cause its implemented via hidden friend
// https://stackoverflow.com/questions/67459950/why-is-a-friend-function-not-treated-as-a-member-of-a-namespace-of-a-class-it-wa
template<typename Matrix NBL_FUNC_REQUIRES(concepts::Matricial<Matrix>)
inline typename matrix_traits<Matrix>::scalar_type determinant(NBL_CONST_REF_ARG(Matrix) mat)
{
	return cpp_compat_intrinsics_impl::determinant_helper<Matrix>::__call(mat);
}

template<typename T>
inline typename cpp_compat_intrinsics_impl::find_lsb_helper<T>::return_t findLSB(NBL_CONST_REF_ARG(T) val)
{
	return cpp_compat_intrinsics_impl::find_lsb_helper<T>::__call(val);
}

template<typename T>
inline typename cpp_compat_intrinsics_impl::find_msb_helper<T>::return_t findMSB(NBL_CONST_REF_ARG(T) val)
{
	return cpp_compat_intrinsics_impl::find_msb_helper<T>::__call(val);
}

// inverse not defined cause its implemented via hidden friend
template<typename Matrix NBL_FUNC_REQUIRES(concepts::Matricial<Matrix>)
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

template<typename LhsT, typename RhsT>
inline typename cpp_compat_intrinsics_impl::mul_helper<LhsT, RhsT>::return_t mul(LhsT lhs, RhsT rhs)
{
	return cpp_compat_intrinsics_impl::mul_helper<LhsT, RhsT>::__call(lhs, rhs);
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

template<typename T, uint16_t Bits>
/**
* @brief Takes the binary representation of `value` as a string of `Bits` bits and returns a value of the same type resulting from reversing the string
*
* @tparam T Type of the value to operate on.
* @tparam Bits The length of the string of bits used to represent `value`.
*
* @param [in] value The value to bitreverse.
*/
inline T bitReverseAs(T val)
{
	return cpp_compat_intrinsics_impl::bitReverseAs_helper<T, Bits>::__call(val);
}

template<typename T NBL_FUNC_REQUIRES(is_unsigned_v<T>)
/**
* @brief Takes the binary representation of `value` and returns a value of the same type resulting from reversing the string of bits as if it was `bits` long.
* Keep in mind `bits` cannot exceed `8 * sizeof(T)`.
*
* @tparam T type of the value to operate on.
*
* @param [in] value The value to bitreverse.
* @param [in] bits The length of the string of bits used to represent `value`.
*/
T bitReverseAs(T val, uint16_t bits)
{
	return cpp_compat_intrinsics_impl::bitReverseAs_helper<T, 0>::__call(val, bits);
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

/**
* @brief Returns x - floor(x).
*
* @tparam T type of the value to operate on.
*
* @param [in] val The value to operate on.
*/
template<typename T>
inline T frac(NBL_CONST_REF_ARG(T) val)
{
	return cpp_compat_intrinsics_impl::frac_helper<T>::__call(val);
}

template<typename T, typename U>
inline T mix(NBL_CONST_REF_ARG(T) x, NBL_CONST_REF_ARG(T) y, NBL_CONST_REF_ARG(U) a)
{
	return cpp_compat_intrinsics_impl::mix_helper<T, U>::__call(x, y, a);
}

}
}

#endif