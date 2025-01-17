#ifndef _NBL_BUILTIN_HLSL_CPP_COMPAT_IMPL_INTRINSICS_IMPL_INCLUDED_
#define _NBL_BUILTIN_HLSL_CPP_COMPAT_IMPL_INTRINSICS_IMPL_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/basic.h>
#include <nbl/builtin/hlsl/matrix_utils/matrix_traits.hlsl>
#include <nbl/builtin/hlsl/matrix_utils/mul_output_t.hlsl>
#include <nbl/builtin/hlsl/concepts.hlsl>
#include <nbl/builtin/hlsl/spirv_intrinsics/core.hlsl>
#include <nbl/builtin/hlsl/spirv_intrinsics/glsl.std.450.hlsl>
#include <nbl/builtin/hlsl/ieee754.hlsl>
#include <nbl/builtin/hlsl/concepts/core.hlsl>
#include <nbl/builtin/hlsl/concepts/vector.hlsl>
#include <nbl/builtin/hlsl/concepts/matrix.hlsl>
#ifndef __HLSL_VERSION
#include <bitset>
#endif

namespace nbl
{
namespace hlsl
{
namespace cpp_compat_intrinsics_impl
{

// TODO: remove
template<typename Integer>
struct find_msb_return_type
{
	using type = int32_t;
};
template<typename Integer, int N>
struct find_msb_return_type<vector<Integer, N> >
{
	using type = vector<int32_t, N>;
};
template<typename Integer>
using find_lsb_return_type = find_msb_return_type<Integer>;

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct bitcount_output;

template<typename Integer>
NBL_PARTIAL_REQ_TOP(is_integral_v<Integer>)
struct bitcount_output<Integer NBL_PARTIAL_REQ_BOT(is_integral_v<Integer>) >
{
	using type = int32_t;
};
template<typename IntegerVector>
NBL_PARTIAL_REQ_TOP(concepts::IntVector<IntegerVector>)
struct bitcount_output<IntegerVector NBL_PARTIAL_REQ_BOT(concepts::IntVector<IntegerVector>) >
{
	using type = vector<int32_t, hlsl::vector_traits<IntegerVector>::Dimension>;
};
#ifndef __HLSL_VERSION
template<typename EnumT>
requires std::is_enum_v<EnumT>
struct bitcount_output<EnumT NBL_PARTIAL_REQ_BOT(hlsl::is_enum_v<EnumT>) >
{
	using type = int32_t;
};
#endif

template<typename T>
using bitcount_output_t = typename bitcount_output<T>::type;

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct dot_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct cross_helper;
template<typename T, typename U NBL_STRUCT_CONSTRAINABLE>
struct clamp_helper;
template<typename Integer NBL_STRUCT_CONSTRAINABLE>
struct find_msb_helper;
template<typename Integer NBL_STRUCT_CONSTRAINABLE>
struct find_lsb_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct bitReverse_helper;
template<typename Matrix NBL_STRUCT_CONSTRAINABLE>
struct transpose_helper;
template<typename Vector NBL_STRUCT_CONSTRAINABLE>
struct length_helper;
template<typename Vector NBL_STRUCT_CONSTRAINABLE>
struct normalize_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct max_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct min_helper;
template<typename Integer NBL_STRUCT_CONSTRAINABLE>
struct bitCount_helper;
template<typename LhsT, typename RhsT NBL_STRUCT_CONSTRAINABLE>
struct mul_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct determinant_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct inverse_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct rsqrt_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct all_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct any_helper;

#ifdef __HLSL_VERSION // HLSL only specializations
template<typename T, typename U> NBL_PARTIAL_REQ_TOP(always_true<decltype(spirv::fClamp<T>(experimental::declval<T>(), experimental::declval<U>(), experimental::declval<U>()))>)
struct clamp_helper<T, U NBL_PARTIAL_REQ_BOT(always_true<decltype(spirv::fClamp<T>(experimental::declval<T>(), experimental::declval<U>(), experimental::declval<U>()))>) >
{
	using return_t = T;
	static return_t __call(NBL_CONST_REF_ARG(T) val, NBL_CONST_REF_ARG(U) min, NBL_CONST_REF_ARG(U) max)
	{
		return spirv::fClamp(val, min, max);
	}
};
template<typename T, typename U> NBL_PARTIAL_REQ_TOP(always_true<decltype(spirv::uClamp<T>(experimental::declval<T>(), experimental::declval<U>(), experimental::declval<U>()))>)
struct clamp_helper<T, U NBL_PARTIAL_REQ_BOT(always_true<decltype(spirv::uClamp<T>(experimental::declval<T>(), experimental::declval<U>(), experimental::declval<U>()))>) >
{
	using return_t = T;
	static return_t __call(NBL_CONST_REF_ARG(T) val, NBL_CONST_REF_ARG(U) min, NBL_CONST_REF_ARG(U) max)
	{
		return spirv::uClamp(val, min, max);
	}
};
template<typename T, typename U> NBL_PARTIAL_REQ_TOP(always_true<decltype(spirv::sClamp<T>(experimental::declval<T>(), experimental::declval<U>(), experimental::declval<U>()))>)
struct clamp_helper<T, U NBL_PARTIAL_REQ_BOT(always_true<decltype(spirv::sClamp<T>(experimental::declval<T>(), experimental::declval<U>(), experimental::declval<U>()))>) >
{
	using return_t = T;
	static return_t __call(NBL_CONST_REF_ARG(T) val, NBL_CONST_REF_ARG(U) min, NBL_CONST_REF_ARG(U) max)
	{
		return spirv::sClamp(val, min, max);
	}
};
template<typename T> NBL_PARTIAL_REQ_TOP(always_true<decltype(spirv::findUMsb<T>(experimental::declval<T>()))>)
struct find_msb_helper<T NBL_PARTIAL_REQ_BOT(always_true<decltype(spirv::findUMsb<T>(experimental::declval<T>()))>) >
{
	using return_t = conditional_t<is_vector_v<T>, vector<int32_t, vector_traits<T>::Dimension>, int32_t>;
	static T __call(NBL_CONST_REF_ARG(T) val)
	{
		return spirv::findUMsb(val);
	}
};
template<typename T> NBL_PARTIAL_REQ_TOP(always_true<decltype(spirv::findSMsb<T>(experimental::declval<T>()))>)
struct find_msb_helper<T NBL_PARTIAL_REQ_BOT(always_true<decltype(spirv::findSMsb<T>(experimental::declval<T>()))>) >
{
	using return_t = conditional_t<is_vector_v<T>, vector<int32_t, vector_traits<T>::Dimension>, int32_t>;
	static T __call(NBL_CONST_REF_ARG(T) val)
	{
		return spirv::findSMsb(val);
	}
};
template<typename UInt64> NBL_PARTIAL_REQ_TOP(is_same_v<UInt64, uint64_t>)
struct find_msb_helper<UInt64 NBL_PARTIAL_REQ_BOT(is_same_v<UInt64, uint64_t>) >
{
	using return_t = int32_t;
	static return_t __call(NBL_CONST_REF_ARG(UInt64) val)
	{
		const uint32_t highBits = uint32_t(val >> 32);
		const int32_t highMsb = find_msb_helper<uint32_t>::__call(highBits);

		if (highMsb == -1)
		{
			const uint32_t lowBits = uint32_t(val);
			const int32_t lowMsb = find_msb_helper<uint32_t>::__call(lowBits);
			if (lowMsb == -1)
				return -1;

			return lowMsb;
		}

		return highMsb + 32;
	}
};
template<typename T> NBL_PARTIAL_REQ_TOP(always_true<decltype(spirv::findILsb<T>(experimental::declval<T>()))>)
struct find_lsb_helper<T NBL_PARTIAL_REQ_BOT(always_true<decltype(spirv::findILsb<T>(experimental::declval<T>()))>) >
{
	using return_t = conditional_t<is_vector_v<T>, vector<int32_t, vector_traits<T>::Dimension>, int32_t>;
	static return_t __call(NBL_CONST_REF_ARG(T) val)
	{
		return spirv::findILsb(val);
	}
};
template<typename UInt64> NBL_PARTIAL_REQ_TOP(is_same_v<UInt64, uint64_t>)
struct find_lsb_helper<UInt64 NBL_PARTIAL_REQ_BOT(is_same_v<UInt64, uint64_t>) >
{
	static int32_t __call(NBL_CONST_REF_ARG(uint64_t) val)
	{
		const uint32_t lowBits = uint32_t(val);
		const int32_t lowLsb = find_lsb_helper<uint32_t>::__call(lowBits);

		if (lowLsb == -1)
		{
			const uint32_t highBits = uint32_t(val >> 32);
			const int32_t highLsb = find_lsb_helper<uint32_t>::__call(highBits);
			if (highLsb == -1)
				return -1;
			else
				return 32 + highLsb;
		}

		return lowLsb;
	}
};
template<typename T> NBL_PARTIAL_REQ_TOP(always_true<decltype(spirv::bitReverse<T>(experimental::declval<T>()))>)
struct bitReverse_helper<T NBL_PARTIAL_REQ_BOT(always_true<decltype(spirv::bitReverse<T>(experimental::declval<T>()))>) >
{
	static inline T __call(NBL_CONST_REF_ARG(T) val)
	{
		return spirv::bitReverse(val);
	}
};
template<typename T> NBL_PARTIAL_REQ_TOP(always_true<decltype(spirv::bitReverse<T>(experimental::declval<T>()))>)
struct transpose_helper<T NBL_PARTIAL_REQ_BOT(always_true<decltype(spirv::bitReverse<T>(experimental::declval<T>()))>) >
{
	using transposed_t = typename matrix_traits<T>::transposed_type;

	static transposed_t __call(NBL_CONST_REF_ARG(T) m)
	{
		return spirv::transpose(m);
	}
};
template<typename T> NBL_PARTIAL_REQ_TOP(always_true<decltype(spirv::length<T>(experimental::declval<T>()))>)
struct length_helper<T NBL_PARTIAL_REQ_BOT(always_true<decltype(spirv::length<T>(experimental::declval<T>()))>) >
{
	static inline typename vector_traits<T>::scalar_type __call(NBL_CONST_REF_ARG(T) vec)
	{
		return spirv::length(vec);
	}
};
template<typename T> NBL_PARTIAL_REQ_TOP(always_true<decltype(spirv::normalize<T>(experimental::declval<T>()))>)
struct normalize_helper<T NBL_PARTIAL_REQ_BOT(always_true<decltype(spirv::normalize<T>(experimental::declval<T>()))>) >
{
	static inline T __call(NBL_CONST_REF_ARG(T) vec)
	{
		return spirv::normalize(vec);
	}
};
template<typename T> NBL_PARTIAL_REQ_TOP(always_true<decltype(spirv::fMax<T>(experimental::declval<T>()))>)
struct max_helper<T NBL_PARTIAL_REQ_BOT(always_true<decltype(spirv::fMax<T>(experimental::declval<T>()))>) >
{
	static T __call(NBL_CONST_REF_ARG(T) a, NBL_CONST_REF_ARG(T) b)
	{
		return spirv::fMax(a, b);
	}
};
template<typename T> NBL_PARTIAL_REQ_TOP(always_true<decltype(spirv::uMax<T>(experimental::declval<T>(), experimental::declval<T>()))>)
struct max_helper<T NBL_PARTIAL_REQ_BOT(always_true<decltype(spirv::uMax<T>(experimental::declval<T>(), experimental::declval<T>()))>) >
{
	static T __call(NBL_CONST_REF_ARG(T) a, NBL_CONST_REF_ARG(T) b)
	{
		return spirv::uMax(a, b);
	}
};
template<typename T> NBL_PARTIAL_REQ_TOP(always_true<decltype(spirv::sMax<T>(experimental::declval<T>(), experimental::declval<T>()))>)
struct max_helper<T NBL_PARTIAL_REQ_BOT(always_true<decltype(spirv::sMax<T>(experimental::declval<T>(), experimental::declval<T>()))>) >
{
	static T __call(NBL_CONST_REF_ARG(T) a, NBL_CONST_REF_ARG(T) b)
	{
		return spirv::sMax(a, b);
	}
};

template<typename T> NBL_PARTIAL_REQ_TOP(always_true<decltype(spirv::fMin<T>(experimental::declval<T>(), experimental::declval<T>()))>)
struct min_helper<T NBL_PARTIAL_REQ_BOT(always_true<decltype(spirv::fMin<T>(experimental::declval<T>()), experimental::declval<T>())>) >
{
	static T __call(NBL_CONST_REF_ARG(T) a, NBL_CONST_REF_ARG(T) b)
	{
		return spirv::fMin(a, b);
	}
};
template<typename T> NBL_PARTIAL_REQ_TOP(always_true<decltype(spirv::uMin<T>(experimental::declval<T>(), experimental::declval<T>()))>)
struct min_helper<T NBL_PARTIAL_REQ_BOT(always_true<decltype(spirv::uMin<T>(experimental::declval<T>(), experimental::declval<T>()))>) >
{
	static T __call(NBL_CONST_REF_ARG(T) a, NBL_CONST_REF_ARG(T) b)
	{
		return spirv::uMin(a, b);
	}
};
template<typename T> NBL_PARTIAL_REQ_TOP(always_true<decltype(spirv::sMin<T>(experimental::declval<T>(), experimental::declval<T>()))>)
struct min_helper<T NBL_PARTIAL_REQ_BOT(always_true<decltype(spirv::sMin<T>(experimental::declval<T>(), experimental::declval<T>()))>) >
{
	static T __call(NBL_CONST_REF_ARG(T) a, NBL_CONST_REF_ARG(T) b)
	{
		return spirv::sMin(a, b);
	}
};
#else // C++ only specializations
template<typename T>
requires concepts::Scalar<T>
struct clamp_helper<T, T>
{
	using return_t = T;
	static inline return_t __call(const T val, const T min, const T max)
	{
		return std::clamp(val, min, max);
	}
};
template<typename T>
requires concepts::SignedIntegralScalar<T>
struct bitReverse_helper<T>
{
	static inline T __call(NBL_CONST_REF_ARG(T) arg)
	{
		return glm::bitfieldReverse(arg);
	}
};
template<typename Matrix>
requires concepts::Matrix<Matrix>
struct transpose_helper<Matrix>
{
	using transposed_t = typename matrix_traits<Matrix>::transposed_type;

	static transposed_t __call(NBL_CONST_REF_ARG(Matrix) m)
	{
		return reinterpret_cast<transposed_t&>(glm::transpose(reinterpret_cast<typename Matrix::Base const&>(m)));
	}
};
template<typename Vector>
requires concepts::FloatingPointVector<Vector>
struct length_helper<Vector>
{
	static inline typename vector_traits<Vector>::scalar_type __call(NBL_CONST_REF_ARG(Vector) vec)
	{
		return std::sqrt(dot_helper<Vector>::__call(vec, vec));
	}
};
template<typename Vectorial>
requires concepts::FloatingPointLikeVectorial<Vectorial>
struct normalize_helper<Vectorial>
{
	static inline Vectorial __call(NBL_CONST_REF_ARG(Vectorial) vec)
	{
		return vec / length_helper<Vectorial>::__call(vec);
	}
};
template<typename T>
requires concepts::Scalar<T>
struct max_helper<T>
{
	static T __call(NBL_CONST_REF_ARG(T) a, NBL_CONST_REF_ARG(T) b)
	{
		return std::max(a, b);
	}
};
template<typename T>
requires concepts::Scalar<T>
struct min_helper<T>
{
	static T __call(NBL_CONST_REF_ARG(T) a, NBL_CONST_REF_ARG(T) b)
	{
		return std::min(a, b);
	}
};
template<typename T>
requires concepts::IntegralScalar<T>
struct find_lsb_helper<T>
{
	using return_t = int32_t;
	static inline T __call(const T arg)
	{
		return glm::findLSB(arg);
	}
};
template<typename Integer>
NBL_PARTIAL_REQ_TOP(concepts::IntegralScalar<Integer>)
struct find_msb_helper<Integer NBL_PARTIAL_REQ_BOT(concepts::IntegralScalar<Integer>) >
{
	using return_t = int32_t;
	static return_t __call(NBL_CONST_REF_ARG(Integer) val)
	{
		return glm::findMSB(val);
	}
};
template<typename EnumType>
requires std::is_enum_v<EnumType>
struct find_lsb_helper<EnumType>
{
	static int32_t __call(NBL_CONST_REF_ARG(EnumType) val)
	{
		using underlying_t = std::underlying_type_t<EnumType>;
		return find_lsb_helper<underlying_t>::__call(static_cast<underlying_t>(val));
	}
};
template<typename EnumType>
requires std::is_enum_v<EnumType>
struct find_msb_helper<EnumType>
{
	using return_t = int32_t;
	static return_t __call(NBL_CONST_REF_ARG(EnumType) val)
	{
		using underlying_t = std::underlying_type_t<EnumType>;
		return find_msb_helper<underlying_t>::__call(static_cast<underlying_t>(val));
	}
};

#endif // C++ only specializations

template<typename Vectorial>
NBL_PARTIAL_REQ_TOP(concepts::Vectorial<Vectorial>)
struct dot_helper<Vectorial NBL_PARTIAL_REQ_BOT(concepts::Vectorial<Vectorial>) >
{
	using scalar_type = typename vector_traits<Vectorial>::scalar_type;

	static const uint32_t ArrayDim = 3;
	// TODO: fix
	//static const uint32_t ArrayDim = vector_traits<Vectorial>::Dimension;

	static inline scalar_type __call(NBL_CONST_REF_ARG(Vectorial) lhs, NBL_CONST_REF_ARG(Vectorial) rhs)
	{
		static array_get<Vectorial, scalar_type> getter;

		scalar_type retval = getter(lhs, 0) * getter(rhs, 0);
		for (uint32_t i = 1; i < ArrayDim; ++i)
			retval = retval + getter(lhs, i) * getter(rhs, i);

		return retval;
	}
};
template<typename FloatingPointLikeVectorial>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointLikeVectorial<FloatingPointLikeVectorial> && (vector_traits<FloatingPointLikeVectorial>::Dimension == 3))
struct cross_helper<FloatingPointLikeVectorial NBL_PARTIAL_REQ_BOT(concepts::FloatingPointLikeVectorial<FloatingPointLikeVectorial> && (vector_traits<FloatingPointLikeVectorial>::Dimension == 3)) >
{
	static FloatingPointLikeVectorial __call(NBL_CONST_REF_ARG(FloatingPointLikeVectorial) lhs, NBL_CONST_REF_ARG(FloatingPointLikeVectorial) rhs)
	{
#ifdef __HLSL_VERSION
		if (hlsl::is_vector_v<FloatingPointLikeVectorial>)
			return spirv::cross(lhs, rhs);
#else
		using traits = hlsl::vector_traits<FloatingPointLikeVectorial>;
		array_get<FloatingPointLikeVectorial, typename traits::scalar_type> getter;
		array_set<FloatingPointLikeVectorial, typename traits::scalar_type> setter;

		FloatingPointLikeVectorial output;
		setter(output, 0, getter(lhs, 1) * getter(rhs, 2) - getter(rhs, 1) * getter(lhs, 2));
		setter(output, 1, getter(lhs, 2) * getter(rhs, 0) - getter(rhs, 2) * getter(lhs, 0));
		setter(output, 2, getter(lhs, 0) * getter(rhs, 1) - getter(rhs, 0) * getter(lhs, 1));

		return output;
#endif
	}
};

template<typename Vector>
NBL_PARTIAL_REQ_TOP(is_vector_v<Vector>)
struct clamp_helper<Vector, typename vector_traits<Vector>::scalar_type NBL_PARTIAL_REQ_BOT(is_vector_v<Vector>) >
{
	using return_t = Vector;
	static return_t __call(NBL_CONST_REF_ARG(Vector) val, NBL_CONST_REF_ARG(typename vector_traits<Vector>::scalar_type) min, NBL_CONST_REF_ARG(typename vector_traits<Vector>::scalar_type) max)
	{
		using traits = hlsl::vector_traits<Vector>;
		array_get<Vector, typename traits::scalar_type> getter;
		array_set<return_t, typename traits::scalar_type> setter;

		return_t output;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			setter(output, i, clamp_helper<typename traits::scalar_type, typename traits::scalar_type>::__call(getter(val, i), min, max));

		return output;
	}
};

template<typename Vector>
NBL_PARTIAL_REQ_TOP(hlsl::is_vector_v<Vector>)
struct bitReverse_helper<Vector NBL_PARTIAL_REQ_BOT(concepts::Vectorial<Vector>) >
{
	static Vector __call(NBL_CONST_REF_ARG(Vector) vec)
	{
#ifdef __HLSL_VERSION
		return spirv::bitReverse(vec);
#else
		Vector output;
		using traits = hlsl::vector_traits<Vector>;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			output[i] = bitReverse_helper<traits::scalar_type>::__call(vec[i]);
		return output;
#endif
	}
};
template<typename Vector>
NBL_PARTIAL_REQ_TOP(is_vector_v<Vector>)
struct min_helper<Vector NBL_PARTIAL_REQ_BOT(is_vector_v<Vector>) >
{
	static Vector __call(NBL_CONST_REF_ARG(Vector) a, NBL_CONST_REF_ARG(Vector) b)
	{
		using traits = hlsl::vector_traits<Vector>;
		array_get<Vector, typename traits::scalar_type> getter;
		array_set<Vector, typename traits::scalar_type> setter;

		Vector output;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			setter(output, i, min_helper<typename traits::scalar_type>::__call(getter(a, i), getter(b, i)));

		return output;
	}
};
template<typename Vector>
NBL_PARTIAL_REQ_TOP(concepts::Vectorial<Vector>)
struct max_helper<Vector NBL_PARTIAL_REQ_BOT(concepts::Vectorial<Vector>) >
{
	static Vector __call(NBL_CONST_REF_ARG(Vector) a, NBL_CONST_REF_ARG(Vector) b)
	{
		using traits = hlsl::vector_traits<Vector>;
		array_get<Vector, typename traits::scalar_type> getter;
		array_set<Vector, typename traits::scalar_type> setter;

		Vector output;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			setter(output, i, max_helper<typename traits::scalar_type>::__call(getter(a, i), getter(b, i)));

		return output;
	}
};

template<typename Integer>
NBL_PARTIAL_REQ_TOP(concepts::IntegralScalar<Integer>)
struct bitCount_helper<Integer NBL_PARTIAL_REQ_BOT(concepts::IntegralScalar<Integer>) >
{
	static bitcount_output_t<Integer> __call(NBL_CONST_REF_ARG(Integer) val)
	{
#ifdef __HLSL_VERSION
		if (sizeof(Integer) == 8u)
		{
			uint32_t lowBits = uint32_t(val);
			uint32_t highBits = uint32_t(uint64_t(val) >> 32u);

			return countbits(lowBits) + countbits(highBits);
		}

		return spirv::bitCount(val);
#else
		using UnsignedInteger = typename hlsl::unsigned_integer_of_size_t<sizeof(Integer)>;
		constexpr int32_t BitCnt = sizeof(Integer) * 8u;
		std::bitset<BitCnt> bitset(static_cast<UnsignedInteger>(val));
		return bitset.count();
#endif
	}
};

template<typename LhsT, typename RhsT>
NBL_PARTIAL_REQ_TOP(concepts::Matrix<LhsT> && (concepts::Matrix<RhsT> || concepts::Vector<RhsT>))
struct mul_helper<LhsT, RhsT NBL_PARTIAL_REQ_BOT(concepts::Matrix<LhsT> && (concepts::Matrix<RhsT> || concepts::Vector<RhsT>)) >
{
	static inline mul_output_t<LhsT, RhsT> __call(LhsT lhs, RhsT rhs)
	{
		return mul(lhs, rhs);
	}
};

template<typename SquareMatrix>
NBL_PARTIAL_REQ_TOP(matrix_traits<SquareMatrix>::Square)
struct determinant_helper<SquareMatrix NBL_PARTIAL_REQ_BOT(matrix_traits<SquareMatrix>::Square) >
{
	static typename matrix_traits<SquareMatrix>::scalar_type __call(NBL_CONST_REF_ARG(SquareMatrix) mat)
	{
#ifdef __HLSL_VERSION
		spirv::determinant(mat);
#else
		return glm::determinant(reinterpret_cast<typename SquareMatrix::Base const&>(mat));
#endif
	}
};

template<typename SquareMatrix>
NBL_PARTIAL_REQ_TOP(concepts::Matrix<SquareMatrix>&& matrix_traits<SquareMatrix>::Square)
struct inverse_helper<SquareMatrix NBL_PARTIAL_REQ_BOT(concepts::Matrix<SquareMatrix>&& matrix_traits<SquareMatrix>::Square) >
{
	static SquareMatrix __call(NBL_CONST_REF_ARG(SquareMatrix) mat)
	{
#ifdef __HLSL_VERSION
		return spirv::matrixInverse(mat);
#else
		return reinterpret_cast<SquareMatrix&>(glm::inverse(reinterpret_cast<typename SquareMatrix::Base const&>(mat)));
#endif
	}
};

template<typename Vector>
NBL_PARTIAL_REQ_TOP(concepts::IntVectorial<Vector>)
struct bitCount_helper<Vector NBL_PARTIAL_REQ_BOT(concepts::IntVectorial<Vector>) >
{
	static bitcount_output_t<Vector> __call(NBL_CONST_REF_ARG(Vector) vec)
	{
		using traits = hlsl::vector_traits<Vector>;
		array_get<Vector, typename traits::scalar_type> getter;
		array_set<Vector, typename traits::scalar_type> setter;

		Vector output;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			setter(output, i, bitCount_helper<typename traits::scalar_type>::__call(getter(vec, i)));

		return output;
	}
};
#ifndef __HLSL_VERSION
template<typename EnumT>
requires std::is_enum_v<EnumT>
struct bitCount_helper<EnumT>
{
	using underlying_t = std::underlying_type_t<EnumT>;

	static bitcount_output_t<EnumT> __call(NBL_CONST_REF_ARG(EnumT) val)
	{
		return bitCount_helper<const underlying_t>::__call(reinterpret_cast<const underlying_t&>(val));
	}
};
#endif

template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<FloatingPoint>)
struct rsqrt_helper<FloatingPoint NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<FloatingPoint>) >
{
	static FloatingPoint __call(NBL_CONST_REF_ARG(FloatingPoint) x)
	{
		// TODO: https://stackoverflow.com/a/62239778
#ifdef __HLSL_VERSION
		return spirv::inverseSqrt(x);
#else
		return 1.0f / std::sqrt(x);
#endif
	}
};

template<typename FloatingPointVector>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointVectorial<FloatingPointVector>)
struct rsqrt_helper<FloatingPointVector NBL_PARTIAL_REQ_BOT(concepts::FloatingPointVectorial<FloatingPointVector>) >
{
	static FloatingPointVector __call(NBL_CONST_REF_ARG(FloatingPointVector) x)
	{
		using traits = hlsl::vector_traits<FloatingPointVector>;
		array_get<FloatingPointVector, typename traits::scalar_type> getter;
		array_set<FloatingPointVector, typename traits::scalar_type> setter;

		FloatingPointVector output;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			setter(output, i, rsqrt_helper<typename traits::scalar_type>::__call(getter(x, i)));

		return output;
	}
};

template<typename BooleanVector>
NBL_PARTIAL_REQ_TOP(is_vector_v<BooleanVector>&& is_same_v<typename vector_traits<BooleanVector>::scalar_type, bool>)
struct all_helper<BooleanVector NBL_PARTIAL_REQ_BOT(is_vector_v<BooleanVector>&& is_same_v<typename vector_traits<BooleanVector>::scalar_type, bool>) >
{
	static bool __call(NBL_CONST_REF_ARG(BooleanVector) x)
	{
		using traits = hlsl::vector_traits<BooleanVector>;
		array_get<BooleanVector, typename traits::scalar_type> getter;
		array_set<BooleanVector, typename traits::scalar_type> setter;

		bool output = true;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			output = output && getter(x, i);

		return output;
	}
};

template<typename BooleanVector>
NBL_PARTIAL_REQ_TOP(is_vector_v<BooleanVector>&& is_same_v<typename vector_traits<BooleanVector>::scalar_type, bool>)
struct any_helper<BooleanVector NBL_PARTIAL_REQ_BOT(is_vector_v<BooleanVector>&& is_same_v<typename vector_traits<BooleanVector>::scalar_type, bool>) >
{
	static bool __call(NBL_CONST_REF_ARG(BooleanVector) x)
	{
		using traits = hlsl::vector_traits<BooleanVector>;
		array_get<BooleanVector, typename traits::scalar_type> getter;
		array_set<BooleanVector, typename traits::scalar_type> setter;

		bool output = false;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			output = output || getter(x, i);

		return output;
	}
};

template<typename Vector>
NBL_PARTIAL_REQ_TOP(concepts::Vectorial<Vector>)
struct find_msb_helper<Vector NBL_PARTIAL_REQ_BOT(concepts::Vectorial<Vector>) >
{
	using return_t = vector<int32_t, vector_traits<Vector>::Dimension>;
	static return_t __call(NBL_CONST_REF_ARG(Vector) vec)
	{
		using traits = hlsl::vector_traits<Vector>;
		using intput_vec_scalar_t = typename traits::scalar_type;
		array_get<Vector, intput_vec_scalar_t> getter;
		array_set<return_t, int32_t> setter;

		return_t output;

		for (uint32_t i = 0; i < traits::Dimension; ++i)
			setter(output, i, find_msb_helper<intput_vec_scalar_t>::__call(getter(vec, i)));

		return output;
	}
};

template<typename Vector>
NBL_PARTIAL_REQ_TOP(concepts::Vectorial<Vector>)
struct find_lsb_helper<Vector NBL_PARTIAL_REQ_BOT(concepts::Vectorial<Vector>) >
{
	using return_t = vector<int32_t, vector_traits<Vector>::Dimension>;
	static return_t __call(NBL_CONST_REF_ARG(Vector) vec)
	{
		using traits = hlsl::vector_traits<Vector>;
		using intput_vec_scalar_t = typename traits::scalar_type;
		array_get<Vector, intput_vec_scalar_t> getter;
		array_set<return_t, int32_t> setter;

		return_t output;

		for (uint32_t i = 0; i < traits::Dimension; ++i)
			setter(output, i, find_lsb_helper<intput_vec_scalar_t>::__call(getter(vec, i)));

		return output;
	}
};
}
}
}

#endif