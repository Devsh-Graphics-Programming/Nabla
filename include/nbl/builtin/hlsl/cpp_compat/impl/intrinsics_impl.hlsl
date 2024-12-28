#ifndef _NBL_BUILTIN_HLSL_CPP_COMPAT_IMPL_INTRINSICS_IMPL_INCLUDED_
#define _NBL_BUILTIN_HLSL_CPP_COMPAT_IMPL_INTRINSICS_IMPL_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/basic.h>
#include <nbl/builtin/hlsl/matrix_utils/matrix_traits.hlsl>
#include <nbl/builtin/hlsl/matrix_utils/mul_output_t.hlsl>
#include <nbl/builtin/hlsl/concepts.hlsl>
#include <nbl/builtin/hlsl/spirv_intrinsics/core.hlsl>
#include <nbl/builtin/hlsl/spirv_intrinsics/glsl.std.450.hlsl>
#include <nbl/builtin/hlsl/ieee754.hlsl>

#ifndef __HLSL_VERSION
#include <bitset>
#endif

namespace nbl
{
namespace hlsl
{
namespace cpp_compat_intrinsics_impl
{
template<typename T>
struct dot_helper
{
	using scalar_type = typename vector_traits<T>::scalar_type;

	static inline scalar_type __call(NBL_CONST_REF_ARG(T) lhs, NBL_CONST_REF_ARG(T) rhs)
	{
		static array_get<T, scalar_type> getter;
		scalar_type retval = getter(lhs, 0) * getter(rhs, 0);

		static const uint32_t ArrayDim = vector_traits<T>::Dimension;
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
	static inline ScalarType __call(NBL_CONST_REF_ARG(VectorType) lhs, NBL_CONST_REF_ARG(VectorType) rhs)\
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

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct cross_helper;

//! this specialization will work only with hlsl::vector<T, 3> type
template<typename FloatingPointVector>
NBL_PARTIAL_REQ_TOP(is_floating_point_v<FloatingPointVector> && is_vector_v<FloatingPointVector> && (vector_traits<FloatingPointVector>::Dimension == 3))
struct cross_helper<FloatingPointVector NBL_PARTIAL_REQ_BOT(is_floating_point_v<FloatingPointVector>&& is_vector_v<FloatingPointVector>&& (vector_traits<FloatingPointVector>::Dimension == 3)) >
{
	static FloatingPointVector __call(NBL_CONST_REF_ARG(FloatingPointVector) lhs, NBL_CONST_REF_ARG(FloatingPointVector) rhs)
	{
#ifdef __HLSL_VERSION
		return spirv::cross(lhs, rhs);
#else
		FloatingPointVector output;
		output.x = lhs[1] * rhs[2] - rhs[1] * lhs[2];
		output.y = lhs[2] * rhs[0] - rhs[2] * lhs[0];
		output.z = lhs[0] * rhs[1] - rhs[0] * lhs[1];

		return output;
#endif
	}
};

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct clamp_helper;

template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(is_floating_point_v<FloatingPoint> && is_scalar_v<FloatingPoint>)
struct clamp_helper<FloatingPoint NBL_PARTIAL_REQ_BOT(is_floating_point_v<FloatingPoint> && is_scalar_v<FloatingPoint>) >
{
	static FloatingPoint __call(NBL_CONST_REF_ARG(FloatingPoint) val, NBL_CONST_REF_ARG(FloatingPoint) min, NBL_CONST_REF_ARG(FloatingPoint) max)
	{
#ifdef __HLSL_VERSION
		return spirv::fClamp(val, min, max);
#else
		return std::clamp(val, min, max);
#endif
	}
};

template<typename UnsignedInteger>
NBL_PARTIAL_REQ_TOP(is_integral_v<UnsignedInteger> && !is_signed_v<UnsignedInteger> && is_scalar_v<UnsignedInteger>)
struct clamp_helper<UnsignedInteger NBL_PARTIAL_REQ_BOT(is_integral_v<UnsignedInteger> && !is_signed_v<UnsignedInteger>&& is_scalar_v<UnsignedInteger>) >
{
	static UnsignedInteger __call(NBL_CONST_REF_ARG(UnsignedInteger) val, NBL_CONST_REF_ARG(UnsignedInteger) min, NBL_CONST_REF_ARG(UnsignedInteger) max)
	{
#ifdef __HLSL_VERSION
		return spirv::uClamp(val, min, max);
#else
		return std::clamp(val, min, max);
#endif
	}
};

template<typename Integer>
NBL_PARTIAL_REQ_TOP(is_integral_v<Integer> && is_signed_v<Integer>&& is_scalar_v<Integer>)
struct clamp_helper<Integer NBL_PARTIAL_REQ_BOT(is_integral_v<Integer>&& is_signed_v<Integer>&& is_scalar_v<Integer>) >
{
	static Integer __call(NBL_CONST_REF_ARG(Integer) val, NBL_CONST_REF_ARG(Integer) min, NBL_CONST_REF_ARG(Integer) max)
	{
#ifdef __HLSL_VERSION
		return spirv::sClamp(val, min, max);
#else
		return std::clamp(val, min, max);
#endif
	}
};

template<typename Vector>
NBL_PARTIAL_REQ_TOP(is_vector_v<Vector>)
struct clamp_helper<Vector NBL_PARTIAL_REQ_BOT(is_vector_v<Vector>) >
{
	static Vector __call(NBL_CONST_REF_ARG(Vector) val, NBL_CONST_REF_ARG(typename vector_traits<Vector>::scalar_type) min, NBL_CONST_REF_ARG(typename vector_traits<Vector>::scalar_type) max)
	{
		using traits = hlsl::vector_traits<Vector>;
		array_get<Vector, typename traits::scalar_type> getter;
		array_set<Vector, typename traits::scalar_type> setter;

		Vector output;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			setter(output, i, clamp_helper<typename traits::scalar_type>::__call(getter(val, i), min, max));

		return output;
	}
};

template<typename Integer>
struct find_msb_helper;

template<>
struct find_msb_helper<uint32_t>
{
	static int32_t __call(NBL_CONST_REF_ARG(uint32_t) val)
	{
#ifdef __HLSL_VERSION
		return spirv::findUMsb(val);
#else
		return glm::findMSB(val);
#endif
	}
};

template<>
struct find_msb_helper<int32_t>
{
	static int32_t __call(NBL_CONST_REF_ARG(int32_t) val)
	{
#ifdef __HLSL_VERSION
		return spirv::findSMsb(val);
#else
		return glm::findMSB(val);
#endif
	}
};

#define DEFINE_FIND_MSB_COMMON_SPECIALIZATION(INPUT_INTEGER_TYPE, INTEGER_TYPE)\
template<>\
struct find_msb_helper<INPUT_INTEGER_TYPE>\
{\
	static int32_t __call(NBL_CONST_REF_ARG(INPUT_INTEGER_TYPE) val)\
	{\
		return find_msb_helper<INTEGER_TYPE>::__call(val);\
	}\
};\

DEFINE_FIND_MSB_COMMON_SPECIALIZATION(int16_t, int32_t)
DEFINE_FIND_MSB_COMMON_SPECIALIZATION(uint16_t, uint32_t)
#ifndef __HLSL_VERSION
DEFINE_FIND_MSB_COMMON_SPECIALIZATION(int8_t, int32_t)
DEFINE_FIND_MSB_COMMON_SPECIALIZATION(uint8_t, uint32_t)
#endif

template<>
struct find_msb_helper<uint64_t>
{
	static int32_t __call(NBL_CONST_REF_ARG(uint64_t) val)
	{
#ifdef __HLSL_VERSION
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
#else
		return glm::findMSB(val);
#endif
	}
};

template<int N>
struct find_msb_helper<vector<uint32_t, N> >
{
	static vector<int32_t, N> __call(NBL_CONST_REF_ARG(vector<uint32_t, N>) val)
	{
#ifdef __HLSL_VERSION
		return spirv::findUMsb(val);
#else
		return glm::findMSB(val);
#endif
	}
};

template<int N>
struct find_msb_helper<vector<int32_t, N> >
{
	static vector<int32_t, N> __call(NBL_CONST_REF_ARG(vector<int32_t, N>) val)
	{
#ifdef __HLSL_VERSION
		return spirv::findSMsb(val);
#else
		return glm::findMSB(val);
#endif
	}
};

#ifndef __HLSL_VERSION

template<typename EnumType>
	requires std::is_enum_v<EnumType>
struct find_msb_helper<EnumType>
{
	static int32_t __call(NBL_CONST_REF_ARG(EnumType) val)
	{
		using underlying_t = std::underlying_type_t<EnumType>;
		return find_msb_helper<underlying_t>::__call(static_cast<underlying_t>(val));
	}
};

#endif

template<typename Integer>
struct find_lsb_helper;

template<>
struct find_lsb_helper<int32_t>
{
	static int32_t __call(NBL_CONST_REF_ARG(int32_t) val)
	{
#ifdef __HLSL_VERSION
		return spirv::findILsb(val);
#else
		return glm::findLSB(val);
#endif
	}
};

template<>
struct find_lsb_helper<uint32_t>
{
	static int32_t __call(NBL_CONST_REF_ARG(uint32_t) val)
	{
#ifdef __HLSL_VERSION
		return spirv::findILsb(val);
#else
		return glm::findLSB(val);
#endif
	}
};

#define DEFINE_FIND_LSB_COMMON_SPECIALIZATION(INPUT_INTEGER_TYPE, INTEGER_TYPE)\
template<>\
struct find_lsb_helper<INPUT_INTEGER_TYPE>\
{\
	static int32_t __call(NBL_CONST_REF_ARG(INPUT_INTEGER_TYPE) val)\
	{\
		return find_lsb_helper<INTEGER_TYPE>::__call(val);\
	}\
};\

DEFINE_FIND_LSB_COMMON_SPECIALIZATION(int16_t, int32_t)
DEFINE_FIND_LSB_COMMON_SPECIALIZATION(uint16_t, uint32_t)
#ifndef __HLSL_VERSION
DEFINE_FIND_LSB_COMMON_SPECIALIZATION(int8_t, int32_t)
DEFINE_FIND_LSB_COMMON_SPECIALIZATION(uint8_t, uint32_t)
#endif

template<>
struct find_lsb_helper<uint64_t>
{
	static int32_t __call(NBL_CONST_REF_ARG(uint64_t) val)
	{
#ifdef __HLSL_VERSION
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
#else
		return glm::findLSB(val);
#endif
	}
};

template<int N>
struct find_lsb_helper<vector<int32_t, N> >
{
	static vector<int32_t, N> __call(NBL_CONST_REF_ARG(vector<int32_t, N>) val)
	{
#ifdef __HLSL_VERSION
		return spirv::findILsb(val);
#else
		return glm::findLSB(val);
#endif
	}
};

template<int N>
struct find_lsb_helper<vector<uint32_t, N> >
{
	static vector<int32_t, N> __call(NBL_CONST_REF_ARG(vector<uint32_t, N>) val)
	{
#ifdef __HLSL_VERSION
		return spirv::findILsb(val);
#else
		return glm::findLSB(val);
#endif
	}
};

#ifndef __HLSL_VERSION

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

#endif

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
struct bitReverse_helper;

template<typename Integer>
NBL_PARTIAL_REQ_TOP(hlsl::is_integral_v<Integer> && hlsl::is_scalar_v<Integer>)
struct bitReverse_helper<Integer NBL_PARTIAL_REQ_BOT(hlsl::is_integral_v<Integer>&& hlsl::is_scalar_v<Integer>) >
{
	static inline Integer __call(NBL_CONST_REF_ARG(Integer) val)
	{
#ifdef __HLSL_VERSION
		return spirv::bitReverse(val);
#else
		return glm::bitfieldReverse(val);
#endif
	}
};

template<typename Vector>
NBL_PARTIAL_REQ_TOP(hlsl::is_vector_v<Vector>)
struct bitReverse_helper<Vector NBL_PARTIAL_REQ_BOT(hlsl::is_integral_v<Vector> && hlsl::is_vector_v<Vector>) >
{
	static Vector __call(NBL_CONST_REF_ARG(Vector) vec)
	{
#ifdef __HLSL_VERSION
		return spirv::bitReverse(vec);
#else
		Vector output;
		using traits = hlsl::vector_traits<Vector>;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			output[i] = bitReverse_helper<traits::scalar_type >::__call(vec[i]);
		return output;
#endif
	}
};

template<typename Matrix>
struct transpose_helper;

template<typename T, int N, int M>
struct transpose_helper<matrix<T, N, M> >
{
	using transposed_t = typename matrix_traits<matrix<T, N, M> >::transposed_type;

	static transposed_t __call(NBL_CONST_REF_ARG(matrix<T, N, M>) m)
	{
#ifdef __HLSL_VERSION
		return spirv::transpose(m);
#else
		return reinterpret_cast<transposed_t&>(glm::transpose(reinterpret_cast<typename matrix<T, N, M>::Base const&>(m)));
#endif
	}
};

template<typename LhsT, typename RhsT>
struct mul_helper
{
	static inline mul_output_t<LhsT, RhsT> __call(LhsT lhs, RhsT rhs)
	{
		return mul(lhs, rhs);
	}
};

// TODO: some struct that with other functions, since more functions will need that.. 
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct bitcount_output;

template<typename Integer>
NBL_PARTIAL_REQ_TOP(hlsl::is_integral_v<Integer> && hlsl::is_scalar_v<Integer>)
struct bitcount_output<Integer NBL_PARTIAL_REQ_BOT(hlsl::is_integral_v<Integer>&& hlsl::is_scalar_v<Integer>) >
{
	using type = int32_t;
};

template<typename IntegerVector>
NBL_PARTIAL_REQ_TOP(hlsl::is_integral_v<IntegerVector> && hlsl::is_vector_v<IntegerVector>)
struct bitcount_output<IntegerVector NBL_PARTIAL_REQ_BOT(hlsl::is_integral_v<IntegerVector> && hlsl::is_vector_v<IntegerVector>) >
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

template<typename Integer NBL_STRUCT_CONSTRAINABLE>
struct bitCount_helper;

template<typename Integer>
NBL_PARTIAL_REQ_TOP(hlsl::is_integral_v<Integer>&& hlsl::is_scalar_v<Integer>)
struct bitCount_helper<Integer NBL_PARTIAL_REQ_BOT(hlsl::is_integral_v<Integer>&& hlsl::is_scalar_v<Integer>) >
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

template<typename Vector>
NBL_PARTIAL_REQ_TOP(hlsl::is_integral_v<Vector> && hlsl::is_vector_v<Vector>)
struct bitCount_helper<Vector NBL_PARTIAL_REQ_BOT(hlsl::is_integral_v<Vector> && hlsl::is_vector_v<Vector>) >
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

template<typename Vector NBL_STRUCT_CONSTRAINABLE>
struct length_helper;

template<typename Vector>
NBL_PARTIAL_REQ_TOP(hlsl::is_floating_point_v<Vector>&& hlsl::is_vector_v<Vector>)
struct length_helper<Vector NBL_PARTIAL_REQ_BOT(hlsl::is_floating_point_v<Vector>&& hlsl::is_vector_v<Vector>) >
{
	static inline typename vector_traits<Vector>::scalar_type __call(NBL_CONST_REF_ARG(Vector) vec)
	{
#ifdef __HLSL_VERSION
		return spirv::length(vec);
#else
		return std::sqrt(dot_helper<Vector>::__call(vec, vec));
#endif
	}
};

template<typename Vector NBL_STRUCT_CONSTRAINABLE>
struct normalize_helper;

template<typename Vector>
NBL_PARTIAL_REQ_TOP(hlsl::is_floating_point_v<Vector> && hlsl::is_vector_v<Vector>)
struct normalize_helper<Vector NBL_PARTIAL_REQ_BOT(hlsl::is_floating_point_v<Vector> && hlsl::is_vector_v<Vector>) >
{
	static inline Vector __call(NBL_CONST_REF_ARG(Vector) vec)
	{
#ifdef __HLSL_VERSION
		return spirv::normalize(vec);
#else
		return vec / length_helper<Vector>::__call(vec);
#endif
	}
};

// MIN

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct min_helper;

template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(is_floating_point_v<FloatingPoint>&& is_scalar_v<FloatingPoint>)
struct min_helper<FloatingPoint NBL_PARTIAL_REQ_BOT(is_floating_point_v<FloatingPoint>&& is_scalar_v<FloatingPoint>) >
{
	static FloatingPoint __call(NBL_CONST_REF_ARG(FloatingPoint) a, NBL_CONST_REF_ARG(FloatingPoint) b)
	{
#ifdef __HLSL_VERSION
		return spirv::fMin(a, b);
#else
		return std::min(a, b);
#endif
	}
};

template<typename UnsignedInteger>
NBL_PARTIAL_REQ_TOP(is_integral_v<UnsignedInteger> && !is_signed_v<UnsignedInteger>&& is_scalar_v<UnsignedInteger>)
struct min_helper<UnsignedInteger NBL_PARTIAL_REQ_BOT(is_integral_v<UnsignedInteger> && !is_signed_v<UnsignedInteger>&& is_scalar_v<UnsignedInteger>) >
{
	static UnsignedInteger __call(NBL_CONST_REF_ARG(UnsignedInteger) a, NBL_CONST_REF_ARG(UnsignedInteger) b)
	{
#ifdef __HLSL_VERSION
		return spirv::uMin(a, b);
#else
		return std::min(a, b);
#endif
	}
};

template<typename Integer>
NBL_PARTIAL_REQ_TOP(is_integral_v<Integer>&& is_signed_v<Integer>&& is_scalar_v<Integer>)
struct min_helper<Integer NBL_PARTIAL_REQ_BOT(is_integral_v<Integer>&& is_signed_v<Integer>&& is_scalar_v<Integer>) >
{
	static Integer __call(NBL_CONST_REF_ARG(Integer) a, NBL_CONST_REF_ARG(Integer) b)
	{
#ifdef __HLSL_VERSION
		return spirv::sMin(a, b);
#else
		return std::min(a, b);
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

// MAX

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct max_helper;

template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(is_floating_point_v<FloatingPoint>&& is_scalar_v<FloatingPoint>)
struct max_helper<FloatingPoint NBL_PARTIAL_REQ_BOT(is_floating_point_v<FloatingPoint>&& is_scalar_v<FloatingPoint>) >
{
	static FloatingPoint __call(NBL_CONST_REF_ARG(FloatingPoint) a, NBL_CONST_REF_ARG(FloatingPoint) b)
	{
#ifdef __HLSL_VERSION
		return spirv::fMax(a, b);
#else
		return std::max(a, b);
#endif
	}
};

template<typename UnsignedInteger>
NBL_PARTIAL_REQ_TOP(is_integral_v<UnsignedInteger> && !is_signed_v<UnsignedInteger>&& is_scalar_v<UnsignedInteger>)
struct max_helper<UnsignedInteger NBL_PARTIAL_REQ_BOT(is_integral_v<UnsignedInteger> && !is_signed_v<UnsignedInteger>&& is_scalar_v<UnsignedInteger>) >
{
	static UnsignedInteger __call(NBL_CONST_REF_ARG(UnsignedInteger) a, NBL_CONST_REF_ARG(UnsignedInteger) b)
	{
#ifdef __HLSL_VERSION
		return spirv::uMax(a, b);
#else
		return std::max(a, b);
#endif
	}
};

template<typename Integer>
NBL_PARTIAL_REQ_TOP(is_integral_v<Integer>&& is_signed_v<Integer>&& is_scalar_v<Integer>)
struct max_helper<Integer NBL_PARTIAL_REQ_BOT(is_integral_v<Integer>&& is_signed_v<Integer>&& is_scalar_v<Integer>) >
{
	static Integer __call(NBL_CONST_REF_ARG(Integer) a, NBL_CONST_REF_ARG(Integer) b)
	{
#ifdef __HLSL_VERSION
		return spirv::sMax(a, b);
#else
		return std::max(a, b);
#endif
	}
};

template<typename Vector>
NBL_PARTIAL_REQ_TOP(is_vector_v<Vector>)
struct max_helper<Vector NBL_PARTIAL_REQ_BOT(is_vector_v<Vector>) >
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

// DETERMINANT

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct determinant_helper;

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

// INVERSE

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct inverse_helper;

template<typename SquareMatrix>
NBL_PARTIAL_REQ_TOP(matrix_traits<SquareMatrix>::Square)
struct inverse_helper<SquareMatrix NBL_PARTIAL_REQ_BOT(matrix_traits<SquareMatrix>::Square) >
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

// RSQRT

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct rsqrt_helper;

template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(is_floating_point_v<FloatingPoint> && is_scalar_v<FloatingPoint>)
struct rsqrt_helper<FloatingPoint NBL_PARTIAL_REQ_BOT(is_floating_point_v<FloatingPoint> && is_scalar_v<FloatingPoint>) >
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
NBL_PARTIAL_REQ_TOP(is_vector_v<FloatingPointVector> && is_floating_point_v<typename vector_traits<FloatingPointVector>::scalar_type>)
struct rsqrt_helper<FloatingPointVector NBL_PARTIAL_REQ_BOT(is_vector_v<FloatingPointVector>&& is_floating_point_v<typename vector_traits<FloatingPointVector>::scalar_type>) >
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

}
}
}

#endif