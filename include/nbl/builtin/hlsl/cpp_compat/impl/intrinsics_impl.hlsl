#ifndef _NBL_BUILTIN_HLSL_CPP_COMPAT_IMPL_INTRINSICS_IMPL_INCLUDED_
#define _NBL_BUILTIN_HLSL_CPP_COMPAT_IMPL_INTRINSICS_IMPL_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/basic.h>
#include <nbl/builtin/hlsl/matrix_utils/matrix_traits.hlsl>
#include <nbl/builtin/hlsl/concepts.hlsl>
#include <nbl/builtin/hlsl/spirv_intrinsics/core.hlsl>
#include <nbl/builtin/hlsl/spirv_intrinsics/glsl.std.450.hlsl>

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

	static inline scalar_type dot_product(NBL_CONST_REF_ARG(T) lhs, NBL_CONST_REF_ARG(T) rhs)
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
	static inline ScalarType dot_product(NBL_CONST_REF_ARG(VectorType) lhs, NBL_CONST_REF_ARG(VectorType) rhs)\
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

template<typename Integer>
struct find_msb_helper;

template<>
struct find_msb_helper<uint32_t>
{
	static int32_t findMSB(NBL_CONST_REF_ARG(uint32_t) val)
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
	static int32_t findMSB(NBL_CONST_REF_ARG(int32_t) val)
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
	static int32_t findMSB(NBL_CONST_REF_ARG(INPUT_INTEGER_TYPE) val)\
	{\
		return find_msb_helper<INTEGER_TYPE>::findMSB(val);\
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
	static int32_t findMSB(NBL_CONST_REF_ARG(uint64_t) val)
	{
#ifdef __HLSL_VERSION
		const uint32_t highBits = uint32_t(val >> 32);
		const int32_t highMsb = find_msb_helper<uint32_t>::findMSB(highBits);

		if (highMsb == -1)
		{
			const uint32_t lowBits = uint32_t(val);
			const int32_t lowMsb = find_msb_helper<uint32_t>::findMSB(lowBits);
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
	static vector<int32_t, N> findMSB(NBL_CONST_REF_ARG(vector<uint32_t, N>) val)
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
	static vector<int32_t, N> findMSB(NBL_CONST_REF_ARG(vector<int32_t, N>) val)
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
	static int32_t findMSB(NBL_CONST_REF_ARG(EnumType) val)
	{
		using underlying_t = std::underlying_type_t<EnumType>;
		return find_msb_helper<underlying_t>::findMSB(static_cast<underlying_t>(val));
	}
};

#endif

template<typename Integer>
struct find_lsb_helper;

template<>
struct find_lsb_helper<int32_t>
{
	static int32_t findLSB(NBL_CONST_REF_ARG(int32_t) val)
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
	static int32_t findLSB(NBL_CONST_REF_ARG(uint32_t) val)
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
	static int32_t findLSB(NBL_CONST_REF_ARG(INPUT_INTEGER_TYPE) val)\
	{\
		return find_lsb_helper<INTEGER_TYPE>::findLSB(val);\
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
	static int32_t findLSB(NBL_CONST_REF_ARG(uint64_t) val)
	{
#ifdef __HLSL_VERSION
		const uint32_t lowBits = uint32_t(val);
		const int32_t lowLsb = find_lsb_helper<uint32_t>::findLSB(lowBits);

		if (lowLsb == -1)
		{
			const uint32_t highBits = uint32_t(val >> 32);
			const int32_t highLsb = find_lsb_helper<uint32_t>::findLSB(highBits);
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
	static vector<int32_t, N> findLSB(NBL_CONST_REF_ARG(vector<int32_t, N>) val)
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
	static vector<int32_t, N> findLSB(NBL_CONST_REF_ARG(vector<uint32_t, N>) val)
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
	static int32_t findLSB(NBL_CONST_REF_ARG(EnumType) val)
	{
		using underlying_t = std::underlying_type_t<EnumType>;
		return find_lsb_helper<underlying_t>::findLSB(static_cast<underlying_t>(val));
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
			output[i] = bitReverse_helper<scalar_type_t<Vector> >::__call(vec[i]);
		return output;
#endif
	}
};


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
	static inline TYPE lerp(NBL_CONST_REF_ARG(TYPE) x, NBL_CONST_REF_ARG(TYPE) y, NBL_CONST_REF_ARG(TYPE) a)\
	{\
		return MIX_FUNCTION(x, y, a);\
	}\
};\
\
template<int N>\
struct lerp_helper<vector<TYPE, N>, vector<TYPE, N> >\
{\
	static inline vector<TYPE, N> lerp(NBL_CONST_REF_ARG(vector<TYPE, N>) x, NBL_CONST_REF_ARG(vector<TYPE, N>) y, NBL_CONST_REF_ARG(vector<TYPE, N>) a)\
	{\
		return MIX_FUNCTION(x, y, a);\
	}\
};\
\
template<int N>\
struct lerp_helper<vector<TYPE, N>, TYPE>\
{\
	static inline vector<TYPE, N> lerp(NBL_CONST_REF_ARG(vector<TYPE, N>) x, NBL_CONST_REF_ARG(vector<TYPE, N>) y, NBL_CONST_REF_ARG(TYPE) a)\
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
	static inline T lerp(NBL_CONST_REF_ARG(T) x, NBL_CONST_REF_ARG(T) y, NBL_CONST_REF_ARG(bool) a)
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

	static inline output_vec_t lerp(NBL_CONST_REF_ARG(output_vec_t) x, NBL_CONST_REF_ARG(output_vec_t) y, NBL_CONST_REF_ARG(vector<bool, N>) a)
	{
		output_vec_t retval;
		for (uint32_t i = 0; i < vector_traits<output_vec_t>::Dimension; i++)
			retval[i] = a[i] ? y[i] : x[i];
		return retval;
	}
};

template<typename Matrix>
struct transpose_helper;

template<typename T, int N, int M>
struct transpose_helper<matrix<T, N, M> >
{
	using transposed_t = typename matrix_traits<matrix<T, N, M> >::transposed_type;

	static transposed_t transpose(NBL_CONST_REF_ARG(matrix<T, N, M>) m)
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
	static inline RhsT multiply(LhsT lhs, RhsT rhs)
	{
		return mul(lhs, rhs);
	}
};

}
}
}

#endif