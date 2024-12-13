#ifndef _NBL_BUILTIN_HLSL_CPP_COMPAT_IMPL_INTRINSICS_IMPL_INCLUDED_
#define _NBL_BUILTIN_HLSL_CPP_COMPAT_IMPL_INTRINSICS_IMPL_INCLUDED_

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

#ifdef __HLSL_VERSION
template<typename Integer>
struct find_msb_helper;
#else
// legacy code wouldn't work without it
template<typename Integer>
struct find_msb_helper
{
	static int findMSB(NBL_CONST_REF_ARG(Integer) val)
	{
		if (is_signed_v<Integer>)
		{
			// GLM accepts only integer types, so idea is to cast input to integer type
			using as_int = typename integer_of_size<sizeof(Integer)>::type;
			const as_int valAsInt = reinterpret_cast<const as_int&>(val);
			return glm::findMSB(valAsInt);
		}
		else
		{
			// GLM accepts only integer types, so idea is to cast input to integer type
			using as_uint = typename unsigned_integer_of_size<sizeof(Integer)>::type;
			const as_uint valAsUnsignedInt = reinterpret_cast<const as_uint&>(val);
			return glm::findMSB(valAsUnsignedInt);
		}
	}
};
#endif

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

#ifdef __HLSL_VERSION
template<typename Integer>
struct find_lsb_helper;
#else
// legacy code wouldn't work without it
template<typename Integer>
struct find_lsb_helper
{
	static int32_t findLSB(NBL_CONST_REF_ARG(Integer) val)
	{
#ifdef __HLSL_VERSION
		return spirv::findILsb(val);
#else
		if (is_signed_v<Integer>)
		{
			// GLM accepts only integer types, so idea is to cast input to integer type
			using as_int = typename integer_of_size<sizeof(Integer)>::type;
			const as_int valAsInt = reinterpret_cast<const as_int&>(val);
			return glm::findLSB(valAsInt);
		}
		else
		{
			// GLM accepts only integer types, so idea is to cast input to integer type
			using as_uint = typename unsigned_integer_of_size<sizeof(Integer)>::type;
			const as_uint valAsUnsignedInt = reinterpret_cast<const as_uint&>(val);
			return glm::findLSB(valAsUnsignedInt);
		}
#endif
	}
};
#endif

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

template<typename T, typename U>
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
		return a ? y : x;
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

}
}
}

#endif