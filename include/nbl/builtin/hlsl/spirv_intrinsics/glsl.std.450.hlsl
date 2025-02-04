#ifndef _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_GLSL_STD_450_INCLUDED_
#define _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_GLSL_STD_450_INCLUDED_

#ifdef __HLSL_VERSION
#include <nbl/builtin/hlsl/vector_utils/vector_traits.hlsl>
#include <nbl/builtin/hlsl/matrix_utils/matrix_traits.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/basic.h>
#include <nbl/builtin/hlsl/concepts.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/output_structs.hlsl>
#include "spirv/unified1/GLSL.std.450.h"

namespace nbl
{
namespace hlsl
{
namespace spirv
{

namespace concepts
{
// scalar or vector whose component type is floating-point.
template<typename T>
NBL_BOOL_CONCEPT FloatingPointVectorOrScalar = is_floating_point_v<T> && (!is_matrix_v<T>);
//  scalar or vector whose component type is 16-bit or 32-bit floating-point.
template<typename T>
NBL_BOOL_CONCEPT FloatingPointVectorOrScalar32or16BitSize = FloatingPointVectorOrScalar<T> && (sizeof(typename vector_traits<T>::scalar_type) == 4 || sizeof(typename vector_traits<T>::scalar_type) == 2);
//is interpreted as signed
//integer scalar or integer vector types
template<typename T>
NBL_BOOL_CONCEPT IntegralVectorOrScalar = is_integral_v<T> && is_signed_v<T> && !is_matrix_v<T>;
//interpreted as unsigned
//integer scalar or integer vector types
template<typename T>
NBL_BOOL_CONCEPT UnsignedIntegralVectorOrScalar = is_integral_v<T> && is_unsigned_v<T> && !is_matrix_v<T>;
//be signed integer scalar or signed integer vector types
//This instruction is currently limited to 32 - bit width components.
template<typename T>
NBL_BOOL_CONCEPT IntegralVectorOrScalar32BitSize = IntegralVectorOrScalar<T> && (sizeof(typename vector_traits<T>::scalar_type) == 4);
//be unsigned integer scalar or unsigned integer vector types
//This instruction is currently limited to 32 - bit width components.
template<typename T>
NBL_BOOL_CONCEPT UnsignedIntegralVectorOrScalar32BitSize = UnsignedIntegralVectorOrScalar<T> && (sizeof(typename vector_traits<T>::scalar_type) == 4);
}

// Find MSB and LSB restricted to 32-bit width component types https://registry.khronos.org/SPIR-V/specs/unified1/GLSL.std.450.html
template<typename T NBL_FUNC_REQUIRES(concepts::IntegralVectorOrScalar32BitSize<T> || concepts::UnsignedIntegralVectorOrScalar32BitSize<T>)
[[vk::ext_instruction(GLSLstd450::GLSLstd450FindILsb, "GLSL.std.450")]]
conditional_t<is_vector_v<T>, vector<int32_t, vector_traits<T>::Dimension>, int32_t> findILsb(T value);

template<typename T NBL_FUNC_REQUIRES(concepts::IntegralVectorOrScalar32BitSize<T>)
[[vk::ext_instruction(GLSLstd450::GLSLstd450FindSMsb, "GLSL.std.450")]]
conditional_t<is_vector_v<T>, vector<int32_t, vector_traits<T>::Dimension>, int32_t> findSMsb(T value);

template<typename T NBL_FUNC_REQUIRES(concepts::UnsignedIntegralVectorOrScalar32BitSize<T>)
[[vk::ext_instruction(GLSLstd450::GLSLstd450FindUMsb, "GLSL.std.450")]]
conditional_t<is_vector_v<T>, vector<int32_t, vector_traits<T>::Dimension>, int32_t> findUMsb(T value);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar32or16BitSize<T>)
[[vk::ext_instruction(GLSLstd450::GLSLstd450Pow, "GLSL.std.450")]]
T pow(T lhs, T rhs);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar32or16BitSize<T>)
[[vk::ext_instruction(GLSLstd450::GLSLstd450Exp, "GLSL.std.450")]]
T exp(T val);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar32or16BitSize<T>)
[[vk::ext_instruction(GLSLstd450::GLSLstd450Exp2, "GLSL.std.450")]]
T exp2(T val);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar32or16BitSize<T>)
[[vk::ext_instruction(GLSLstd450::GLSLstd450Log, "GLSL.std.450")]]
T log(T val);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar32or16BitSize<T>)
[[vk::ext_instruction(GLSLstd450::GLSLstd450Log2, "GLSL.std.450")]]
T log2(T val);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450::GLSLstd450Sqrt, "GLSL.std.450")]]
T sqrt(T val);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450::GLSLstd450InverseSqrt, "GLSL.std.450")]]
T inverseSqrt(T val);
 
template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450::GLSLstd450Floor, "GLSL.std.450")]]
T floor(T val);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar<T> && is_scalar_v<T>)
[[vk::ext_instruction(GLSLstd450::GLSLstd450Cross, "GLSL.std.450")]]
vector<T, 3> cross(NBL_CONST_REF_ARG(vector<T, 3>) lhs, NBL_CONST_REF_ARG(vector<T, 3>) rhs);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450::GLSLstd450FMix, "GLSL.std.450")]]
T fMix(T x, T y, T a);

template<typename T, int N>
[[vk::ext_instruction(GLSLstd450::GLSLstd450Determinant, "GLSL.std.450")]]
T determinant(NBL_CONST_REF_ARG(matrix<T, N, N>) mat);

template<typename T, int N>
[[vk::ext_instruction(GLSLstd450MatrixInverse, "GLSL.std.450")]]
matrix<T, N, N> matrixInverse(NBL_CONST_REF_ARG(matrix<T, N, N>) mat);

[[vk::ext_instruction(GLSLstd450UnpackSnorm2x16, "GLSL.std.450")]]
float32_t2 unpackSnorm2x16(uint32_t p);

[[vk::ext_instruction(GLSLstd450UnpackSnorm4x8, "GLSL.std.450")]]
float32_t4 unpackSnorm4x8(uint32_t p);

[[vk::ext_instruction(GLSLstd450UnpackUnorm4x8, "GLSL.std.450")]]
float32_t4 unpackUnorm4x8(uint32_t p);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450Length, "GLSL.std.450")]]
typename vector_traits<T>::scalar_type length(T vec);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450Normalize, "GLSL.std.450")]]
T normalize(T vec);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450FClamp, "GLSL.std.450")]]
T fClamp(T val, T _min, T _max);
template<typename T NBL_FUNC_REQUIRES(concepts::UnsignedIntegralVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450UClamp, "GLSL.std.450")]]
T uClamp(T val, T _min, T _max);
template<typename T NBL_FUNC_REQUIRES(concepts::IntegralVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450SClamp, "GLSL.std.450")]]
T sClamp(T val, T _min, T _max);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450FMin, "GLSL.std.450")]]
T fMin(T val);
template<typename T NBL_FUNC_REQUIRES(concepts::UnsignedIntegralVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450UMin, "GLSL.std.450")]]
T uMin(T val);
template<typename T NBL_FUNC_REQUIRES(concepts::IntegralVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450SMin, "GLSL.std.450")]]
T sMin(T val);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450FMax, "GLSL.std.450")]]
T fMax(T val);
template<typename T NBL_FUNC_REQUIRES(concepts::UnsignedIntegralVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450UMax, "GLSL.std.450")]]
T uMax(T val);
template<typename T NBL_FUNC_REQUIRES(concepts::IntegralVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450SMax, "GLSL.std.450")]]
T sMax(T val);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450FAbs, "GLSL.std.450")]]
T fAbs(T val);
template<typename T NBL_FUNC_REQUIRES(concepts::IntegralVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450SAbs, "GLSL.std.450")]]
T sAbs(T val);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar32or16BitSize<T>)
[[vk::ext_instruction(GLSLstd450Sin, "GLSL.std.450")]]
T sin(T val);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar32or16BitSize<T>)
[[vk::ext_instruction(GLSLstd450Cos, "GLSL.std.450")]]
T cos(T val);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar32or16BitSize<T>)
[[vk::ext_instruction(GLSLstd450Acos, "GLSL.std.450")]]
T acos(T val);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450Fract, "GLSL.std.450")]]
T fract(T val);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450Round, "GLSL.std.450")]]
T round(T val);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450RoundEven, "GLSL.std.450")]]
T roundEven(T val);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450Trunc, "GLSL.std.450")]]
T trunc(T val);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450Ceil, "GLSL.std.450")]]
T ceil(T val);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450Fma, "GLSL.std.450")]]
T fma(T x, T y, T z);

template<typename T, typename U NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar<T> && 
	(concepts::IntegralVectorOrScalar<U> || concepts::UnsignedIntegralVectorOrScalar<U>) && 
	(vector_traits<T>::Dimension == vector_traits<U>::Dimension))
[[vk::ext_instruction(GLSLstd450Ldexp, "GLSL.std.450")]]
T ldexp(T arg, U exp);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450FSign, "GLSL.std.450")]]
T fSign(T val);

template<typename T NBL_FUNC_REQUIRES(concepts::IntegralVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450SSign, "GLSL.std.450")]]
T sSign(T val);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar32or16BitSize<T>)
[[vk::ext_instruction(GLSLstd450Radians, "GLSL.std.450")]]
T radians(T degrees);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar32or16BitSize<T>)
[[vk::ext_instruction(GLSLstd450Degrees, "GLSL.std.450")]]
T degrees(T radians);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450Step, "GLSL.std.450")]]
T step(T edge, T x);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450SmoothStep, "GLSL.std.450")]]
T smoothStep(T edge0, T edge1, T x);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450FaceForward, "GLSL.std.450")]]
T faceForward(T N, T I, T Nref);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450Reflect, "GLSL.std.450")]]
T reflect(T I, T N);

template<typename T, typename U NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450Refract, "GLSL.std.450")]]
T refract(T I, T N, U Nref);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450ModfStruct, "GLSL.std.450")]]
ModfOutput<T> modfStruct(T val);

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointVectorOrScalar<T>)
[[vk::ext_instruction(GLSLstd450FrexpStruct, "GLSL.std.450")]]
FrexpOutput<T> frexpStruct(T val);

}
}
}

#endif
#endif