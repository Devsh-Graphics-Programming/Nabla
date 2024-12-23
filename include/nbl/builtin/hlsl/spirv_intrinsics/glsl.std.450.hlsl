#ifndef _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_GLSL_STD_450_INCLUDED_
#define _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_GLSL_STD_450_INCLUDED_

#ifdef __HLSL_VERSION
#include <nbl/builtin/hlsl/cpp_compat/basic.h>
#include <nbl/builtin/hlsl/concepts.hlsl>
#include "spirv/unified1/GLSL.std.450.h"

namespace nbl
{
namespace hlsl
{
namespace spirv
{

// Find MSB and LSB restricted to 32-bit width component types https://registry.khronos.org/SPIR-V/specs/unified1/GLSL.std.450.html
template<typename Integral32 NBL_FUNC_REQUIRES(is_same_v<Integral32, int32_t> || is_same_v<Integral32, uint32_t>)
[[vk::ext_instruction(GLSLstd450::GLSLstd450FindILsb, "GLSL.std.450")]]
Integral32 findILsb(Integral32 value);

template<int N>
[[vk::ext_instruction(GLSLstd450::GLSLstd450FindILsb, "GLSL.std.450")]]
vector<int32_t, N> findILsb(vector<int32_t, N> value);

template<int N>
[[vk::ext_instruction(GLSLstd450::GLSLstd450FindILsb, "GLSL.std.450")]]
vector<uint32_t, N> findILsb(vector<uint32_t, N> value);

template<typename Int32_t NBL_FUNC_REQUIRES(is_same_v<Int32_t, int32_t>)
[[vk::ext_instruction(GLSLstd450::GLSLstd450FindSMsb, "GLSL.std.450")]]
int32_t findSMsb(Int32_t value);

template<int N>
[[vk::ext_instruction(GLSLstd450::GLSLstd450FindSMsb, "GLSL.std.450")]]
vector<int32_t, N> findSMsb(vector<int32_t, N> value);

template<typename Uint32_t NBL_FUNC_REQUIRES(is_same_v<Uint32_t, uint32_t>)
[[vk::ext_instruction(GLSLstd450::GLSLstd450FindUMsb, "GLSL.std.450")]]
int32_t findUMsb(Uint32_t value);

template<int N>
[[vk::ext_instruction(GLSLstd450::GLSLstd450FindUMsb, "GLSL.std.450")]]
vector<uint32_t, N> findUMsb(vector<uint32_t, N> value);

template<typename FloatingPoint>
[[vk::ext_instruction(GLSLstd450::GLSLstd450Pow, "GLSL.std.450")]]
enable_if_t<is_floating_point<FloatingPoint>::value && !is_matrix_v<FloatingPoint>, FloatingPoint> pow(FloatingPoint lhs, FloatingPoint rhs);

template<typename FloatingPoint>
[[vk::ext_instruction(GLSLstd450::GLSLstd450Exp, "GLSL.std.450")]]
enable_if_t<is_floating_point<FloatingPoint>::value && !is_matrix_v<FloatingPoint>, FloatingPoint> exp(FloatingPoint val);

template<typename FloatingPoint>
[[vk::ext_instruction(GLSLstd450::GLSLstd450Exp2, "GLSL.std.450")]]
enable_if_t<is_floating_point<FloatingPoint>::value && !is_matrix_v<FloatingPoint>, FloatingPoint> exp2(FloatingPoint val);

template<typename FloatingPoint>
[[vk::ext_instruction(GLSLstd450::GLSLstd450Log, "GLSL.std.450")]]
enable_if_t<is_floating_point<FloatingPoint>::value && !is_matrix_v<FloatingPoint>, FloatingPoint> log(FloatingPoint val);

template<typename FloatingPoint>
[[vk::ext_instruction(GLSLstd450::GLSLstd450InverseSqrt, "GLSL.std.450")]]
enable_if_t<is_floating_point_v<FloatingPoint> && !is_matrix_v<FloatingPoint>, FloatingPoint> inverseSqrt(FloatingPoint val);
 
template<typename FloatingPoint>
[[vk::ext_instruction(GLSLstd450::GLSLstd450Floor, "GLSL.std.450")]]
enable_if_t<is_floating_point_v<FloatingPoint> && !is_matrix_v<FloatingPoint>, FloatingPoint> floor(FloatingPoint val);

template<typename FloatingPoint>
[[vk::ext_instruction(GLSLstd450::GLSLstd450Cross, "GLSL.std.450")]]
enable_if_t<is_floating_point_v<FloatingPoint>, vector<FloatingPoint, 3> > cross(in vector<FloatingPoint, 3> lhs, in vector<FloatingPoint, 3> rhs);

template<typename FloatingPoint>
[[vk::ext_instruction(GLSLstd450::GLSLstd450FMix, "GLSL.std.450")]]
enable_if_t<is_floating_point_v<FloatingPoint> && !is_matrix_v<FloatingPoint>, FloatingPoint> fMix(FloatingPoint x, FloatingPoint y, FloatingPoint a);

template<typename T, int N>
[[vk::ext_instruction(GLSLstd450::GLSLstd450Determinant, "GLSL.std.450")]]
T determinant(in matrix<T, N, N> mat);

template<typename T, int N>
[[vk::ext_instruction(GLSLstd450MatrixInverse, "GLSL.std.450")]]
matrix<T, N, N> matrixInverse(in matrix<T, N, N> mat);

[[vk::ext_instruction(GLSLstd450UnpackSnorm2x16, "GLSL.std.450")]]
float32_t2 unpackSnorm2x16(uint32_t p);

[[vk::ext_instruction(GLSLstd450UnpackSnorm4x8, "GLSL.std.450")]]
float32_t4 unpackSnorm4x8(uint32_t p);

[[vk::ext_instruction(GLSLstd450UnpackUnorm4x8, "GLSL.std.450")]]
float32_t4 unpackUnorm4x8(uint32_t p);

template<typename FloatingPointVector>
[[vk::ext_instruction(GLSLstd450Length, "GLSL.std.450")]]
enable_if_t<is_floating_point_v<FloatingPointVector>&& is_vector_v<FloatingPointVector>, FloatingPointVector> length(FloatingPointVector vec);

template<typename FloatingPointVector>
[[vk::ext_instruction(GLSLstd450Normalize, "GLSL.std.450")]]
enable_if_t<is_floating_point_v<FloatingPointVector> && is_vector_v<FloatingPointVector>, FloatingPointVector> normalize(FloatingPointVector vec);

}
}
}

#endif
#endif