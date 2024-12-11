#ifndef _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_GLSL_STD_450_INCLUDED_
#define _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_GLSL_STD_450_INCLUDED_

#ifdef __HLSL_VERSION
#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include "spirv/unified1/GLSL.std.450.h"

namespace nbl 
{
namespace hlsl
{
namespace spirv
{

// Find MSB and LSB restricted to 32-bit width component types https://registry.khronos.org/SPIR-V/specs/unified1/GLSL.std.450.html

template<typename Integral>
[[vk::ext_instruction(GLSLstd450::GLSLstd450FindILsb, "GLSL.std.450")]]
enable_if_t<is_integral_v<Integral> && (sizeof(scalar_type_t<Integral>) == 4), Integral> findILsb(Integral value);

template<typename Integral>
[[vk::ext_instruction(GLSLstd450::GLSLstd450FindSMsb, "GLSL.std.450")]]
enable_if_t<is_integral_v<Integral> && (sizeof(scalar_type_t<Integral>) == 4), Integral> findSMsb(Integral value);

template<typename Integral>
[[vk::ext_instruction(GLSLstd450::GLSLstd450FindUMsb, "GLSL.std.450")]]
enable_if_t<is_integral_v<Integral> && (sizeof(scalar_type_t<Integral>) == 4), Integral> findUMsb(Integral value);

template<typename T>
[[vk::ext_instruction(GLSLstd450::GLSLstd450Exp2, "GLSL.std.450")]]
enable_if_t<is_fundamental<T>::value, T> Exp2(T val);

template<typename FloatingPoint>
[[vk::ext_instruction(GLSLstd450::GLSLstd450InverseSqrt, "GLSL.std.450")]]
enable_if_t<is_floating_point_v<FloatingPoint>, FloatingPoint> InverseSqrt(FloatingPoint val);

template<typename FloatingPoint>
[[vk::ext_instruction(GLSLstd450::GLSLstd450Floor, "GLSL.std.450")]]
enable_if_t<is_floating_point_v<FloatingPoint>, FloatingPoint> Floor(FloatingPoint val);

}
}
}

#endif
#endif