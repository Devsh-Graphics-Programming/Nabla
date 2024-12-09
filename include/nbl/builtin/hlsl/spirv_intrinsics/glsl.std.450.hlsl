#ifndef _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_GLSL_STD_450_INCLUDED_
#define _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_GLSL_STD_450_INCLUDED_

#ifdef __HLSL_VERSION
#include "spirv/unified1/GLSL.std.450.h"
#endif

namespace nbl 
{
namespace hlsl
{
#ifdef __HLSL_VERSION
namespace spirv
{

//[[vk::ext_instruction(GLSLstd450FindILsb, "GLSL.std.450")]]
//int FindILsb(int32_t val);

[[vk::ext_instruction(GLSLstd450FindUMsb, "GLSL.std.450")]]
int FindSMsb(int32_t val);

[[vk::ext_instruction(GLSLstd450FindUMsb, "GLSL.std.450")]]
int FindUMsb(uint32_t val);
}
#endif
}
}

#endif