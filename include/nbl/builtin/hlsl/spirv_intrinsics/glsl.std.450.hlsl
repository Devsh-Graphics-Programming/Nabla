#ifndef _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_GLSL_STD_450_INCLUDED_
#define _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_GLSL_STD_450_INCLUDED_

#ifdef __HLSL_VERSION
#include "spirv/unified1/GLSL.std.450.h"

namespace nbl 
{
namespace hlsl
{
namespace spirv
{

//[[vk::ext_instruction(GLSLstd450FindILsb, "GLSL.std.450")]]
//int findMSB(int32_t val);

[[vk::ext_instruction(GLSLstd450FindUMsb, "GLSL.std.450")]]
int findMSB(int32_t val);

[[vk::ext_instruction(GLSLstd450FindUMsb, "GLSL.std.450")]]
int findMSB(uint32_t val);
}
}
}

#endif
#endif