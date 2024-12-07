#ifndef _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_GLSL_STD_450_INCLUDED_
#define _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_GLSL_STD_450_INCLUDED_

namespace nbl 
{
namespace hlsl
{
namespace spirv
{

//[[vk::ext_instruction(73 /*FindILsb*/, "GLSL.std.450")]]
//int FindILsb(int32_t val);

[[vk::ext_instruction(74 /*FindUMsb*/, "GLSL.std.450")]]
int FindSMsb(int32_t val);

[[vk::ext_instruction(75 /*FindUMsb*/, "GLSL.std.450")]]
int FindUMsb(uint32_t val);
}
}
}

#endif