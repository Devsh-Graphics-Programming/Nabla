// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_FRAGMENT_SHADER_PIXEL_INTERLOCK_INCLUDED_
#define _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_FRAGMENT_SHADER_PIXEL_INTERLOCK_INCLUDED_

#include "spirv/unified1/spirv.hpp"

namespace nbl 
{
namespace hlsl
{
namespace spirv
{

namespace execution_mode
{
	void PixelInterlockOrderedEXT()
	{
		vk::ext_execution_mode(spv::ExecutionModePixelInterlockOrderedEXT);
	}
}

[[vk::ext_capability(spv::CapabilityFragmentShaderPixelInterlockEXT)]]
[[vk::ext_extension("SPV_EXT_fragment_shader_interlock")]]
[[vk::ext_instruction(spv::OpBeginInvocationInterlockEXT)]]
void beginInvocationInterlockEXT();

[[vk::ext_capability(spv::CapabilityFragmentShaderPixelInterlockEXT)]]
[[vk::ext_extension("SPV_EXT_fragment_shader_interlock")]]
[[vk::ext_instruction(spv::OpEndInvocationInterlockEXT)]]
void endInvocationInterlockEXT();

}
}
}

#endif
