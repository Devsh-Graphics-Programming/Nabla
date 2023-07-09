// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_BASIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_BASIC_INCLUDED_

//! all functions must be called in uniform control flow (all workgroup invocations active)
namespace nbl
{
namespace hlsl
{
namespace workgroup
{
	static const uint MaxWorkgroupSizeLog2 = 11;
	static const uint MaxWorkgroupSize = 0x1u << MaxWorkgroupSizeLog2;
	
	bool Elect()
	{
		return gl_LocalInvocationIndex==0u;
	}
	
	void Barrier() {
		GroupMemoryBarrierWithGroupSync();
	}
	
	[[vk::ext_instruction(/* OpMemoryBarrier */ 225)]] // https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpControlBarrier
	void spirv_subgroupMemoryBarrierShared(uint memoryScope, uint memorySemantics);

	void MemoryBarrierShared() {
		spirv_subgroupMemoryBarrierShared(2, 0x8 | 0x100); // REVIEW: Need advice on memory semantics
	}
}
}
}
#endif
