// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_BASIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_BASIC_INCLUDED_


//#include "nbl/builtin/hlsl/math/typeless_arithmetic.hlsl"
#include "nbl/builtin/hlsl/subgroup/basic_portability.hlsl"

#ifndef _NBL_GL_LOCAL_INVOCATION_IDX_DECLARED_
#define _NBL_GL_LOCAL_INVOCATION_IDX_DECLARED_
const uint gl_LocalInvocationIndex : SV_GroupIndex;
#endif

//! all functions must be called in uniform control flow (all workgroup invocations active)
namespace nbl
{
namespace hlsl
{
namespace workgroup
{
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
		spirv_subgroupMemoryBarrierShared(2, 0x0); // REVIEW: Need advice on memory semantics
	}
}
}
}
#endif
