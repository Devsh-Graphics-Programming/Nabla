// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_BASIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_BASIC_INCLUDED_

// TODO (PentaKon): All of these have to be ported to spirv intrinsics

// [[vk::ext_extension("GL_KHR_shader_subgroup_basic")]] REVIEW-519: Extensions don't seem to be needed?
void spirv_subgroup_basic_ext(){}

namespace nbl 
{
namespace hlsl
{
namespace subgroup
{
	// REVIEW: No need for #ifdef check because these are always available in HLSL 2021 it seems
	uint Size() {
		return WaveGetLaneCount();
	}
	
	uint SizeLog2() {
		return firstbithigh(Size());
	}
	
	uint InvocationID() {
		return WaveGetLaneIndex();
	}

	uint SubgroupID() {
		// TODO (PentaKon): This is not always correct (subgroup IDs aren't always aligned with invocation index per the spec)
		return gl_LocalInvocationIndex >> SizeLog2();
	}
	
	uint LastSubgroupInvocation() {
		uint lastSubgroupInvocation = Size() - 1u;
		if(SubgroupID() == ((_NBL_HLSL_WORKGROUP_SIZE_ - 1u) >> SizeLog2())) {
			lastSubgroupInvocation &= _NBL_HLSL_WORKGROUP_SIZE_ - 1u; // if the workgroup size is not a power of 2, then the lastSubgroupInvocation for the last subgroup of the workgroup will not be equal to the subgroupMask but something smaller
		}
		return lastSubgroupInvocation;
	}

	bool Elect() {
		return WaveIsFirstLane();
	}

	// REVIEW: Masks don't seem to be available in Wave Ops
	uint64_t EqMask() { // return type for these must be 64-bit
		// REVIEW: I think the spec says that the subgroup size can potentially go up to 128 in which case uint64 won't work properly...
		//  Should we somehow block shaders from running if subgroup size > 64?
		//  Use uint64_t2 ?
		return 0x1u << InvocationID();
	}
	
	uint64_t GeMask() {
		return 0xffffffffu << InvocationID();
	}

	uint64_t GtMask() {
		return GeMask()<<1;
	}

	uint64_t LeMask() {
		return ~GtMask();
	}

	uint64_t LtMask() {
		return ~GeMask();
	}
}
}
}

#endif