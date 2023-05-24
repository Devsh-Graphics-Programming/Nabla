// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_BASIC_PORTABILITY_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_BASIC_PORTABILITY_INCLUDED_

namespace nbl 
{
namespace hlsl
{
	static const uint MaxWorkgroupSizeLog2 = 11;
	static const uint MaxWorkgroupSize = 0x1u << MaxWorkgroupSizeLog2;
	
namespace subgroup
{
	static const uint MinSubgroupSizeLog2 = 2;
	static const uint MinSubgroupSize = 0x1u << MinSubgroupSizeLog2;
	
#ifdef NBL_HLSL_IMPL_GL_NV_shader_thread_group
	static const uint MaxSubgroupSizeLog2 = 5;
#elif defined(NBL_HLSL_IMPL_GL_AMD_gcn_shader)||defined(NBL_HLSL_IMPL_GL_ARB_shader_ballot)
	static const uint MaxSubgroupSizeLog2 = 6;
#else
	static const uint MaxSubgroupSizeLog2 = 7;
#endif
	static const uint MaxSubgroupSize = (0x1u << MaxSubgroupSizeLog2);
	
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

	bool Elect() {
		return WaveIsFirstLane();
	}

	template<typename T>
	T BroadcastFirst(T value)
	{
		return WaveReadLaneFirst(value);
	}

	template<typename T>
	T Broadcast(T value, uint invocationId) {
		return WaveReadLaneAt(value, invocationId);
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

	// REVIEW: Should we define the Small/LargeSubgroupXMask and Small/LargeHalfSubgroupSizes?

	// WAVE BARRIERS

	// REVIEW: Review everything related to subgroup barriers and SPIR-V injection
	// REVIEW: Should we check #ifdef NBL_GL_KHR_shader_subgroup_XYZ before applying spirv_subgroupBarriers? Maybe add GroupBarriers if not defined?

	[[vk::ext_instruction(/* OpControlBarrier */ 224)]] // https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpControlBarrier
	void spirv_subgroupBarrier(uint executionScope, uint memoryScope, uint memorySemantics);

	// REVIEW Should we name the Barriers with the Subgroup prefix just to make it clearer when calling?
	// REVIEW Proper Memory Semantics!! Link here: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#Memory_Semantics_-id-
	// REVIEW: Need advice on memory semantics. Would think SubgroupMemory(0x80) | AcquireRelease(0x8) is the correct bitmask but SubgroupMemory doesn't seem to be supported as  Vulkan storage class
	
	void Barrier() {
		// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#_scope_id
		// Subgroup scope is number 3

		// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#_memory_semantics_id
		// By providing memory semantics None we do both control and memory barrier as is done in GLSL
		spirv_subgroupBarrier(3, 3, 0x8 | 0x100);
	}

	[[vk::ext_instruction(/* OpMemoryBarrier */ 225)]] // https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpControlBarrier
	void spirv_subgroupMemoryBarrierShared(uint memoryScope, uint memorySemantics);

	void MemoryBarrierShared() {
		spirv_subgroupMemoryBarrierShared(3, 0x8 | 0x100);
	}
}
}
}

#endif