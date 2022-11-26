#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_BASIC_PORTABILITY_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_BASIC_PORTABILITY_INCLUDED_

namespace nbl 
{
namespace hlsl
{
	static const uint MaxWorkgroupSizeLog2 = 11;
	static const uint MaxWorkgroupSize = 0x1 << MaxWorkgroupSizeLog2;
	
namespace subgroup
{
	static const uint MinSubgroupSizeLog2 = 2;
	static const uint MinSubgroupSize = 0x1 << MinSubgroupSizeLog2;
	
#ifdef NBL_HLSL_IMPL_GL_NV_shader_thread_group
	static const uint MaxSubgroupSizeLog2 = 5;
#elif defined(NBL_HLSL_IMPL_GL_AMD_gcn_shader)||defined(NBL_HLSL_IMPL_GL_ARB_shader_ballot)
	static const uint MaxSubgroupSizeLog2 = 6;
#else
	static const uint MaxSubgroupSizeLog2 = 7;
#endif
	static const uint MaxSubgroupSize = (0x1<<MaxSubgroupSizeLog2);
	
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

	// REVIEW: Masks don't seem to be available in Wave Ops
	uint64_t EqMask() { // return type for these must be 64-bit
		// REVIEW: I think the spec says that the subgroup size can potentially go up to 128 in which case uint64 won't work properly...
		//  Should we somehow block shaders from running if subgroup size > 64?
		//  Use uint64_t2 ?
		return 0x1 << InvocationID();
	}
	
	uint64_t GeMask() {
		return 0xffffffff<<InvocationID();
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
	
	[[vk::ext_instruction(/* subgroupBarrier */ 224)]] // https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpControlBarrier
	void spirv_subgroupBarrier(uint executionScope, uint memoryScope, uint memorySemantics);
	
	void Barrier() {
		// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#_scope_id
		// Subgroup scope is number 3
		
		// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#_memory_semantics_id
		// By providing memory semantics None we do both control and memory barrier as is done in GLSL
		spirv_subgroupBarrier(3, 3, 0x0);
	}
	
	[[vk::ext_instruction(/* subgroupMemoryBarrierShared */ 225)]] // https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpControlBarrier
	void spirv_subgroupMemoryBarrierShared(uint memoryScope, uint memorySemantics);
	
	void MemoryBarrierShared() {
		spirv_subgroupMemoryBarrierShared(3, 0x0); // REVIEW: Need advice on memory semantics. Would think SubgroupMemory(0x80) but have no idea
	}
}
}
}

#endif