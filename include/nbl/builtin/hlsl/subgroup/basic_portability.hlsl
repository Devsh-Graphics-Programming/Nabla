#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_BASIC_PORTABILITY_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_BASIC_PORTABILITY_INCLUDED_

namespace nbl 
{
namespace hlsl
{
	static const uint MaxWorkgroupSizeLog2 = 11;
	static const uint MaxWorkgroupSize = 0x1 << MaxWorkgroupSizeLog2;
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
	uint subgroupSize() {
		return WaveGetLaneCount();
	}
	
	uint subgroupSizeLog2() {
		return firstbithigh(subgroupSize());
	}
	
	uint subgroupInvocationID() {
		return WaveGetLaneIndex();
	}

	// REVIEW: Masks don't seem to be available in Wave Ops
	uint64_t subgroupEqMask() { // return type for these must be 64-bit
		// REVIEW: I think the spec says that the subgroup size can potentially go up to 128 in which case uint64 won't work properly...
		//  Should we somehow block shaders from running if subgroup size > 64?
		//  Use uint64_t2 ?
		return 0x1 << subgroupInvocationID();
	}
	
	uint64_t subgroupGeMask() {
		return 0xffffffff<<subgroupInvocationID();
	}
	
	uint64_t subgroupGtMask() {
		return subgroupGeMask()<<1;
	}
	
	uint64_t subgroupLeMask() {
		return ~subgroupGtMask();
	}
	
	uint64_t subgroupLtMask() {
		return ~subgroupGeMask();
	}
	
	// REVIEW: Should we define the Small/LargeSubgroupXMask and Small/LargeHalfSubgroupSizes?
	
	// WAVE BARRIERS
	
	void subgroupBarrier() {
		
	}
}
}

#endif