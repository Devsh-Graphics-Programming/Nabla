#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_BASIC_PORTABILITY_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_BASIC_PORTABILITY_INCLUDED_

namespace nbl 
{
namespace hlsl
{
	// USE GLSL OR HLSL TERMINOLOGY? (Workgroup vs Threadgroup, Subgroup vs Wave, Invocation vs Thread)
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
	
	
	
	// WAVE BARRIERS
}
}

#endif