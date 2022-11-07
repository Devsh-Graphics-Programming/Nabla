#ifndef _NBL_BUILTIN_GLSL_SUBGROUP_BASIC_PORTABILITY_INCLUDED_
#define _NBL_BUILTIN_GLSL_SUBGROUP_BASIC_PORTABILITY_INCLUDED_



#extension GL_KHR_shader_subgroup_basic : enable



#include <nbl/builtin/glsl/macros.glsl>


#define nbl_glsl_MaxWorkgroupSizeLog2 11
#define nbl_glsl_MaxWorkgroupSize (0x1<<nbl_glsl_MaxWorkgroupSizeLog2)


#define nbl_glsl_MinSubgroupSizeLog2 2
#define nbl_glsl_MinSubgroupSize (0x1<<nbl_glsl_MinSubgroupSizeLog2)

#ifdef NBL_GLSL_IMPL_GL_NV_shader_thread_group
	#define nbl_glsl_MaxSubgroupSizeLog2 5
#elif defined(NBL_GLSL_IMPL_GL_AMD_gcn_shader)||defined(NBL_GLSL_IMPL_GL_ARB_shader_ballot)
	#define nbl_glsl_MaxSubgroupSizeLog2 6
#else
	#define nbl_glsl_MaxSubgroupSizeLog2 7
#endif
#define nbl_glsl_MaxSubgroupSize (0x1<<nbl_glsl_MaxSubgroupSizeLog2)


#ifdef NBL_GL_KHR_shader_subgroup_basic_subgroup_size
	#define nbl_glsl_SubgroupSize gl_SubgroupSize
	#define nbl_glsl_SubgroupSizeLog2 findMSB(gl_SubgroupSize)
#else
	#define NBL_GLSL_SUBGROUP_SIZE_IS_CONSTEXPR
	#define nbl_glsl_SubgroupSizeLog2 2
	#define nbl_glsl_SubgroupSize (0x1<<nbl_glsl_SubgroupSizeLog2)

	#if NBL_GLSL_EVAL(nbl_glsl_SubgroupSizeLog2)<NBL_GLSL_EVAL(nbl_glsl_MinSubgroupSizeLog2)
		#error "Something went very wrong when figuring out nbl_glsl_SubgroupSize!"
	#endif
#endif


// TODO: detect compute shader stage, provide nbl_glsl_MaxNumSubgroups,nbl_glsl_MinNumSubgroups,nbl_glsl_NumSubgroups and nbl_glsl_LargeSubgroupID,nbl_glsl_SmallSubgroupID,nbl_glsl_SubgroupID
#ifdef NBL_GL_KHR_shader_subgroup_basic_subgroup_invocation_id
	#define nbl_glsl_SubgroupInvocationID gl_SubgroupInvocationID
    
	#define nbl_glsl_SubgroupEqMask gl_SubgroupEqMask
	#define nbl_glsl_SubgroupGeMask gl_SubgroupGeMask
	#define nbl_glsl_SubgroupGtMask gl_SubgroupGtMask
	#define nbl_glsl_SubgroupLeMask gl_SubgroupLeMask
	#define nbl_glsl_SubgroupLtMask gl_SubgroupLtMask
#else
	#define nbl_glsl_SubgroupInvocationID (gl_LocalInvocationIndex&(nbl_glsl_SubgroupSize-1))

	#define nbl_glsl_SubgroupEqMask (0x1<<nbl_glsl_SubgroupInvocationID)
	#define nbl_glsl_SubgroupGeMask (0xffffffff<<nbl_glsl_SubgroupInvocationID)
	#define nbl_glsl_SubgroupGtMask (nbl_glsl_SubgroupGeMask<<1)
	#define nbl_glsl_SubgroupLeMask (~nbl_glsl_SubgroupGtMask)
	#define nbl_glsl_SubgroupLtMask (~nbl_glsl_SubgroupGeMask)
#endif


#ifdef NBL_GL_KHR_shader_subgroup_basic_subgroup_size
	#define nbl_glsl_LargeSubgroupInvocationID gl_SubgroupInvocationID
	#define nbl_glsl_SmallSubgroupInvocationID gl_SubgroupInvocationID

	#define nbl_glsl_LargeSubgroupEqMask gl_SubgroupEqMask
	#define nbl_glsl_LargeSubgroupGeMask gl_SubgroupGeMask
	#define nbl_glsl_LargeSubgroupGtMask gl_SubgroupGtMask
	#define nbl_glsl_LargeSubgroupLeMask gl_SubgroupLeMask
	#define nbl_glsl_LargeSubgroupLtMask gl_SubgroupLtMask
	#define nbl_glsl_SmallSubgroupEqMask gl_SubgroupEqMask
	#define nbl_glsl_SmallSubgroupGeMask gl_SubgroupGeMask
	#define nbl_glsl_SmallSubgroupGtMask gl_SubgroupGtMask
	#define nbl_glsl_SmallSubgroupLeMask gl_SubgroupLeMask
	#define nbl_glsl_SmallSubgroupLtMask gl_SubgroupLtMask
#else
	#define nbl_glsl_LargeSubgroupInvocationID (gl_LocalInvocationIndex&(nbl_glsl_MaxSubgroupSize-1))
	#define nbl_glsl_SmallSubgroupInvocationID (gl_LocalInvocationIndex&(nbl_glsl_MinSubgroupSize-1))

	#define nbl_glsl_LargeSubgroupEqMask (0x1<<nbl_glsl_LargeSubgroupInvocationID)
	#define nbl_glsl_LargeSubgroupGeMask (0xffffffff<<nbl_glsl_LargeSubgroupInvocationID)
	#define nbl_glsl_LargeSubgroupGtMask (nbl_glsl_LargeSubgroupGeMask<<1)
	#define nbl_glsl_LargeSubgroupLeMask (~nbl_glsl_LargeSubgroupGtMask)
	#define nbl_glsl_LargeSubgroupLtMask (~nbl_glsl_LargeSubgroupGeMask)
	#define nbl_glsl_SmallSubgroupEqMask (0x1<<nbl_glsl_SmallSubgroupInvocationID)
	#define nbl_glsl_SmallSubgroupGeMask (0xffffffff<<nbl_glsl_SmallSubgroupInvocationID)
	#define nbl_glsl_SmallSubgroupGtMask (nbl_glsl_SmallSubgroupGeMask<<1)
	#define nbl_glsl_SmallSubgroupLeMask (~nbl_glsl_SmallSubgroupGtMask)
	#define nbl_glsl_SmallSubgroupLtMask (~nbl_glsl_SmallSubgroupGeMask)
#endif


#define nbl_glsl_LargeHalfSubgroupSizeLog2 (nbl_glsl_MaxSubgroupSizeLog2-1)
#define nbl_glsl_LargeHalfSubgroupSize (nbl_glsl_MaxSubgroupSize>>1)
#define nbl_glsl_SmallHalfSubgroupSizeLog2 (nbl_glsl_MinSubgroupSizeLog2-1)
#define nbl_glsl_SmallHalfSubgroupSize (nbl_glsl_MinSubgroupSize>>1)
#define nbl_glsl_HalfSubgroupSizeLog2 (nbl_glsl_SubgroupSizeLog2-1)
#define nbl_glsl_HalfSubgroupSize (nbl_glsl_SubgroupSize>>1)


void nbl_glsl_subgroupBarrier()
{
	#ifdef NBL_GL_KHR_shader_subgroup_basic
	subgroupBarrier();
	#endif
}

void nbl_glsl_subgroupMemoryBarrier()
{
	#ifdef NBL_GL_KHR_shader_subgroup_basic
	subgroupMemoryBarrier();
	#else
	memoryBarrier();
	#endif
}

void nbl_glsl_subgroupMemoryBarrierBuffer()
{
	#ifdef NBL_GL_KHR_shader_subgroup_basic
	subgroupMemoryBarrierBuffer();
	#else
	memoryBarrierBuffer();
	#endif
}

void nbl_glsl_subgroupMemoryBarrierShared()
{
	#ifdef NBL_GL_KHR_shader_subgroup_basic
	subgroupMemoryBarrierShared();
	#else
	memoryBarrierShared();
	#endif
}

void nbl_glsl_subgroupMemoryBarrierImage()
{
	#ifdef NBL_GL_KHR_shader_subgroup_basic
	subgroupMemoryBarrierImage();
	#else
	memoryBarrierImage();
	#endif
}

/* 
bool nbl_glsl_largeSubgroupElect()
{
}
bool nbl_glsl_smallSubgroupElect()
{
}
bool nbl_glsl_subgroupElect()
{
	#ifdef NBL_GL_KHR_shader_subgroup_basic_subgroup_elect
	return subgroupElect();
	#else
	// TODO: do a bunch of `atomicXor` on a shared memory address OR do a ballotARB?
	atomicXor(ADDRESS,nbl_glsl_SubgroupEqMask);
	memoryBarrierShared();
	return (ADDRESS&nbl_glsl_SubgroupLeMask)==nbl_glsl_SubgroupEqMask;
	#endif
}
*/

#endif