#ifndef _NBL_BUILTIN_GLSL_SUBGROUP_SHUFFLE_PORTABILITY_INCLUDED_
#define _NBL_BUILTIN_GLSL_SUBGROUP_SHUFFLE_PORTABILITY_INCLUDED_


#include <nbl/builtin/glsl/subgroup/shared_shuffle_portability.glsl>


// TODO: A SPIRV-Cross contribution so we can set NBL_GL_KHR_shader_subgroup_shuffle when AMD_gcn_shader or NVidia shuffle extensions are available 
#ifdef NBL_GL_KHR_shader_subgroup_shuffle


#define nbl_glsl_subgroupShuffle subgroupShuffle

#define nbl_glsl_subgroupShuffleXor subgroupShuffleXor


#else


#ifdef _NBL_GLSL_SCRATCH_SHARED_DEFINED_
	#if NBL_GLSL_LESS(_NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_,_NBL_GLSL_SUBGROUP_SHUFFLE_EMULATION_SHARED_SIZE_NEEDED_)
		#error "Not enough shared memory declared"
	#endif
#else
	#if NBL_GLSL_GREATER(_NBL_GLSL_SUBGROUP_SHUFFLE_EMULATION_SHARED_SIZE_NEEDED_,0)
		#define _NBL_GLSL_SCRATCH_SHARED_DEFINED_ nbl_glsl_subgroupShuffleEmulationScratchShared
		#define _NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_ _NBL_GLSL_SUBGROUP_SHUFFLE_EMULATION_SHARED_SIZE_NEEDED_
		shared uint _NBL_GLSL_SCRATCH_SHARED_DEFINED_[_NBL_GLSL_SUBGROUP_SHUFFLE_EMULATION_SHARED_SIZE_NEEDED_];
	#endif
#endif


uint nbl_glsl_subgroupShuffle(uint value, uint id)
{
	_NBL_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex] = value;
	nbl_glsl_subgroupBarrier();
	nbl_glsl_subgroupMemoryBarrierShared();
	return _NBL_GLSL_SCRATCH_SHARED_DEFINED_[(gl_LocalInvocationIndex&(-nbl_glsl_SubgroupSize))|id];
}
int nbl_glsl_subgroupShuffle(int value, uint id)
{
	return int(nbl_glsl_subgroupShuffle(uint(value),id));
}
float nbl_glsl_subgroupShuffle(float value, uint id)
{
	return uintBitsToFloat(nbl_glsl_subgroupShuffle(floatBitsToUint(value),id));
}

uint nbl_glsl_subgroupShuffleXor(uint value, uint id)
{
	_NBL_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex] = value;
	nbl_glsl_subgroupBarrier();
	nbl_glsl_subgroupMemoryBarrierShared();
	return _NBL_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex^id];
}
int nbl_glsl_subgroupShuffleXor(int value, uint id)
{
	return int(nbl_glsl_subgroupShuffleXor(uint(value),id));
}
float nbl_glsl_subgroupShuffleXor(float value, uint id)
{
	return uintBitsToFloat(nbl_glsl_subgroupShuffleXor(floatBitsToUint(value),id));
}


#endif // NBL_GL_KHR_shader_subgroup_shuffle


#endif
