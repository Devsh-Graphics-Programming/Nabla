#ifndef _NBL_BUILTIN_GLSL_WORKGROUP_SHUFFLE_INCLUDED_
#define _NBL_BUILTIN_GLSL_WORKGROUP_SHUFFLE_INCLUDED_



#include <nbl/builtin/glsl/workgroup/shared_shuffle.glsl>


#ifdef _NBL_GLSL_SCRATCH_SHARED_DEFINED_
	#if NBL_GLSL_EVAL(_NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_)<NBL_GLSL_EVAL(_NBL_GLSL_WORKGROUP_SHUFFLE_SHARED_SIZE_NEEDED_)
		#error "Not enough shared memory declared"
	#endif
#else
	#if _NBL_GLSL_WORKGROUP_SHUFFLE_SHARED_SIZE_NEEDED_>0
		#define _NBL_GLSL_SCRATCH_SHARED_DEFINED_ nbl_glsl_workgroupShuffleScratchShared
		#define _NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_ _NBL_GLSL_WORKGROUP_SHUFFLE_SHARED_SIZE_NEEDED_
		shared uint _NBL_GLSL_SCRATCH_SHARED_DEFINED_[_NBL_GLSL_WORKGROUP_SHUFFLE_SHARED_SIZE_NEEDED_];
	#endif
#endif

// Shuffle

// uint
uint nbl_glsl_workgroupShuffle_noBarriers(in uint val, in uint id)
{
	_NBL_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex] = val;
	barrier();
	memoryBarrierShared();
	return _NBL_GLSL_SCRATCH_SHARED_DEFINED_[id];
}

uint nbl_glsl_workgroupShuffle(in uint val, in uint id)
{
	barrier();
	memoryBarrierShared();
	const uint retval = nbl_glsl_workgroupShuffle_noBarriers(val,id);
	barrier();
	memoryBarrierShared();
	return retval;
}

// float
float nbl_glsl_workgroupShuffle_noBarriers(in float val, in uint id) 
{
	uint ret = nbl_glsl_workgroupShuffle_noBarriers(floatBitsToUint(val), id);
	return uintBitsToFloat(ret);
}

float nbl_glsl_workgroupShuffle(in float val, in uint id) 
{
	uint ret = nbl_glsl_workgroupShuffle(floatBitsToUint(val), id);
	return uintBitsToFloat(ret);
}

// int
int nbl_glsl_workgroupShuffle_noBarriers(in int val, in uint id) 
{
	uint ret = nbl_glsl_workgroupShuffle_noBarriers(uint(val), id);
	return int(ret);
}

int nbl_glsl_workgroupShuffle(in int val, in uint id) 
{
	uint ret = nbl_glsl_workgroupShuffle(uint(val), id);
	return int(ret);
}

// Shuffle XOR 

// uint
uint nbl_glsl_workgroupShuffleXor_noBarriers(in uint val, in uint mask)
{
	uint xor = gl_LocalInvocationIndex ^ mask;
	return nbl_glsl_workgroupShuffle_noBarriers(val, xor);
}

uint nbl_glsl_workgroupShuffleXor(in uint val, in uint mask)
{
	uint xor = gl_LocalInvocationIndex ^ mask;
	return nbl_glsl_workgroupShuffle(val, xor);
}

// float
float nbl_glsl_workgroupShuffleXor_noBarriers(in float val, in uint mask) {
	uint xor = gl_LocalInvocationIndex ^ mask;
	return nbl_glsl_workgroupShuffle_noBarriers(val, xor);
}

float nbl_glsl_workgroupShuffleXor(in float val, in uint mask) 
{
	uint xor = gl_LocalInvocationIndex ^ mask;
	return nbl_glsl_workgroupShuffle(val, xor);
}

// int
int nbl_glsl_workgroupShuffleXor_noBarriers(in int val, in uint mask)
{
	uint xor = gl_LocalInvocationIndex ^ mask;
	return nbl_glsl_workgroupShuffle_noBarriers(val, xor);
}

int nbl_glsl_workgroupShuffleXor(in int val, in uint mask) 
{
	uint xor = gl_LocalInvocationIndex ^ mask;
	return nbl_glsl_workgroupShuffle(val, xor);
}

/** TODO @Hazardu or @Przemog you can express all of them in terms of the uint variants to safe yourself the trouble of repeated code, this could also be a recruitment task.

bool nbl_glsl_workgroupShuffle(in bool val, in uint id);
bool nbl_glsl_workgroupShuffleXor(in bool val, in uint mask);

BONUS: Optimize nbl_glsl_workgroupShuffleXor, and implement it with `subgroupShuffleXor` without a workgroup barrier for `mask<gl_SubgroupSize` if extension is available
*/

#endif
