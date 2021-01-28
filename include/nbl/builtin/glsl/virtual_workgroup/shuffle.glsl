#ifndef _NBL_BUILTIN_GLSL_VIRTUAL_WORKGROUP_SHUFFLE_INCLUDED_
#define _NBL_BUILTIN_GLSL_VIRTUAL_WORKGROUP_SHUFFLE_INCLUDED_



#include <nbl/builtin/glsl/workgroup/shared_shuffle.glsl>
#define _NBL_GLSL_VIRTUAL_WORKGROUP_SHUFFLE_SHARED_SIZE_NEEDED_  _NBL_GLSL_WORKGROUP_SHUFFLE_SHARED_SIZE_NEEDED_

#ifdef _NBL_GLSL_SCRATCH_SHARED_DEFINED_
	#if NBL_GLSL_EVAL(_NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_)<NBL_GLSL_EVAL(_NBL_GLSL_VIRTUAL_WORKGROUP_SHUFFLE_SHARED_SIZE_NEEDED_)
		#error "Not enough shared memory declared"
	#endif
#else
	#if _NBL_GLSL_VIRTUAL_WORKGROUP_SHUFFLE_SHARED_SIZE_NEEDED_>0
		#define _NBL_GLSL_SCRATCH_SHARED_DEFINED_ nbl_glsl_virtualWorkgroupShuffleScratchShared
		#define _NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_ _NBL_GLSL_VIRTUAL_WORKGROUP_SHUFFLE_SHARED_SIZE_NEEDED_
		shared uint _NBL_GLSL_SCRATCH_SHARED_DEFINED_[_NBL_GLSL_VIRTUAL_WORKGROUP_SHUFFLE_SHARED_SIZE_NEEDED_];
	#endif
#endif

// Shuffle

// uint
uint nbl_glsl_virtualWorkgroupShuffle_noBarriers(in uint index, in uint val, in uint id)
{
	_NBL_GLSL_SCRATCH_SHARED_DEFINED_[index] = val;
	barrier();
	memoryBarrierShared();
	return _NBL_GLSL_SCRATCH_SHARED_DEFINED_[id];
}

uint nbl_glsl_virtualWorkgroupShuffle(in uint index, in uint val, in uint id)
{
	barrier();
	memoryBarrierShared();
	const uint retval = nbl_glsl_virtualWorkgroupShuffle_noBarriers(index, val, id);
	barrier();
	memoryBarrierShared();
	return retval;
}

// float
float nbl_glsl_virtualWorkgroupShuffle_noBarriers(in uint index, in float val, in uint id) 
{
	uint ret = nbl_glsl_virtualWorkgroupShuffle_noBarriers(index, floatBitsToUint(val), id);
	return uintBitsToFloat(ret);
}

float nbl_glsl_virtualWorkgroupShuffle(in uint index, in float val, in uint id) 
{
	uint ret = nbl_glsl_virtualWorkgroupShuffle(index, floatBitsToUint(val), id);
	return uintBitsToFloat(ret);
}

// int
int nbl_glsl_virtualWorkgroupShuffle_noBarriers(in uint index, in int val, in uint id) 
{
	uint ret = nbl_glsl_virtualWorkgroupShuffle_noBarriers(index, uint(val), id);
	return int(ret);
}

int nbl_glsl_virtualWorkgroupShuffle(in uint index, in int val, in uint id) 
{
	uint ret = nbl_glsl_virtualWorkgroupShuffle(index, uint(val), id);
	return int(ret);
}

// Shuffle XOR 

// uint
uint nbl_glsl_virtualWorkgroupShuffleXor_noBarriers(in uint index, in uint val, in uint mask)
{
	uint xor = index ^ mask;
	return nbl_glsl_virtualWorkgroupShuffle_noBarriers(index, val, xor);
}

uint nbl_glsl_virtualWorkgroupShuffleXor(in uint index, in uint val, in uint mask)
{
	uint xor = index ^ mask;
	return nbl_glsl_virtualWorkgroupShuffle(index, val, xor);
}

// float
float nbl_glsl_virtualWorkgroupShuffleXor_noBarriers(in uint index, in float val, in uint mask) {
	uint xor = index ^ mask;
	return nbl_glsl_virtualWorkgroupShuffle_noBarriers(index, val, xor);
}

float nbl_glsl_virtualWorkgroupShuffleXor(in uint index, in float val, in uint mask) 
{
	uint xor = index ^ mask;
	return nbl_glsl_virtualWorkgroupShuffle(index, val, xor);
}

// int
int nbl_glsl_virtualWorkgroupShuffleXor_noBarriers(in uint index, in int val, in uint mask)
{
	uint xor = index ^ mask;
	return nbl_glsl_virtualWorkgroupShuffle_noBarriers(index, val, xor);
}

int nbl_glsl_virtualWorkgroupShuffleXor(in uint index, in int val, in uint mask) 
{
	uint xor = index ^ mask;
	return nbl_glsl_virtualWorkgroupShuffle(index, val, xor);
}

/** TODO @Hazardu or @Przemog you can express all of them in terms of the uint variants to safe yourself the trouble of repeated code, this could also be a recruitment task.

bool nbl_glsl_virtualWorkgroupShuffle(in bool val, in uint id);
bool nbl_glsl_virtualWorkgroupShuffleXor(in bool val, in uint mask);

BONUS: Optimize nbl_glsl_virtualWorkgroupShuffleXor, and implement it with `subgroupShuffleXor` without a workgroup barrier for `mask<gl_SubgroupSize` if extension is available
*/

#endif
