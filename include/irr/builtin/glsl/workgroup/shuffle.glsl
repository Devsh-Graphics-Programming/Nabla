#ifndef _IRR_BUILTIN_GLSL_WORKGROUP_SHUFFLE_INCLUDED_
#define _IRR_BUILTIN_GLSL_WORKGROUP_SHUFFLE_INCLUDED_



#include <irr/builtin/glsl/workgroup/shared_shuffle.glsl>


#ifdef _IRR_GLSL_SCRATCH_SHARED_DEFINED_
	#if IRR_GLSL_EVAL(_IRR_GLSL_SCRATCH_SHARED_SIZE_DEFINED_)<IRR_GLSL_EVAL(_IRR_GLSL_WORKGROUP_SHUFFLE_SHARED_SIZE_NEEDED_)
		#error "Not enough shared memory declared"
	#endif
#else
	#if _IRR_GLSL_WORKGROUP_SHUFFLE_SHARED_SIZE_NEEDED_>0
		#define _IRR_GLSL_SCRATCH_SHARED_DEFINED_ irr_glsl_workgroupShuffleScratchShared
		#define _IRR_GLSL_SCRATCH_SHARED_SIZE_DEFINED_ _IRR_GLSL_WORKGROUP_SHUFFLE_SHARED_SIZE_NEEDED_
		shared uint _IRR_GLSL_SCRATCH_SHARED_DEFINED_[_IRR_GLSL_WORKGROUP_SHUFFLE_SHARED_SIZE_NEEDED_];
	#endif
#endif


uint irr_glsl_workgroupShuffle_noBarriers(in uint val, in uint id)
{
	_IRR_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex] = val;
	barrier();
	memoryBarrierShared();
	return _IRR_GLSL_SCRATCH_SHARED_DEFINED_[id];
}
uint irr_glsl_workgroupShuffle(in uint val, in uint id)
{
	barrier();
	memoryBarrierShared();
	const uint retval = irr_glsl_workgroupShuffle_noBarriers(val,id);
	barrier();
	memoryBarrierShared();
	return retval;
}


/** TODO @Hazardu or @Przemog you can express all of them in terms of the uint variants to safe yourself the trouble of repeated code, this could also be a recruitment task.

bool irr_glsl_workgroupShuffle(in bool val, in uint id);
float irr_glsl_workgroupShuffle(in float val, in uint id);
int irr_glsl_workgroupShuffle(in int val, in uint id);

bool irr_glsl_workgroupShuffleXor(in bool val, in uint mask);
float irr_glsl_workgroupShuffleXor(in float val, in uint mask);
uint irr_glsl_workgroupShuffleXor(in uint val, in uint mask);
int irr_glsl_workgroupShuffleXor(in int val, in uint mask);

BONUS: Optimize irr_glsl_workgroupShuffleXor, and implement it with `subgroupShuffleXor` without a workgroup barrier for `mask<gl_SubgroupSize` if extension is available
*/

#endif
