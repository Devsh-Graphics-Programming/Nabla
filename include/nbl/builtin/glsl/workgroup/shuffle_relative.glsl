#ifndef _NBL_BUILTIN_GLSL_WORKGROUP_SHUFFLE_RELATIVE_INCLUDED_
#define _NBL_BUILTIN_GLSL_WORKGROUP_SHUFFLE_RELATIVE_INCLUDED_


#include <nbl/builtin/glsl/workgroup/shared_shuffle_relative.glsl>


#ifdef _NBL_GLSL_SCRATCH_SHARED_DEFINED_
	#if NBL_GLSL_EVAL(_NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_)<NBL_GLSL_EVAL(_NBL_GLSL_WORKGROUP_SHUFFLE_RELATIVE_SHARED_SIZE_NEEDED_)
		#error "Not enough shared memory declared"
	#endif
#else
	#if _NBL_GLSL_WORKGROUP_SHUFFLE_RELATIVE_SHARED_SIZE_NEEDED_>0
		#define _NBL_GLSL_SCRATCH_SHARED_DEFINED_ nbl_glsl_workgroupShuffleRelativeScratchShared
		#define _NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_ _NBL_GLSL_WORKGROUP_SHUFFLE_RELATIVE_SHARED_SIZE_NEEDED_
		shared uint _NBL_GLSL_SCRATCH_SHARED_DEFINED_[_NBL_GLSL_WORKGROUP_SHUFFLE_RELATIVE_SHARED_SIZE_NEEDED_];
	#endif
#endif


/** TODO @Hazardu or @Przemog you can express all of them in terms of the uint variants to safe yourself the trouble of repeated code, this could also be a recruitment task.

bool nbl_glsl_workgroupShuffleUp(in bool val, in uint delta);
float nbl_glsl_workgroupShuffleUp(in float val, in uint delta);
uint nbl_glsl_workgroupShuffleUp(in uint val, in uint delta);
int nbl_glsl_workgroupShuffleUp(in int val, in uint delta);

bool nbl_glsl_workgroupShuffleDown(in bool val, in uint delta);
float nbl_glsl_workgroupShuffleDown(in float val, in uint delta);
uint nbl_glsl_workgroupShuffleDown(in uint val, in uint delta);
int nbl_glsl_workgroupShuffleDown(in int val, in uint delta);

BONUS: Make fuctions with suffix "Wraparound" which dont return undefined values when `gl_LocalInvocationIndex+/-delta` is outside the valid invocationID range
*/

#endif
