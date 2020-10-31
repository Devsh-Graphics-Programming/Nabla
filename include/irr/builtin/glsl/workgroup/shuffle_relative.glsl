#ifndef _IRR_BUILTIN_GLSL_WORKGROUP_SHUFFLE_RELATIVE_INCLUDED_
#define _IRR_BUILTIN_GLSL_WORKGROUP_SHUFFLE_RELATIVE_INCLUDED_


#include <irr/builtin/glsl/workgroup/basic.glsl>


#define _IRR_GLSL_WORKGROUP_SHUFFLE_RELATIVE_SHARED_SIZE_NEEDED_  _IRR_GLSL_WORKGROUP_SIZE_

#ifdef _IRR_GLSL_SCRATCH_SHARED_DEFINED_
	#if IRR_GLSL_EVAL(_IRR_GLSL_SCRATCH_SHARED_SIZE_DEFINED_)<IRR_GLSL_EVAL(_IRR_GLSL_WORKGROUP_SHUFFLE_RELATIVE_SHARED_SIZE_NEEDED_)
		#error "Not enough shared memory declared"
	#endif
#else
	#if _IRR_GLSL_WORKGROUP_SHUFFLE_RELATIVE_SHARED_SIZE_NEEDED_>0
		#define _IRR_GLSL_SCRATCH_SHARED_DEFINED_ irr_glsl_workgroupShuffleRelativeScratchShared
		shared uint _IRR_GLSL_SCRATCH_SHARED_DEFINED_[_IRR_GLSL_WORKGROUP_SHUFFLE_RELATIVE_SHARED_SIZE_NEEDED_];
	#endif
#endif


/** TODO @Hazardu or @Przemog you can express all of them in terms of the uint variants to safe yourself the trouble of repeated code, this could also be a recruitment task.

bool irr_glsl_workgroupShuffleUp(in bool val, in uint delta);
float irr_glsl_workgroupShuffleUp(in float val, in uint delta);
uint irr_glsl_workgroupShuffleUp(in uint val, in uint delta);
int irr_glsl_workgroupShuffleUp(in int val, in uint delta);

bool irr_glsl_workgroupShuffleDown(in bool val, in uint delta);
float irr_glsl_workgroupShuffleDown(in float val, in uint delta);
uint irr_glsl_workgroupShuffleDown(in uint val, in uint delta);
int irr_glsl_workgroupShuffleDown(in int val, in uint delta);

BONUS: Make fuctions with suffix "Wraparound" which dont return undefined values when `gl_LocalInvocationIndex+/-delta` is outside the valid invocationID range
*/

#endif
