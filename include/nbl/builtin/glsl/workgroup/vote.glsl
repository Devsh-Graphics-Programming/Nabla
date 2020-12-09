#ifndef _NBL_BUILTIN_GLSL_WORKGROUP_VOTE_INCLUDED_
#define _NBL_BUILTIN_GLSL_WORKGROUP_VOTE_INCLUDED_



#include <nbl/builtin/glsl/workgroup/shared_vote.glsl>



#ifdef _NBL_GLSL_SCRATCH_SHARED_DEFINED_
	#if NBL_GLSL_EVAL(_NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_)<NBL_GLSL_EVAL(_NBL_GLSL_WORKGROUP_VOTE_SHARED_SIZE_NEEDED_)
		#error "Not enough shared memory declared"
	#endif
#else
	#if _NBL_GLSL_WORKGROUP_VOTE_SHARED_SIZE_NEEDED_>0
		#define _NBL_GLSL_SCRATCH_SHARED_DEFINED_ nbl_glsl_workgroupVoteScratchShared
		#define _NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_ _NBL_GLSL_WORKGROUP_VOTE_SHARED_SIZE_NEEDED_
		shared uint _NBL_GLSL_SCRATCH_SHARED_DEFINED_[_NBL_GLSL_WORKGROUP_VOTE_SHARED_SIZE_NEEDED_];
	#endif
#endif


bool nbl_glsl_workgroupAll_noBarriers(in bool value)
{
	// TODO: Optimization using subgroupAll in an ifdef NBL_GL_something (need to do feature mapping first), probably only first 2 lines need to change
	_NBL_GLSL_SCRATCH_SHARED_DEFINED_[0u] = 1u;
	barrier();
	memoryBarrierShared();
	atomicAnd(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[0u],value ? 1u:0u);
	barrier();

	return _NBL_GLSL_SCRATCH_SHARED_DEFINED_[0u]!=0u;
}
bool nbl_glsl_workgroupAll(in bool value)
{
	barrier();
	memoryBarrierShared();
	const bool retval = nbl_glsl_workgroupAll_noBarriers(value);
	barrier();
	return retval;
}

bool nbl_glsl_workgroupAny_noBarriers(in bool value)
{
	// TODO: Optimization using subgroupAny in an ifdef NBL_GL_something (need to do feature mapping first), probably only first 2 lines need to change
	_NBL_GLSL_SCRATCH_SHARED_DEFINED_[0u] = 0u;
	barrier();
	memoryBarrierShared();
	atomicOr(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[0u],value ? 1u:0u);
	barrier();

	return _NBL_GLSL_SCRATCH_SHARED_DEFINED_[0u]!=0u;
}
bool nbl_glsl_workgroupAny(in bool value)
{
	barrier();
	memoryBarrierShared();
	const bool retval = nbl_glsl_workgroupAny_noBarriers(value);
	barrier();
	return retval;
}

/** TODO @Cyprian or @Anastazluk
bool nbl_glsl_workgroupAllEqual(in bool val);
float nbl_glsl_workgroupAllEqual(in float val);
uint nbl_glsl_workgroupAllEqual(in uint val);
int nbl_glsl_workgroupAllEqual(in int val);
**/


#endif
