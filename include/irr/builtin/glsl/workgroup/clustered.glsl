#ifndef _IRR_BUILTIN_GLSL_WORKGROUP_CLUSTERED_INCLUDED_
#define _IRR_BUILTIN_GLSL_WORKGROUP_CLUSTERED_INCLUDED_



#include <irr/builtin/glsl/workgroup/shared_clustered.glsl>


#ifdef _IRR_GLSL_SCRATCH_SHARED_DEFINED_
	#if IRR_GLSL_EVAL(_IRR_GLSL_SCRATCH_SHARED_SIZE_DEFINED_)<IRR_GLSL_EVAL(_IRR_GLSL_WORKGROUP_CLUSTERED_SHARED_SIZE_NEEDED_)
		#error "Not enough shared memory declared"
	#endif
#else
	#define _IRR_GLSL_SCRATCH_SHARED_DEFINED_ irr_glsl_workgroupClusteredScratchShared
	#define _IRR_GLSL_SCRATCH_SHARED_SIZE_DEFINED_ _IRR_GLSL_WORKGROUP_CLUSTERED_SHARED_SIZE_NEEDED_
	shared uint _IRR_GLSL_SCRATCH_SHARED_DEFINED_[_IRR_GLSL_WORKGROUP_CLUSTERED_SHARED_SIZE_NEEDED_];
#endif



#include <irr/builtin/glsl/workgroup/ballot.glsl>


/** TODO: @Hazardu or @Przemog or lets have it as a recruitment task
// `clusterSize` needs to be Power of Two, but the workgroup size does not!
// use `irr_glsl_subgroupOp` to implement optimally

float irr_glsl_workgroupClusteredAnd(in float val, in uint clusterSize);
uint irr_glsl_workgroupClusteredAnd(in uint val, in uint clusterSize);
int irr_glsl_workgroupClusteredAnd(in int val, in uint clusterSize);

bool irr_glsl_workgroupClusteredXor(in bool val, in uint clusterSize);
float irr_glsl_workgroupClusteredXor(in float val, in uint clusterSize);
uint irr_glsl_workgroupClusteredXor(in uint val, in uint clusterSize);
int irr_glsl_workgroupClusteredXor(in int val, in uint clusterSize);

bool irr_glsl_workgroupClusteredOr(in bool val, in uint clusterSize);
float irr_glsl_workgroupClusteredOr(in float val, in uint clusterSize);
uint irr_glsl_workgroupClusteredOr(in uint val, in uint clusterSize);
int irr_glsl_workgroupClusteredOr(in int val, in uint clusterSize);

bool irr_glsl_workgroupClusteredAdd(in bool val, in uint clusterSize);
float irr_glsl_workgroupClusteredAdd(in float val, in uint clusterSize);
uint irr_glsl_workgroupClusteredAdd(in uint val, in uint clusterSize);
int irr_glsl_workgroupClusteredAdd(in int val, in uint clusterSize);

// mul and min/max dont need boolean variants, since they're achievable with And and Or
float irr_glsl_workgroupClusteredMul(in float val, in uint clusterSize);
uint irr_glsl_workgroupClusteredMul(in uint val, in uint clusterSize);
int irr_glsl_workgroupClusteredMul(in int val, in uint clusterSize);

float irr_glsl_workgroupClusteredMin(in float val, in uint clusterSize);
uint irr_glsl_workgroupClusteredMin(in uint val, in uint clusterSize);
int irr_glsl_workgroupClusteredMin(in int val, in uint clusterSize);

float irr_glsl_workgroupClusteredMax(in float val, in uint clusterSize);
uint irr_glsl_workgroupClusteredMax(in uint val, in uint clusterSize);
int irr_glsl_workgroupClusteredMax(in int val, in uint clusterSize);
*/


#endif
