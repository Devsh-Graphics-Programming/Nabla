#ifndef _NBL_GLSL_SCAN_VIRTUAL_WORKGROUP_INCLUDED_
#define _NBL_GLSL_SCAN_VIRTUAL_WORKGROUP_INCLUDED_

#include <nbl/builtin/glsl/limits/numeric.glsl>
#include <nbl/builtin/glsl/math/typeless_arithmetic.glsl>
#include <nbl/builtin/glsl/workgroup/arithmetic.glsl>
#include <nbl/builtin/glsl/scan/declarations.glsl>

// TODO: declare nbl_glsl_scan_virtualWorkgroup{$BinopName}
void nbl_glsl_scan_virtualWorkgroup(in uint treeLevel, in uint localWorkgroupIndex)
{
	const nbl_glsl_scan_Parameters_t params = nbl_glsl_scan_getParameters();
	const uint levelInvocationIndex = localWorkgroupIndex*_NBL_GLSL_WORKGROUP_SIZE_+gl_LocalInvocationIndex;
	const bool lastInvocationInGroup = gl_LocalInvocationIndex==(_NBL_GLSL_WORKGROUP_SIZE_-1);
	
	const uint lastLevel = params.topLevel<<1u;
	const uint pseudoLevel = treeLevel>params.topLevel ? (lastLevel-treeLevel):treeLevel;

	const bool inRange = levelInvocationIndex<=params.lastElement[pseudoLevel];
	
#	ifndef _NBL_GLSL_SCAN_BIN_OP_
#		error "_NBL_GLSL_SCAN_BIN_OP_ must be defined!"
#	elif _NBL_GLSL_SCAN_BIN_OP_==_NBL_GLSL_SCAN_OP_AND_
#		define REDUCTION nbl_glsl_workgroupAnd
#		define EXCLUSIVE nbl_glsl_workgroupExclusiveAnd
#		define INCLUSIVE nbl_glsl_workgroupInclusiveAnd
#		error "Cannot figure out what IDENTITY should be"
#	elif _NBL_GLSL_SCAN_BIN_OP_==_NBL_GLSL_SCAN_OP_XOR_
#		define REDUCTION nbl_glsl_workgroupXor
#		define EXCLUSIVE nbl_glsl_workgroupExclusiveXor
#		define INCLUSIVE nbl_glsl_workgroupInclusiveXor
#		define IDENTITY 0
#	elif _NBL_GLSL_SCAN_BIN_OP_==_NBL_GLSL_SCAN_OP_OR_
#		define REDUCTION nbl_glsl_workgroupOr
#		define EXCLUSIVE nbl_glsl_workgroupExclusiveOr
#		define INCLUSIVE nbl_glsl_workgroupInclusiveOr
#		define IDENTITY 0
#	elif _NBL_GLSL_SCAN_BIN_OP_==_NBL_GLSL_SCAN_OP_ADD_
#		define REDUCTION nbl_glsl_workgroupAdd
#		define EXCLUSIVE nbl_glsl_workgroupExclusiveAdd
#		define INCLUSIVE nbl_glsl_workgroupInclusiveAdd
#		define IDENTITY 0
#	elif _NBL_GLSL_SCAN_BIN_OP_==_NBL_GLSL_SCAN_OP_MUL_
#		define REDUCTION nbl_glsl_workgroupMul
#		define EXCLUSIVE nbl_glsl_workgroupExclusiveMul
#		define INCLUSIVE nbl_glsl_workgroupInclusiveMul
#		define IDENTITY 1
#	elif _NBL_GLSL_SCAN_BIN_OP_==_NBL_GLSL_SCAN_OP_MIN_
#		define REDUCTION nbl_glsl_workgroupMin
#		define EXCLUSIVE nbl_glsl_workgroupExclusiveMin
#		define INCLUSIVE nbl_glsl_workgroupInclusiveMin
#		error "Cannot figure out what IDENTITY should be"
#	elif _NBL_GLSL_SCAN_BIN_OP_==_NBL_GLSL_SCAN_OP_MAX_
#		define REDUCTION nbl_glsl_workgroupMax
#		define EXCLUSIVE nbl_glsl_workgroupExclusiveMax
#		define INCLUSIVE nbl_glsl_workgroupInclusiveMax
#		error "Cannot figure out what IDENTITY should be"
#	else
#		error "_NBL_GLSL_SCAN_BIN_OP_ invalid value!"
#	endif
	nbl_glsl_scan_Storage_t data = IDENTITY;
	if (inRange)
		nbl_glsl_scan_getData(data,levelInvocationIndex,localWorkgroupIndex,treeLevel,pseudoLevel);

	if (treeLevel<params.topLevel)
		data = REDUCTION(data);
	else if (treeLevel==params.topLevel)
		data = EXCLUSIVE(data);
	else
		data = INCLUSIVE(data);
	
	nbl_glsl_scan_setData(data,levelInvocationIndex,localWorkgroupIndex,treeLevel,pseudoLevel,inRange);
#	undef REDUCTION
#	undef EXCLUSIVE
#	undef INCLUSIVE
#	undef IDENTITY
}

#ifndef _NBL_GLSL_SCAN_MAIN_DEFINED_
#include <nbl/builtin/glsl/scan/default_scheduler.glsl>
void nbl_glsl_scan_main(in nbl_glsl_scan_DefaultSchedulerParameters_t schedulerParams)
{
	const uint topLevel = nbl_glsl_scan_getParameters().topLevel;
	// persistent workgroups
	while (true)
	{
		uint treeLevel,localWorkgroupIndex;
		if (nbl_glsl_scan_scheduler_getWork(schedulerParams,topLevel,treeLevel,localWorkgroupIndex))
			return;

		nbl_glsl_scan_virtualWorkgroup(treeLevel,localWorkgroupIndex);

		nbl_glsl_scan_scheduler_markComplete(schedulerParams,topLevel,treeLevel,localWorkgroupIndex);
	}
}
#define _NBL_GLSL_SCAN_MAIN_DEFINED_
#endif


#endif