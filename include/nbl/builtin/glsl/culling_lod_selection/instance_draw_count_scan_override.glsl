#ifndef _NBL_GLSL_CULLING_LOD_SELECTION_DRAW_INSTANCE_COUNT_SCAN_OVERRIDE_GLSL_INCLUDED_
#define _NBL_GLSL_CULLING_LOD_SELECTION_DRAW_INSTANCE_COUNT_SCAN_OVERRIDE_GLSL_INCLUDED_


#define _NBL_GLSL_SCAN_STORAGE_TYPE_ uint
#include <nbl/builtin/glsl/scan/declarations.glsl>

// disable stuff we dont use
#define NBL_GLSL_CULLING_LOD_SELECTION_DISPATCH_INDIRECT_DESCRIPTOR_QUALIFIERS restrict writeonly
#define NBL_GLSL_CULLING_LOD_SELECTION_INSTANCE_LIST_DESCRIPTOR_DECLARED
#define NBL_GLSL_CULLING_LOD_SELECTION_INSTANCE_PVS_INSTANCES_DESCRIPTOR_DECLARED
#define NBL_GLSL_CULLING_LOD_SELECTION_INSTANCE_DRAWCALL_INCLUSIVE_COUNTS_DESCRIPTOR_QUALIFIERS restrict
#define NBL_GLSL_CULLING_LOD_SELECTION_PVS_INSTANCE_DRAWS_DESCRIPTOR_DECLARED
#define NBL_GLSL_CULLING_LOD_SELECTION_DRAWCALLS_TO_SCAN_DESCRIPTOR_DECLARED
#define NBL_GLSL_CULLING_LOD_SELECTION_DRAW_COUNTS_TO_SCAN_DESCRIPTOR_DECLARED
#include <nbl/builtin/glsl/culling_lod_selection/input_descriptor_set.glsl>

//
uint nbl_glsl_scan_getIndirectElementCount()
{
	return totalInstanceCountAfterCull;
}

//
void nbl_glsl_scan_getData(
	inout nbl_glsl_scan_Storage_t data,
	in uint levelInvocationIndex,
	in uint localWorkgroupIndex,
	in uint treeLevel,
	in uint pseudoLevel
)
{
	const nbl_glsl_scan_Parameters_t params = nbl_glsl_scan_getParameters();

	uint offset = levelInvocationIndex;
	const bool notFirstOrLastLevel = bool(pseudoLevel);
	if (notFirstOrLastLevel)
		offset += params.temporaryStorageOffset[pseudoLevel-1u];

	// TODO: optimize the branches some more :D
	if (pseudoLevel!=treeLevel) // downsweep
	{
		const bool notFirstInvocationInGroup = gl_LocalInvocationIndex!=0u;
		if (bool(localWorkgroupIndex) && gl_LocalInvocationIndex==0u)
			data = scanScratch.data[localWorkgroupIndex+params.temporaryStorageOffset[pseudoLevel]];

		if (notFirstOrLastLevel)
		{
			if (notFirstInvocationInGroup)
				data = scanScratch.data[offset-1u];
		}
		else
			data += lodDrawcallInclusiveCounts[offset];
	}
	else
	{
		if (notFirstOrLastLevel)
			data = scanScratch.data[offset];
		else
			data = lodDrawcallInclusiveCounts[offset];
	}
}

//
void nbl_glsl_scan_setData(
	in uint data,
	in uint levelInvocationIndex,
	in uint localWorkgroupIndex,
	in uint treeLevel,
	in uint pseudoLevel,
	in bool inRange
)
{
	const nbl_glsl_scan_Parameters_t params = nbl_glsl_scan_getParameters();
	if (treeLevel<params.topLevel)
	{
		const bool lastInvocationInGroup = gl_LocalInvocationIndex==(gl_WorkGroupSize.x-1);
		if (lastInvocationInGroup)
			scanScratch.data[localWorkgroupIndex+params.temporaryStorageOffset[treeLevel]] = data;
	}
	else if (inRange)
	{
		if (bool(pseudoLevel))
		{
			const uint offset = params.temporaryStorageOffset[pseudoLevel-1u];
			scanScratch.data[levelInvocationIndex+offset] = data;
		}
		else
			lodDrawcallInclusiveCounts[levelInvocationIndex] = data;
	}
}

//
void nbl_glsl_scan_main();
void main()
{
	const uint pvsInstanceCount = nbl_glsl_scan_getIndirectElementCount();

	if (gl_GlobalInvocationID.x==0u)
	{
		dispatchIndirect.instanceDrawCull.num_groups_x = nbl_glsl_utils_computeOptimalPersistentWorkgroupDispatchSize(
				max(pvsInstanceCount,1u),
				_NBL_GLSL_CULLING_LOD_SELECTION_CULL_WORKGROUP_SIZE_
		);
	}
	else if (gl_GlobalInvocationID.x==1u)
		dispatchIndirect.instanceRefCountingSortScatter.num_groups_x = 1u;

	if (bool(pvsInstanceCount))
		nbl_glsl_scan_main();
}
#define _NBL_GLSL_MAIN_DEFINED_

#endif