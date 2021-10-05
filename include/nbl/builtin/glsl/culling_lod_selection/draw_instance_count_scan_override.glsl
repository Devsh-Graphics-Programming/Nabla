#ifndef _NBL_GLSL_CULLING_LOD_SELECTION_DRAW_INSTANCE_COUNT_SCAN_OVERRIDE_GLSL_INCLUDED_
#define _NBL_GLSL_CULLING_LOD_SELECTION_DRAW_INSTANCE_COUNT_SCAN_OVERRIDE_GLSL_INCLUDED_


#define _NBL_GLSL_SCAN_STORAGE_TYPE_ uint
#include <nbl/builtin/glsl/scan/declarations.glsl>

// disable stuff we dont use
#define NBL_GLSL_CULLING_LOD_SELECTION_DISPATCH_INDIRECT_DESCRIPTOR_BINDING
#define NBL_GLSL_CULLING_LOD_SELECTION_INSTANCE_LIST_DESCRIPTOR_BINDING
#define NBL_GLSL_CULLING_LOD_SELECTION_INSTANCE_DRAWCALL_OFFSETS_DESCRIPTOR_BINDING
#define NBL_GLSL_CULLING_LOD_SELECTION_INSTANCE_DRAWCALL_EXCLUSIVE_COUNTS_DESCRIPTOR_BINDING
#define NBL_GLSL_CULLING_LOD_SELECTION_PVS_INSTANCE_DRAWS_DESCRIPTOR_BINDING
#define NBL_GLSL_CULLING_LOD_SELECTION_DRAW_COUNTS_TO_SCAN_DESCRIPTOR_BINDING
#include <nbl/builtin/glsl/culling_lod_selection/input_descriptor_set.glsl>

#define NBL_GLSL_CULLING_LOD_SELECTION_DRAW_CALLS_DESCRIPTOR_QUALIFIERS restrict coherent
#define NBL_GLSL_CULLING_LOD_SELECTION_PER_VIEW_PER_INSTANCE_DESCRIPTOR_BINDING
#define NBL_GLSL_CULLING_LOD_SELECTION_PER_INSTANCE_REDIRECT_ATTRS_DESCRIPTOR_BINDING
#define NBL_GLSL_CULLING_LOD_SELECTION_DRAW_CALL_COUNTS_DESCRIPTOR_BINDING
#include <nbl/builtin/glsl/culling_lod_selection/output_descriptor_modifiers.glsl>

uint nbl_glsl_scan_getPaddedData(
	in uint levelInvocationIndex,
	in uint localWorkgroupIndex,
	in uint treeLevel,
	in bool inRange,
	in uint identity
)
{
	const nbl_glsl_scan_Parameters_t params = nbl_glsl_scan_getParameters();
	uint data = identity;
	if (inRange)
	{
		uint offset = levelInvocationIndex;
		if (treeLevel>params.topLevel)
		{
			const uint lastLevel = params.topLevel<<1u;
			const bool notFirstInvocationInGroup = gl_LocalInvocationIndex!=0u;
			const uint pseudoLevel = lastLevel-treeLevel;
			if (bool(localWorkgroupIndex) && gl_LocalInvocationIndex==0u)
				data = scanScratch.data[localWorkgroupIndex+params.temporaryStorageOffset[pseudoLevel]];

			if (treeLevel!=lastLevel)
			{
				offset += params.temporaryStorageOffset[pseudoLevel-1u];
				if (notFirstInvocationInGroup)
					data = scanScratch.data[offset-1u];
			}
			else
			{
#				if _NBL_GLSL_SCAN_TYPE_==_NBL_GLSL_SCAN_TYPE_EXCLUSIVE_
				offset--;
				if (notFirstInvocationInGroup)
#				endif
					data += nbl_glsl_culling_lod_selection_drawCallGetInstanceCount(drawcallsToScan.dwordOffsets[offset]);
			}
		}
		else
		{
			if (bool(treeLevel))
			{
				offset += params.temporaryStorageOffset[treeLevel-1u];
				data = scanScratch.data[offset];
			}
			else
				data = nbl_glsl_culling_lod_selection_drawCallGetInstanceCount(drawcallsToScan.dwordOffsets[offset]);
		}
	}
	return data;
}

void nbl_glsl_scan_setData(
	in uint data,
	in uint levelInvocationIndex,
	in uint localWorkgroupIndex,
	in uint treeLevel,
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
		const uint lastLevel = params.topLevel<<1u;
		if (treeLevel!=lastLevel)
		{
			uint pseudoLevel;
			if (treeLevel!=params.topLevel)
				pseudoLevel = lastLevel-treeLevel;
			else
				pseudoLevel = treeLevel;
			const uint offset = params.temporaryStorageOffset[pseudoLevel-1u];
			scanScratch.data[levelInvocationIndex+offset] = data;
		}
		else
		{
			const uint drawcallDWORDOffset = drawcallsToScan.dwordOffsets[levelInvocationIndex];
			if (bool(drawcallDWORDOffset&0x80000000u))
				nbl_glsl_culling_lod_selection_drawArraysSetBaseInstance(drawcallDWORDOffset,data);
			else
				nbl_glsl_culling_lod_selection_drawElementsSetBaseInstance(drawcallDWORDOffset,data);
		}
	}
}


#endif