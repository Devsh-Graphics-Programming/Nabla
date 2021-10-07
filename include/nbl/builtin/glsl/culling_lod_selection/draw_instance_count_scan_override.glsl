#ifndef _NBL_GLSL_CULLING_LOD_SELECTION_DRAW_INSTANCE_COUNT_SCAN_OVERRIDE_GLSL_INCLUDED_
#define _NBL_GLSL_CULLING_LOD_SELECTION_DRAW_INSTANCE_COUNT_SCAN_OVERRIDE_GLSL_INCLUDED_


#define _NBL_GLSL_SCAN_STORAGE_TYPE_ uint
#include <nbl/builtin/glsl/scan/declarations.glsl>

// disable stuff we dont use
#define NBL_GLSL_CULLING_LOD_SELECTION_DISPATCH_INDIRECT_DESCRIPTOR_BINDING
#define NBL_GLSL_CULLING_LOD_SELECTION_INSTANCE_LIST_DESCRIPTOR_BINDING
#define NBL_GLSL_CULLING_LOD_SELECTION_INSTANCE_LOD_INFO_UVEC4_OFFSETS_DESCRIPTOR_BINDING
#define NBL_GLSL_CULLING_LOD_SELECTION_INSTANCE_DRAWCALL_INCLUSIVE_COUNTS_DESCRIPTOR_BINDING
#define NBL_GLSL_CULLING_LOD_SELECTION_PVS_INSTANCE_DRAWS_DESCRIPTOR_BINDING
#define NBL_GLSL_CULLING_LOD_SELECTION_DRAW_COUNTS_TO_SCAN_DESCRIPTOR_BINDING
#include <nbl/builtin/glsl/culling_lod_selection/input_descriptor_set.glsl>

#define NBL_GLSL_CULLING_LOD_SELECTION_DRAW_CALLS_DESCRIPTOR_QUALIFIERS restrict
#define NBL_GLSL_CULLING_LOD_SELECTION_PER_VIEW_PER_INSTANCE_DESCRIPTOR_BINDING
#define NBL_GLSL_CULLING_LOD_SELECTION_PER_INSTANCE_REDIRECT_ATTRS_DESCRIPTOR_BINDING
#define NBL_GLSL_CULLING_LOD_SELECTION_DRAW_CALL_COUNTS_DESCRIPTOR_BINDING
#include <nbl/builtin/glsl/culling_lod_selection/output_descriptor_modifiers.glsl>

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
		{
			offset--;
			if (notFirstInvocationInGroup)
				data += nbl_glsl_culling_lod_selection_drawCallGetInstanceCount(drawcallsToScan.dwordOffsets[offset]);
		}
	}
	else
	{
		if (notFirstOrLastLevel)
			data = scanScratch.data[offset];
		else
			data = nbl_glsl_culling_lod_selection_drawCallGetInstanceCount(drawcallsToScan.dwordOffsets[offset]);
	}
}

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
		{
			const uint drawcallDWORDOffset = drawcallsToScan.dwordOffsets[levelInvocationIndex];
			if (bool(drawcallDWORDOffset&0x80000000u))
				nbl_glsl_culling_lod_selection_drawArraysSetBaseInstance(drawcallDWORDOffset&0x7fffffffu,data);
			else
				nbl_glsl_culling_lod_selection_drawElementsSetBaseInstance(drawcallDWORDOffset,data);
		}
	}
}


#endif