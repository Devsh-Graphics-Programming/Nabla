#ifndef _NBL_GLSL_SCAN_DESCRIPTORS_INCLUDED_
#define _NBL_GLSL_SCAN_DESCRIPTORS_INCLUDED_


#ifndef _NBL_GLSL_SCAN_DESCRIPTOR_SET_DEFINED_
#define _NBL_GLSL_SCAN_DESCRIPTOR_SET_DEFINED_ 0
#endif

#ifndef _NBL_GLSL_SCAN_INPUT_DESCRIPTOR_DEFINED_
#ifndef _NBL_GLSL_SCAN_INPUT_BINDING_DEFINED_
#define _NBL_GLSL_SCAN_INPUT_BINDING_DEFINED_ 0
#endif
#include <nbl/builtin/glsl/scan/declarations.glsl>
layout(set=_NBL_GLSL_SCAN_DESCRIPTOR_SET_DEFINED_, binding=_NBL_GLSL_SCAN_INPUT_BINDING_DEFINED_, std430) restrict buffer ScanBuffer
{
	nbl_glsl_scan_Storage_t data[];
} scanBuffer;
#define _NBL_GLSL_SCAN_INPUT_DESCRIPTOR_DEFINED_
#endif

#ifndef _NBL_GLSL_SCAN_SCRATCH_DESCRIPTOR_DEFINED_
#ifndef _NBL_GLSL_SCAN_SCRATCH_BINDING_DEFINED_
#define _NBL_GLSL_SCAN_SCRATCH_BINDING_DEFINED_ 1
#endif
layout(set=_NBL_GLSL_SCAN_DESCRIPTOR_SET_DEFINED_, binding=_NBL_GLSL_SCAN_SCRATCH_BINDING_DEFINED_, std430) restrict coherent buffer ScanScratchBuffer
{
	uint workgroupsStarted;
	uint data[];
} scanScratch;
#define _NBL_GLSL_SCAN_SCRATCH_DESCRIPTOR_DEFINED_
#endif


#ifndef _NBL_GLSL_SCAN_GET_PADDED_DATA_DEFINED_
#include <nbl/builtin/glsl/scan/declarations.glsl>
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
#				if _NBL_GLSL_SCAN_TYPE_==_NBL_GLSL_SCAN_TYPE_EXCLUSIVE_
			offset--;
			if (notFirstInvocationInGroup)
#				endif
				data += scanBuffer.data[offset];
		}
	}
	else
	{
		if (notFirstOrLastLevel)
			data = scanScratch.data[offset];
		else
			data = scanBuffer.data[offset];
	}
}
#define _NBL_GLSL_SCAN_GET_PADDED_DATA_DEFINED_
#endif

#ifndef _NBL_GLSL_SCAN_SET_DATA_DEFINED_
#include <nbl/builtin/glsl/scan/declarations.glsl>
void nbl_glsl_scan_setData(
	in nbl_glsl_scan_Storage_t data,
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
		const bool lastInvocationInGroup = gl_LocalInvocationIndex==(_NBL_GLSL_WORKGROUP_SIZE_-1);
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
			scanBuffer.data[levelInvocationIndex] = data;
	}
}
#define _NBL_GLSL_SCAN_SET_DATA_DEFINED_
#endif

#endif