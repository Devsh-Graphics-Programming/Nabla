#ifndef _NBL_GLSL_EXT_SCAN_INCLUDED_

#include <nbl/builtin/glsl/workgroup/arithmetic.glsl>

#define STRIDED_IDX(i) (((i) + 1)*pc.stride-1)

#ifndef _NBL_GLSL_EXT_SCAN_SET_DATA_DECLARED_

void nbl_glsl_ext_Scan_setData(in uint idx, in uint val);

#define _NBL_GLSL_EXT_SCAN_SET_DATA_DECLARED_
#endif

//
// Upsweep
//

uint ScanExclusive(in uint val)
{
	switch (pc.scan_op)
	{
	case (1 << 0):
		return nbl_glsl_workgroupExclusiveAnd(val);
	case (1 << 1):
		return nbl_glsl_workgroupExclusiveXor(val);
	case (1 << 2):
		return nbl_glsl_workgroupExclusiveOr(val);
	case (1 << 3):
		return nbl_glsl_workgroupExclusiveAdd(val);
	case (1 << 4):
		return nbl_glsl_workgroupExclusiveMul(val);
	case (1 << 5):
		return nbl_glsl_workgroupExclusiveMin(val);
	case (1 << 6):
		return nbl_glsl_workgroupExclusiveMax(val);
	}
}

uint ScanInclusive(in uint val)
{
	switch (pc.scan_op)
	{
	case (1 << 0):
		return nbl_glsl_workgroupInclusiveAnd(val);
	case (1 << 1):
		return nbl_glsl_workgroupInclusiveXor(val);
	case (1 << 2):
		return nbl_glsl_workgroupInclusiveOr(val);
	case (1 << 3):
		return nbl_glsl_workgroupInclusiveAdd(val);
	case (1 << 4):
		return nbl_glsl_workgroupInclusiveMul(val);
	case (1 << 5):
		return nbl_glsl_workgroupInclusiveMin(val);
	case (1 << 6):
		return nbl_glsl_workgroupInclusiveMax(val);
	}
}

uint nbl_glsl_ext_Scan_upsweep(in uint val)
{
	return (gl_NumWorkGroups.x == 1u) ? ScanExclusive(val) : ScanInclusive(val);
}

//
//	Downsweep
//

shared uint global_offset;

uint ScanOperation(in uint a, in uint b)
{
	switch (pc.scan_op)
	{
	case (1 << 0):
		return a & b;
	case (1 << 1):
		return a ^ b;
	case (1 << 2):
		return a | b;
	case (1 << 3):
		return a + b;
	case (1 << 4):
		return a * b;
	case (1 << 5):
		return min(a, b);
	case (1 << 6):
		return max(a, b);
	}
}

uint nbl_glsl_ext_Scan_downsweep(in uint idx)
{
	if (gl_LocalInvocationIndex == (_NBL_GLSL_WORKGROUP_SIZE_ - 1u))
		global_offset = inout_values[idx];
	barrier();

	uint data = global_offset;
	if (gl_LocalInvocationIndex != 0u && (gl_GlobalInvocationID.x < pc.element_count_pass))
	{
		uint prev_idx = STRIDED_IDX(gl_GlobalInvocationID.x - 1u);
		data = ScanOperation(data, inout_values[prev_idx]);
	}
	barrier();

	return data;
}

#define _NBL_GLSL_EXT_SCAN_INCLUDED_
#endif
