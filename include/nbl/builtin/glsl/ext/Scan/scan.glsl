#ifndef _NBL_GLSL_EXT_SCAN_INCLUDED_

#include <nbl/builtin/glsl/workgroup/arithmetic.glsl>
#include <nbl/builtin/glsl/math/typeless_arithmetic.glsl>

#define STRIDED_IDX(i) (((i) + 1)*(nbl_glsl_ext_Scan_Parameters_t_getStride())-1)

#ifndef _NBL_GLSL_EXT_SCAN_SET_DATA_DECLARED_

void nbl_glsl_ext_Scan_setData(in uint idx, in uint val);

#define _NBL_GLSL_EXT_SCAN_SET_DATA_DECLARED_
#endif

#ifndef _NBL_GLSL_EXT_SCAN_GET_PADDED_DATA_DECLARED_

uint nbl_glsl_ext_Scan_getPaddedData(in uint idx, in uint pad_val, bool is_upsweep);

#define _NBL_GLSL_EXT_SCAN_GET_PADDED_DATA_DECLARED_
#endif

#ifndef _NBL_GLSL_EXT_SCAN_GET_PARAMETERS_DEFINED_
#error "You need to define `nbl_glsl_ext_Scan_getParameters` and mark `_NBL_GLSL_EXT_SCAN_GET_PARAMETERS_DEFINED_`!"
#endif
#ifndef _NBL_GLSL_EXT_SCAN_SET_DATA_DEFINED_
#error "You need to define `nbl_glsl_ext_Scan_setData` and mark `_NBL_GLSL_EXT_SCAN_SET_DATA_DEFINED_`!"
#endif
#ifndef _NBL_GLSL_EXT_SCAN_GET_PADDED_DATA_DEFINED_
#error "You need to define `nbl_glsl_ext_Scan_getPaddedData` and mark `_NBL_GLSL_EXT_SCAN_GET_PADDED_DATA_DEFINED_`!"
#endif

//
// Upsweep
//

#define NBL_GLSL_EXT_SCAN_DEFINE_UPSWEEP(IDX, IDENTITY, WORKGROUP_OP_EXC, WORKGROUP_OP_INC) uint data = nbl_glsl_ext_Scan_getPaddedData(IDX, IDENTITY, true); \
	return (gl_NumWorkGroups.x == 1u) ? WORKGROUP_OP_EXC(data) : WORKGROUP_OP_INC(data);

uint nbl_glsl_ext_Scan_upsweepAnd(in uint idx)
{
	NBL_GLSL_EXT_SCAN_DEFINE_UPSWEEP(idx, uint(-1), nbl_glsl_workgroupExclusiveAnd, nbl_glsl_workgroupInclusiveAnd)
}

uint nbl_glsl_ext_Scan_upsweepXor(in uint idx)
{
	NBL_GLSL_EXT_SCAN_DEFINE_UPSWEEP(idx, 0u, nbl_glsl_workgroupExclusiveXor, nbl_glsl_workgroupInclusiveXor)
}

uint nbl_glsl_ext_Scan_upsweepOr(in uint idx)
{
	NBL_GLSL_EXT_SCAN_DEFINE_UPSWEEP(idx, 0u, nbl_glsl_workgroupExclusiveOr, nbl_glsl_workgroupInclusiveOr)
}

uint nbl_glsl_ext_Scan_upsweepAdd(in uint idx)
{
	NBL_GLSL_EXT_SCAN_DEFINE_UPSWEEP(idx, 0u, nbl_glsl_workgroupExclusiveAdd, nbl_glsl_workgroupInclusiveAdd)
}

uint nbl_glsl_ext_Scan_upsweepMul(in uint idx)
{
	NBL_GLSL_EXT_SCAN_DEFINE_UPSWEEP(idx, 1u, nbl_glsl_workgroupExclusiveMul, nbl_glsl_workgroupInclusiveMul)
}

uint nbl_glsl_ext_Scan_upsweepMin(in uint idx)
{
	NBL_GLSL_EXT_SCAN_DEFINE_UPSWEEP(idx, uint(-1), nbl_glsl_workgroupExclusiveMin, nbl_glsl_workgroupInclusiveMin)
}

uint nbl_glsl_ext_Scan_upsweepMax(in uint idx)
{
	NBL_GLSL_EXT_SCAN_DEFINE_UPSWEEP(idx, 0u, nbl_glsl_workgroupExclusiveMax, nbl_glsl_workgroupInclusiveMax)
}

//
//	Downsweep
//

shared uint global_offset;

#define NBL_GLSL_EXT_SCAN_DEFINE_DOWNSWEEP(IDX, IDENTITY, BIN_OP)\
if (gl_LocalInvocationIndex == (_NBL_GLSL_WORKGROUP_SIZE_ - 1u))\
	global_offset = nbl_glsl_ext_Scan_getPaddedData(IDX, IDENTITY, false);\
barrier();\
uint data = global_offset;\
if (gl_LocalInvocationIndex != 0u && (gl_GlobalInvocationID.x < nbl_glsl_ext_Scan_Parameters_t_getElementCountPass()))\
{\
	uint prev_idx = STRIDED_IDX(gl_GlobalInvocationID.x - 1u);\
	data = BIN_OP(data, nbl_glsl_ext_Scan_getPaddedData(prev_idx, IDENTITY, false));\
}\
barrier();\
return data;

uint nbl_glsl_ext_Scan_downsweepAnd(in uint idx)
{
	NBL_GLSL_EXT_SCAN_DEFINE_DOWNSWEEP(idx, uint(-1), nbl_glsl_and)
}

uint nbl_glsl_ext_Scan_downsweepXor(in uint idx)
{
	NBL_GLSL_EXT_SCAN_DEFINE_DOWNSWEEP(idx, 0u, nbl_glsl_xor)
}

uint nbl_glsl_ext_Scan_downsweepOr(in uint idx)
{
	NBL_GLSL_EXT_SCAN_DEFINE_DOWNSWEEP(idx, 0u, nbl_glsl_or)
}

uint nbl_glsl_ext_Scan_downsweepAdd(in uint idx)
{
	NBL_GLSL_EXT_SCAN_DEFINE_DOWNSWEEP(idx, 0u, nbl_glsl_add)
}

uint nbl_glsl_ext_Scan_downsweepMul(in uint idx)
{
	NBL_GLSL_EXT_SCAN_DEFINE_DOWNSWEEP(idx, 1u, nbl_glsl_mul)
}

uint nbl_glsl_ext_Scan_downsweepMin(in uint idx)
{
	NBL_GLSL_EXT_SCAN_DEFINE_DOWNSWEEP(idx, 0u, min)
}

uint nbl_glsl_ext_Scan_downsweepMax(in uint idx)
{
	NBL_GLSL_EXT_SCAN_DEFINE_DOWNSWEEP(idx, uint(-1), max)
}

#define _NBL_GLSL_EXT_SCAN_INCLUDED_
#endif
