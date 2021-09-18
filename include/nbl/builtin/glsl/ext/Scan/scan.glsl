#ifndef _NBL_GLSL_EXT_SCAN_INCLUDED_
#define _NBL_GLSL_EXT_SCAN_INCLUDED_

#include <nbl/builtin/glsl/workgroup/arithmetic.glsl>
#include <nbl/builtin/glsl/math/typeless_arithmetic.glsl>
#include <nbl/builtin/glsl/limits/numeric.glsl>

#include "nbl/builtin/glsl/ext/Scan/parameters.glsl"

#ifndef _NBL_GLSL_EXT_SCAN_GET_PADDED_DATA_DECLARED_

uint nbl_glsl_ext_Scan_getPaddedData(in uint idx, in uint pad_val);
int nbl_glsl_ext_Scan_getPaddedData(in uint idx, in int pad_val);
float nbl_glsl_ext_Scan_getPaddedData(in uint idx, in float pad_val);

#define _NBL_GLSL_EXT_SCAN_GET_PADDED_DATA_DECLARED_
#endif

#ifndef _NBL_GLSL_EXT_SCAN_GET_PARAMETERS_DEFINED_
#error "You need to define `nbl_glsl_ext_Scan_getParameters` and mark `_NBL_GLSL_EXT_SCAN_GET_PARAMETERS_DEFINED_`!"
#endif

#ifndef _NBL_GLSL_EXT_SCAN_GET_PADDED_DATA_DEFINED_
#error "You need to define `nbl_glsl_ext_Scan_getPaddedData` and mark `_NBL_GLSL_EXT_SCAN_GET_PADDED_DATA_DEFINED_`!"
#endif

//
// Upsweep
//

#define NBL_GLSL_EXT_SCAN_DEFINE_UPSWEEP(BIN_OP_NAME, TYPE, IDENTITY) void nbl_glsl_ext_Scan_upsweep##BIN_OP_NAME(in uint idx, out TYPE val)\
{\
	TYPE data = nbl_glsl_ext_Scan_getPaddedData(idx, IDENTITY);\
	if (gl_NumWorkGroups.x == 1u)\
		val = nbl_glsl_workgroupExclusive##BIN_OP_NAME(data);\
	else\
		val = nbl_glsl_workgroupInclusive##BIN_OP_NAME(data);\
}

NBL_GLSL_EXT_SCAN_DEFINE_UPSWEEP(And, uint, ~0u)
NBL_GLSL_EXT_SCAN_DEFINE_UPSWEEP(And, int, int(~0u))
NBL_GLSL_EXT_SCAN_DEFINE_UPSWEEP(And, float, uintBitsToFloat(~0u))

NBL_GLSL_EXT_SCAN_DEFINE_UPSWEEP(Xor, uint, 0u)
NBL_GLSL_EXT_SCAN_DEFINE_UPSWEEP(Xor, int, 0)
NBL_GLSL_EXT_SCAN_DEFINE_UPSWEEP(Xor, float, uintBitsToFloat(0u))

NBL_GLSL_EXT_SCAN_DEFINE_UPSWEEP(Or, uint, 0u)
NBL_GLSL_EXT_SCAN_DEFINE_UPSWEEP(Or, int, 0)
NBL_GLSL_EXT_SCAN_DEFINE_UPSWEEP(Or, float, uintBitsToFloat(0u))

NBL_GLSL_EXT_SCAN_DEFINE_UPSWEEP(Add, uint, 0u)
NBL_GLSL_EXT_SCAN_DEFINE_UPSWEEP(Add, int, 0)
NBL_GLSL_EXT_SCAN_DEFINE_UPSWEEP(Add, float, 0.0)

NBL_GLSL_EXT_SCAN_DEFINE_UPSWEEP(Mul, uint, 1u)
NBL_GLSL_EXT_SCAN_DEFINE_UPSWEEP(Mul, int, 1)
NBL_GLSL_EXT_SCAN_DEFINE_UPSWEEP(Mul, float, 1.0)

NBL_GLSL_EXT_SCAN_DEFINE_UPSWEEP(Min, uint, ~0u)
NBL_GLSL_EXT_SCAN_DEFINE_UPSWEEP(Min, int, INT_MAX)
NBL_GLSL_EXT_SCAN_DEFINE_UPSWEEP(Min, float, FLT_INF)

NBL_GLSL_EXT_SCAN_DEFINE_UPSWEEP(Max, uint, 0u)
NBL_GLSL_EXT_SCAN_DEFINE_UPSWEEP(Max, int, INT_MIN)
NBL_GLSL_EXT_SCAN_DEFINE_UPSWEEP(Max, float, -FLT_INF)

//
//	Downsweep
//

#define NBL_GLSL_EXT_SCAN_DEFINE_DOWNSWEEP(BIN_OP_NAME, TYPE, IDENTITY, BIN_OP, CONV, INVCONV)\
void nbl_glsl_ext_Scan_downsweep##BIN_OP_NAME(in uint idx, out TYPE val)\
{\
	TYPE data;\
	const uint last_idx = _NBL_GLSL_WORKGROUP_SIZE_ - 1u;\
	if (gl_LocalInvocationIndex == last_idx)\
		data = nbl_glsl_ext_Scan_getPaddedData(idx, IDENTITY);\
	data = nbl_glsl_workgroupBroadcast(data, last_idx);\
	if (gl_LocalInvocationIndex != 0u && (gl_GlobalInvocationID.x < nbl_glsl_ext_Scan_Parameters_t_getElementCountPass()))\
	{\
		uint prev_idx = gl_GlobalInvocationID.x * nbl_glsl_ext_Scan_Parameters_t_getStride() - 1u;\
		data = INVCONV(BIN_OP(CONV(data), CONV(nbl_glsl_ext_Scan_getPaddedData(prev_idx, IDENTITY))));\
	}\
	barrier();\
	val = data;\
}

// A lot of this clutter related to CONV/INVCONV could be removed by extending `typeless_arithmetic.glsl`
// to allow `nbl_glsl_and` and friends to work on ints and floats, for example. This could also benefit
// other places plagued by CONV/INCONV in the code base like reduction functions in arithmetic.glsl by
// doing all the conversion in a single place it typeless_arithmetic.glsl.
// Note: Except that `float` bitwise operators cannot be trusted, they might flush denormalized bitpatterns to 0.
NBL_GLSL_EXT_SCAN_DEFINE_DOWNSWEEP(And, uint, ~0u, nbl_glsl_and, nbl_glsl_identityFunction, nbl_glsl_identityFunction)
NBL_GLSL_EXT_SCAN_DEFINE_DOWNSWEEP(And, int, int(~0u), nbl_glsl_and, uint, int)
NBL_GLSL_EXT_SCAN_DEFINE_DOWNSWEEP(And, float, uintBitsToFloat(~0u), nbl_glsl_and, floatBitsToUint, uintBitsToFloat)

NBL_GLSL_EXT_SCAN_DEFINE_DOWNSWEEP(Xor, uint, 0u, nbl_glsl_xor, nbl_glsl_identityFunction, nbl_glsl_identityFunction)
NBL_GLSL_EXT_SCAN_DEFINE_DOWNSWEEP(Xor, int, int(0u), nbl_glsl_xor, uint, int)
NBL_GLSL_EXT_SCAN_DEFINE_DOWNSWEEP(Xor, float, uintBitsToFloat(0u), nbl_glsl_xor, floatBitsToUint, uintBitsToFloat)

NBL_GLSL_EXT_SCAN_DEFINE_DOWNSWEEP(Or, uint, 0u, nbl_glsl_or, nbl_glsl_identityFunction, nbl_glsl_identityFunction)
NBL_GLSL_EXT_SCAN_DEFINE_DOWNSWEEP(Or, int, int(0u), nbl_glsl_or, uint, int)
NBL_GLSL_EXT_SCAN_DEFINE_DOWNSWEEP(Or, float, uintBitsToFloat(0u), nbl_glsl_or, floatBitsToUint, uintBitsToFloat)

NBL_GLSL_EXT_SCAN_DEFINE_DOWNSWEEP(Add, uint, 0u, nbl_glsl_add, nbl_glsl_identityFunction, nbl_glsl_identityFunction)
NBL_GLSL_EXT_SCAN_DEFINE_DOWNSWEEP(Add, int, 0, nbl_glsl_add, uint, int)
NBL_GLSL_EXT_SCAN_DEFINE_DOWNSWEEP(Add, float, 0.0, nbl_glsl_add, float, float)

NBL_GLSL_EXT_SCAN_DEFINE_DOWNSWEEP(Mul, uint, 1u, nbl_glsl_mul, nbl_glsl_identityFunction, nbl_glsl_identityFunction)
NBL_GLSL_EXT_SCAN_DEFINE_DOWNSWEEP(Mul, int, 1, nbl_glsl_mul, uint, int)
NBL_GLSL_EXT_SCAN_DEFINE_DOWNSWEEP(Mul, float, 1.0, nbl_glsl_mul, float, float)

NBL_GLSL_EXT_SCAN_DEFINE_DOWNSWEEP(Min, uint, ~0u, min, nbl_glsl_identityFunction, nbl_glsl_identityFunction)
NBL_GLSL_EXT_SCAN_DEFINE_DOWNSWEEP(Min, int, INT_MAX, min, int, int)
NBL_GLSL_EXT_SCAN_DEFINE_DOWNSWEEP(Min, float, FLT_INF, min, float, float)

NBL_GLSL_EXT_SCAN_DEFINE_DOWNSWEEP(Max, uint, 0u, max, nbl_glsl_identityFunction, nbl_glsl_identityFunction)
NBL_GLSL_EXT_SCAN_DEFINE_DOWNSWEEP(Max, int, INT_MIN, max, int, int)
NBL_GLSL_EXT_SCAN_DEFINE_DOWNSWEEP(Max, float, -FLT_INF, max, float, float)

#endif
