#ifndef _NBL_HLSL_SCAN_PARAMETERS_STRUCT_INCLUDED_
#define _NBL_HLSL_SCAN_PARAMETERS_STRUCT_INCLUDED_

#define NBL_BUILTIN_MAX_SCAN_LEVELS 7

#ifdef __cplusplus
#define uint uint32_t
#endif

namespace nbl
{
namespace hlsl
{
namespace scan
{
	// REVIEW: Putting topLevel second allows better alignment for packing of constant variables, assuming lastElement has length 4. (https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-packing-rules)
	struct Parameters_t {
		uint lastElement[NBL_BUILTIN_MAX_SCAN_LEVELS/2+1];
		uint topLevel;
		uint temporaryStorageOffset[NBL_BUILTIN_MAX_SCAN_LEVELS/2];
	}
}
}
}

#ifdef __cplusplus
#undef uint
#endif

#endif