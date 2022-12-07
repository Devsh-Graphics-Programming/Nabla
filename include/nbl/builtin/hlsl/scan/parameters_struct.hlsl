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
	struct Parameters_t {
		uint topLevel;
		uint lastElement[NBL_BUILTIN_MAX_SCAN_LEVELS/2+1];
		uint temporaryStorageOffset[NBL_BUILTIN_MAX_SCAN_LEVELS/2];
	}
}
}
}

#ifdef __cplusplus
#undef uint
#endif

#endif