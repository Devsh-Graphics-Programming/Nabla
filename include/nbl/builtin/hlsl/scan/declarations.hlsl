#ifndef _NBL_HLSL_SCAN_DECLARATIONS_INCLUDED_
#define _NBL_HLSL_SCAN_DECLARATIONS_INCLUDED_

// REVIEW: Not sure if this file is needed in HLSL implementation

#include <nbl/builtin/hlsl/scan/parameters_struct.hlsl>


#ifndef _NBL_HLSL_SCAN_GET_PARAMETERS_DECLARED_
namespace nbl
{
namespace hlsl
{
namespace scan
{
	Parameters_t getParameters();
}
}
}
#define _NBL_HLSL_SCAN_GET_PARAMETERS_DECLARED_
#endif

#ifndef _NBL_HLSL_SCAN_GET_PADDED_DATA_DECLARED_
namespace nbl
{
namespace hlsl
{
namespace scan
{
	template<typename Storage_t>
	void getData(
		inout Storage_t data,
		in uint levelInvocationIndex,
		in uint localWorkgroupIndex,
		in uint treeLevel,
		in uint pseudoLevel
	);
}
}
}
#define _NBL_HLSL_SCAN_GET_PADDED_DATA_DECLARED_
#endif

#ifndef _NBL_HLSL_SCAN_SET_DATA_DECLARED_
namespace nbl
{
namespace hlsl
{
namespace scan
{
	template<typename Storage_t>
	void setData(
		in Storage_t data,
		in uint levelInvocationIndex,
		in uint localWorkgroupIndex,
		in uint treeLevel,
		in uint pseudoLevel,
		in bool inRange
	);
}
}
}
#define _NBL_HLSL_SCAN_SET_DATA_DECLARED_
#endif

#endif