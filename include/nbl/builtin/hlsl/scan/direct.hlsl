#ifndef _NBL_HLSL_WORKGROUP_SIZE_
#define _NBL_HLSL_WORKGROUP_SIZE_ 256
#endif

#include <nbl/builtin/hlsl/scan/descriptors.hlsl>
#include <nbl/builtin/hlsl/scan/virtual_workgroup.hlsl>
#include <nbl/builtin/hlsl/scan/default_scheduler.hlsl>

namespace nbl
{
namespace hlsl
{
namespace scan
{
#ifndef _NBL_HLSL_SCAN_PUSH_CONSTANTS_DEFINED_
	cbuffer PC // REVIEW: register and packoffset selection
	{
		Parameters_t scanParams;
		DefaultSchedulerParameters_t schedulerParams;
	};
#define _NBL_HLSL_SCAN_PUSH_CONSTANTS_DEFINED_
#endif

#ifndef _NBL_HLSL_SCAN_GET_PARAMETERS_DEFINED_
Parameters_t getParameters()
{
	return pc.scanParams;
}
#define _NBL_HLSL_SCAN_GET_PARAMETERS_DEFINED_
#endif

#ifndef _NBL_HLSL_SCAN_GET_SCHEDULER_PARAMETERS_DEFINED_
DefaultSchedulerParameters_t getSchedulerParameters()
{
	return pc.schedulerParams;
}
#define _NBL_HLSL_SCAN_GET_SCHEDULER_PARAMETERS_DEFINED_
#endif
}
}
}

#ifndef _NBL_HLSL_MAIN_DEFINED_
[numthreads(_NBL_HLSL_WORKGROUP_SIZE_, 1, 1)]
void CSMain()
{
	nbl::hlsl::scan::main();
}
#define _NBL_HLSL_MAIN_DEFINED_
#endif