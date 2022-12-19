#ifndef _NBL_HLSL_WORKGROUP_SIZE_
#define _NBL_HLSL_WORKGROUP_SIZE_ 256
#define _NBL_HLSL_WORKGROUP_SIZE_LOG2_ 8
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
#ifndef _NBL_HLSL_SCAN_GET_PARAMETERS_DEFINED_
Parameters_t scanParams;
Parameters_t getParameters()
{
	return scanParams;
}
#define _NBL_HLSL_SCAN_GET_PARAMETERS_DEFINED_
#endif

uint getIndirectElementCount();

#ifndef _NBL_HLSL_SCAN_GET_SCHEDULER_PARAMETERS_DEFINED_
DefaultSchedulerParameters_t schedulerParams;
DefaultSchedulerParameters_t getSchedulerParameters()
{
	scheduler::computeParameters(getIndirectElementCount(),scanParams,schedulerParams);
	return schedulerParams;
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
	if (bool(nbl::hlsl::scan::getIndirectElementCount()))
		nbl::hlsl::scan::main();
}
#define _NBL_HLSL_MAIN_DEFINED_
#endif