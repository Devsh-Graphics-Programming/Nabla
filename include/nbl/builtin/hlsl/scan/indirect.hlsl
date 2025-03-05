#include "nbl/builtin/hlsl/scan/descriptors.hlsl"
#include "nbl/builtin/hlsl/scan/virtual_workgroup.hlsl"
#include "nbl/builtin/hlsl/scan/default_scheduler.hlsl"

namespace nbl
{
namespace hlsl
{
namespace scan
{
    
uint32_t getIndirectElementCount();

Parameters_t scanParams;
    
Parameters_t getParameters()
{
    return scanParams;
}

DefaultSchedulerParameters_t schedulerParams;

DefaultSchedulerParameters_t getSchedulerParameters()
{
    scheduler::computeParameters(getIndirectElementCount(),scanParams,schedulerParams);
	return schedulerParams;
}

}
}
}

[numthreads(WORKGROUP_SIZE,1,1)]
void main()
{
    if(bool(nbl::hlsl::scan::getIndirectElementCount())) {
        // TODO call main from virtual_workgroup.hlsl
    }
}
#endif