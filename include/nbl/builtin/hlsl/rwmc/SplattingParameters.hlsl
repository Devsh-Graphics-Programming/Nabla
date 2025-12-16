#ifndef _NBL_BUILTIN_HLSL_RWMC_SPLATTING_PARAMETERS_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_RWMC_SPLATTING_PARAMETERS_HLSL_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/tgmath.hlsl"

namespace nbl
{
namespace hlsl
{
namespace rwmc
{

struct SplattingParameters
{
    using scalar_t = float;

    static SplattingParameters create(const scalar_t base, const scalar_t start)
    {
        SplattingParameters retval;
        retval.rcpLog2Base = scalar_t(1.0) / hlsl::log2(base);
        retval.baseRootOfStart = hlsl::log2(start) * retval.rcpLog2Base;
        return retval;
    }
   
    scalar_t baseRootOfStart;
    scalar_t rcpLog2Base;
};

}
}
}

#endif