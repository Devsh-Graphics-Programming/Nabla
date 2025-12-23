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

    static SplattingParameters create(const scalar_t base, const scalar_t start, const uint32_t cascadeCount)
    {
        SplattingParameters retval;
        const scalar_t log2Base = hlsl::log2(base);
        const scalar_t log2Start = hlsl::log2(start);
        retval.lastCascadeLuma = hlsl::exp2(log2Start + log2Base * (cascadeCount - 1));
        retval.rcpLog2Base = scalar_t(1.0) / log2Base;
        retval.baseRootOfStart = log2Start * retval.rcpLog2Base;
        return retval;
    }

    scalar_t lastCascadeLuma;
    scalar_t baseRootOfStart;
    scalar_t rcpLog2Base;
};

}
}
}

#endif