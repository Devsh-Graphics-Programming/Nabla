#ifndef _NBL_BUILTIN_HLSL_RWMC_SPLATTING_PARAMETERS_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_RWMC_SPLATTING_PARAMETERS_HLSL_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

namespace nbl
{
namespace hlsl
{
namespace rwmc
{

struct SplattingParameters
{
    // float16_t log2Start; 0
    // float16_t log2Base; 1
    // pack as Half2x16
    int32_t packedLog2;
};

}
}
}

#endif