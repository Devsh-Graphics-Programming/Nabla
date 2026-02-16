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
    using scalar_t = float32_t;

    // float16_t log2Start; 0
    // float16_t rcpLog2Base; 1
    // pack as Half2x16
    int32_t packedLog2;

    template<typename CascadeLayerType>
    scalar_t getLuma(NBL_CONST_REF_ARG(CascadeLayerType) col)
    {
        return hlsl::dot<CascadeLayerType>(hlsl::transpose(colorspace::scRGBtoXYZ)[1], col);
    }
};

}
}
}

#endif