#ifndef _NBL_BUILTIN_HLSL_RWMC_SPLATTING_PARAMETERS_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_RWMC_SPLATTING_PARAMETERS_HLSL_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include <nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl>

namespace nbl
{
namespace hlsl
{
namespace rwmc
{

struct SplattingParameters
{
    using scalar_t = float32_t;

    // float16_t baseRootOfStart; 0
    // float16_t rcpLog2Base; 1
    // pack as Half2x16
    int32_t packedLog2;

    float32_t2 unpackedLog2Parameters()
    {
        return hlsl::unpackHalf2x16(packedLog2);
    }

    scalar_t baseRootOfStart()
    {
        return unpackedLog2Parameters()[0];
    }

    scalar_t rcpLog2Base()
    {
        return unpackedLog2Parameters()[1];
    }

    template<typename CascadeLayerType>
    scalar_t calcLuma(NBL_CONST_REF_ARG(CascadeLayerType) col)
    {
        return hlsl::dot<CascadeLayerType>(hlsl::transpose(colorspace::scRGBtoXYZ)[1], col);
    }
};

}
}
}

#endif
