#ifndef _NBL_BUILTIN_HLSL_RWMC_SPLATTING_PARAMETERS_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_RWMC_SPLATTING_PARAMETERS_HLSL_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include <nbl/builtin/hlsl/colorspace.hlsl>

namespace nbl
{
namespace hlsl
{
namespace rwmc
{

struct SSplattingParameters
{
    using scalar_t = float32_t;
    scalar_t RcpLog2Base;
    scalar_t Log2BaseRootOfStart;
    scalar_t BrightSampleLumaBias;

    template<typename CascadeLayerType, typename Colorspace = colorspace::scRGB>
    scalar_t calcLuma(NBL_CONST_REF_ARG(CascadeLayerType) col)
    {
        return hlsl::dot<CascadeLayerType>(hlsl::transpose(Colorspace::ToXYZ())[1], col);
    }
};

struct SPackedSplattingParameters
{
    // float16_t rcpLog2Base; 0
    // float16_t log2BaseRootOfStart; 1
    // pack as Half2x16
    int32_t PackedRcpLog2BaseAndLog2BaseRoot;
    float32_t BrightSampleLumaBias;

    static SPackedSplattingParameters create(float32_t base, float32_t start, uint32_t cascadeCount)
    {
        const float32_t rcpLog2Base = 1.0f / hlsl::log2(base);
        const float32_t log2BaseRootOfStart = hlsl::log2(start) * rcpLog2Base;
        const float32_t brightSampleLumaBias = (log2BaseRootOfStart + float32_t(cascadeCount - 1u)) / rcpLog2Base;
        float32_t2 packLogs = float32_t2(rcpLog2Base, log2BaseRootOfStart);
        
        SPackedSplattingParameters retval;
        retval.PackedRcpLog2BaseAndLog2BaseRoot = hlsl::packHalf2x16(packLogs);
        retval.BrightSampleLumaBias = brightSampleLumaBias;
        return retval;
    }

    SSplattingParameters unpack()
    {
        SSplattingParameters retval;
        const float32_t2 unpackedRcpLog2BaseAndLog2BaseRoot = hlsl::unpackHalf2x16(PackedRcpLog2BaseAndLog2BaseRoot);
        retval.RcpLog2Base = unpackedRcpLog2BaseAndLog2BaseRoot[0];
        retval.Log2BaseRootOfStart = unpackedRcpLog2BaseAndLog2BaseRoot[1];
        retval.BrightSampleLumaBias = BrightSampleLumaBias;
        return retval;
    }
};

}
}
}

#endif
