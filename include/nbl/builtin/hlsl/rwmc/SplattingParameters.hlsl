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
    // float16_t baseRootOfStart; 0
    // float16_t rcpLog2Base; 1
    // pack as Half2x16
    int32_t PackedBaseRootAndRcpLog2Base;

    // float16_t log2BaseRootOfStart; 2
    // float16_t brightSampleLumaBias; 3
    // pack as Half2x16
    int32_t PackedLog2BaseRootAndBrightSampleLumaBias;

    static SPackedSplattingParameters create(float32_t base, float32_t start, uint32_t cascadeCount)
    {
        const float32_t rcpLog2Base = 1.0f / hlsl::log2(base);
        const float32_t baseRootOfStart = hlsl::exp2(hlsl::log2(start) * rcpLog2Base);
        const float32_t log2BaseRootOfStart = hlsl::log2(baseRootOfStart);
        const float32_t brightSampleLumaBias = (log2BaseRootOfStart + float32_t(cascadeCount - 1u)) / rcpLog2Base;
        float32_t2 packLogs = float32_t2(baseRootOfStart, rcpLog2Base);
        float32_t2 packPrecomputed = float32_t2(log2BaseRootOfStart, brightSampleLumaBias);
        
        SPackedSplattingParameters retval;
        retval.PackedBaseRootAndRcpLog2Base = hlsl::packHalf2x16(packLogs);
        retval.PackedLog2BaseRootAndBrightSampleLumaBias = hlsl::packHalf2x16(packPrecomputed);
        return retval;
    }

    SSplattingParameters unpack()
    {
        SSplattingParameters retval;
        const float32_t2 unpackedBaseRootAndRcpLog2Base = hlsl::unpackHalf2x16(PackedBaseRootAndRcpLog2Base);
        const float32_t2 unpackedLog2BaseRootAndBrightSampleLumaBias = hlsl::unpackHalf2x16(PackedLog2BaseRootAndBrightSampleLumaBias);
        retval.RcpLog2Base = unpackedBaseRootAndRcpLog2Base[1];
        retval.Log2BaseRootOfStart = unpackedLog2BaseRootAndBrightSampleLumaBias[0];
        retval.BrightSampleLumaBias = unpackedLog2BaseRootAndBrightSampleLumaBias[1];
        return retval;
    }
};

}
}
}

#endif
