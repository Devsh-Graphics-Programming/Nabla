#ifndef _NBL_HLSL_RWMC_CASCADE_ACCUMULATOR_INCLUDED_
#define _NBL_HLSL_RWMC_CASCADE_ACCUMULATOR_INCLUDED_
#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/promote.hlsl>
#include <nbl/builtin/hlsl/vector_utils/vector_traits.hlsl>
#include <nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl>
#include <nbl/builtin/hlsl/rwmc/SplattingParameters.hlsl>

namespace nbl
{
namespace hlsl
{
namespace rwmc
{

template<typename CascadeLayerType, uint32_t CascadeCount NBL_PRIMARY_REQUIRES(concepts::Vector<CascadeLayerType>)
struct CascadeAccumulator
{
    struct CascadeEntry
    {
        uint32_t cascadeSampleCounter[CascadeCount];
        CascadeLayerType data[CascadeCount];

        void addSampleIntoCascadeEntry(CascadeLayerType _sample, uint32_t lowerCascadeIndex, float lowerCascadeLevelWeight, float higherCascadeLevelWeight, uint32_t sampleCount)
        {
            const float reciprocalSampleCount = 1.0f / float(sampleCount);

            uint32_t lowerCascadeSampleCount = cascadeSampleCounter[lowerCascadeIndex];
            data[lowerCascadeIndex] += (_sample * lowerCascadeLevelWeight - (sampleCount - lowerCascadeSampleCount) * data[lowerCascadeIndex]) * reciprocalSampleCount;
            cascadeSampleCounter[lowerCascadeIndex] = sampleCount;

            uint32_t higherCascadeIndex = lowerCascadeIndex + 1u;
            if (higherCascadeIndex < CascadeCount)
            {
                uint32_t higherCascadeSampleCount = cascadeSampleCounter[higherCascadeIndex];
                data[higherCascadeIndex] += (_sample * higherCascadeLevelWeight - (sampleCount - higherCascadeSampleCount) * data[higherCascadeIndex]) * reciprocalSampleCount;
                cascadeSampleCounter[higherCascadeIndex] = sampleCount;
            }
        }
    };

    using cascade_layer_scalar_type = typename vector_traits<CascadeLayerType>::scalar_type;
    using this_t = CascadeAccumulator<CascadeLayerType, CascadeCount>;
    using input_sample_type = CascadeLayerType;
    using output_storage_type = CascadeEntry;
    using initialization_data = SplattingParameters;
    output_storage_type accumulation;
    
    SplattingParameters splattingParameters;

    static this_t create(NBL_CONST_REF_ARG(SplattingParameters) settings)
    {
        this_t retval;
        for (int i = 0; i < CascadeCount; ++i)
        {
            retval.accumulation.data[i] = promote<CascadeLayerType, float32_t>(0.0f);
            retval.accumulation.cascadeSampleCounter[i] = 0u;
        }
        retval.splattingParameters = settings;

        return retval;
    }
    
    cascade_layer_scalar_type getLuma(NBL_CONST_REF_ARG(CascadeLayerType) col)
    {
        return hlsl::dot<CascadeLayerType>(hlsl::transpose(colorspace::scRGBtoXYZ)[1], col);
    }

    // most of this code is stolen from https://cg.ivd.kit.edu/publications/2018/rwmc/tool/split.cpp
    void addSample(uint32_t sampleCount, input_sample_type _sample)
    {
        const cascade_layer_scalar_type log2Start = splattingParameters.log2Start;
        const cascade_layer_scalar_type log2Base = splattingParameters.log2Base;
        const cascade_layer_scalar_type luma = getLuma(_sample);
        const cascade_layer_scalar_type log2Luma = log2<cascade_layer_scalar_type>(luma);
        const cascade_layer_scalar_type cascade = log2Luma * 1.f / log2Base - log2Start / log2Base;
        const cascade_layer_scalar_type clampedCascade = clamp(cascade, 0, CascadeCount - 1);
        // c<=0 -> 0, c>=Count-1 -> Count-1 
        uint32_t lowerCascadeIndex = floor<cascade_layer_scalar_type>(cascade);
        // 0 whenever clamped or `cascade` is integer (when `clampedCascade` is integer)
        cascade_layer_scalar_type higherCascadeWeight = clampedCascade - floor<cascade_layer_scalar_type>(clampedCascade);
        // never 0 thanks to magic of `1-fract(x)`
        cascade_layer_scalar_type lowerCascadeWeight = cascade_layer_scalar_type(1) - higherCascadeWeight;

        // handle super bright sample case
        if (cascade > CascadeCount - 1)
            lowerCascadeWeight = exp2(log2Start + log2Base * (CascadeCount - 1) - log2Luma);

        accumulation.addSampleIntoCascadeEntry(_sample, lowerCascadeIndex, lowerCascadeWeight, higherCascadeWeight, sampleCount);
    }

    
};

}
}
}

#endif