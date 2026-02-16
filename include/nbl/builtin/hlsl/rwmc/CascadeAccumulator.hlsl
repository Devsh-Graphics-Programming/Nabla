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
struct DefaultCascades
{
    using layer_type = CascadeLayerType;

    uint32_t cascadeSampleCounter[CascadeCount];
    CascadeLayerType data[CascadeCount];

    void addSampleIntoCascadeEntry(CascadeLayerType _sample, uint16_t lowerCascadeIndex, SplattingParameters::scalar_t lowerCascadeLevelWeight, SplattingParameters::scalar_t higherCascadeLevelWeight, uint32_t sampleCount)
    {
        const float reciprocalSampleCount = 1.0f / float(sampleCount);

        uint32_t lowerCascadeSampleCount = cascadeSampleCounter[lowerCascadeIndex];
        data[lowerCascadeIndex] += (_sample * lowerCascadeLevelWeight - (sampleCount - lowerCascadeSampleCount) * data[lowerCascadeIndex]) * reciprocalSampleCount;
        cascadeSampleCounter[lowerCascadeIndex] = sampleCount;

        uint16_t higherCascadeIndex = lowerCascadeIndex + 1u;
        if (higherCascadeIndex < CascadeCount)
        {
            uint32_t higherCascadeSampleCount = cascadeSampleCounter[higherCascadeIndex];
            data[higherCascadeIndex] += (_sample * higherCascadeLevelWeight - (sampleCount - higherCascadeSampleCount) * data[higherCascadeIndex]) * reciprocalSampleCount;
            cascadeSampleCounter[higherCascadeIndex] = sampleCount;
        }
    }
};

template<typename CascadesType>
struct CascadeAccumulator
{
    using input_sample_type = typename CascadesType::layer_type;
    using cascade_layer_scalar_type = typename vector_traits<input_sample_type>::scalar_type;
    using this_t = CascadeAccumulator<CascadesType>;
    using output_storage_type = CascadesType;
    output_storage_type accumulation;
    
    SplattingParameters splattingParameters;

    static this_t create(NBL_CONST_REF_ARG(SplattingParameters) settings)
    {
        this_t retval;
        for (int i = 0; i < CascadeCount; ++i)
        {
            retval.accumulation.data[i] = promote<input_sample_type, float32_t>(0.0f);
            retval.accumulation.cascadeSampleCounter[i] = 0u;
        }
        retval.splattingParameters = settings;

        return retval;
    }

    // most of this code is stolen from https://cg.ivd.kit.edu/publications/2018/rwmc/tool/split.cpp
    void addSample(uint32_t sampleCount, input_sample_type _sample)
    {
        const float32_t2 unpackedParams = hlsl::unpackHalf2x16(splattingParameters.packedLog2);
        const SplattingParameters::scalar_t log2Start = unpackedParams[0];
        const SplattingParameters::scalar_t rcpLog2Base = unpackedParams[1];
        const SplattingParameters::scalar_t luma = splattingParameters.getLuma<input_sample_type>(_sample);
        const SplattingParameters::scalar_t log2Luma = log2<SplattingParameters::scalar_t>(luma);
        const SplattingParameters::scalar_t cascade = log2Luma * rcpLog2Base - log2Start * rcpLog2Base;
        const SplattingParameters::scalar_t lastCascade = CascadeCount - 1;
        const SplattingParameters::scalar_t clampedCascade = clamp(cascade, 0, lastCascade);
        // c<=0 -> 0, c>=Count-1 -> Count-1 
        uint16_t lowerCascadeIndex = floor<SplattingParameters::scalar_t>(cascade);
        // 0 whenever clamped or `cascade` is integer (when `clampedCascade` is integer)
        SplattingParameters::scalar_t higherCascadeWeight = clampedCascade - floor<SplattingParameters::scalar_t>(clampedCascade);
        // never 0 thanks to magic of `1-fract(x)`
        SplattingParameters::scalar_t lowerCascadeWeight = SplattingParameters::scalar_t(1) - higherCascadeWeight;

        // handle super bright sample case
        if (cascade > lastCascade)
            lowerCascadeWeight = exp2(log2Start + (1.0f/rcpLog2Base) * (lastCascade) - log2Luma);

        accumulation.addSampleIntoCascadeEntry(_sample, lowerCascadeIndex, lowerCascadeWeight, higherCascadeWeight, sampleCount);
    }

    
};

}
}
}

#endif