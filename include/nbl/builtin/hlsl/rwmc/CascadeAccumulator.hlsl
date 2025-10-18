#ifndef _NBL_HLSL_RWMC_CASCADE_ACCUMULATOR_INCLUDED_
#define _NBL_HLSL_RWMC_CASCADE_ACCUMULATOR_INCLUDED_
#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include <nbl/builtin/hlsl/vector_utils/vector_traits.hlsl>
#include <nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl>

namespace nbl
{
namespace hlsl
{
namespace rwmc
{

struct CascadeSettings
{
    uint32_t size;
    uint32_t start;
    uint32_t base;
};

template<typename CascadeLayerType, uint32_t CascadeSize>
struct CascadeEntry
{
    CascadeLayerType data[CascadeSize];
};

template<typename CascadeLayerType, uint32_t CascadeSize>
struct CascadeAccumulator
{
    using output_storage_type = CascadeEntry<CascadeLayerType, CascadeSize>;
    using initialization_data = CascadeSettings;
    output_storage_type accumulation;
    uint32_t cascadeSampleCounter[CascadeSize];
    CascadeSettings cascadeSettings;

    void initialize(in CascadeSettings settings)
    {
        for (int i = 0; i < CascadeSize; ++i)
        {
            accumulation.data[i] = (CascadeLayerType)0.0f;
            cascadeSampleCounter[i] = 0u;
        }

        cascadeSettings.size = settings.size;
        cascadeSettings.start = settings.start;
        cascadeSettings.base = settings.base;
    }

    typename vector_traits<CascadeLayerType>::scalar_type getLuma(NBL_CONST_REF_ARG(CascadeLayerType) col)
    {
        return hlsl::dot<CascadeLayerType>(hlsl::transpose(colorspace::scRGBtoXYZ)[1], col);
    }

    // most of this code is stolen from https://cg.ivd.kit.edu/publications/2018/rwmc/tool/split.cpp
    void addSample(uint32_t sampleIndex, float32_t3 sample)
    {
        float lowerScale = cascadeSettings.start;
        float upperScale = lowerScale * cascadeSettings.base;

        const float luma = getLuma(sample);

        uint32_t lowerCascadeIndex = 0u;
        while (!(luma < upperScale) && lowerCascadeIndex < cascadeSettings.size - 2)
        {
            lowerScale = upperScale;
            upperScale *= cascadeSettings.base;
            ++lowerCascadeIndex;
        }

        float lowerCascadeLevelWeight;
        float higherCascadeLevelWeight;

        if (luma <= lowerScale)
            lowerCascadeLevelWeight = 1.0f;
        else if (luma < upperScale)
            lowerCascadeLevelWeight = max(0.0f, (lowerScale / luma - lowerScale / upperScale) / (1.0f - lowerScale / upperScale));
        else // Inf, NaN ...
            lowerCascadeLevelWeight = 0.0f;

        if (luma < upperScale)
            higherCascadeLevelWeight = max(0.0f, 1.0f - lowerCascadeLevelWeight);
        else
            higherCascadeLevelWeight = upperScale / luma;

        uint32_t higherCascadeIndex = lowerCascadeIndex + 1u;

        const uint32_t sampleCount = sampleIndex + 1u;
        const float reciprocalSampleCount = 1.0f / float(sampleCount);
        accumulation.data[lowerCascadeIndex] += (sample * lowerCascadeLevelWeight - (sampleCount - (cascadeSampleCounter[lowerCascadeIndex])) * accumulation.data[lowerCascadeIndex]) * reciprocalSampleCount;
        accumulation.data[higherCascadeIndex] += (sample * higherCascadeLevelWeight - (sampleCount - (cascadeSampleCounter[higherCascadeIndex])) * accumulation.data[higherCascadeIndex]) * reciprocalSampleCount;
        cascadeSampleCounter[lowerCascadeIndex] = sampleCount;
        cascadeSampleCounter[higherCascadeIndex] = sampleCount;
    }
};

}
}
}

#endif