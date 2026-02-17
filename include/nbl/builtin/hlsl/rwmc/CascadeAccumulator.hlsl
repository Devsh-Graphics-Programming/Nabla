#ifndef _NBL_HLSL_RWMC_CASCADE_ACCUMULATOR_INCLUDED_
#define _NBL_HLSL_RWMC_CASCADE_ACCUMULATOR_INCLUDED_
#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/promote.hlsl>
#include <nbl/builtin/hlsl/vector_utils/vector_traits.hlsl>
#include <nbl/builtin/hlsl/rwmc/SplattingParameters.hlsl>

namespace nbl
{
namespace hlsl
{
namespace rwmc
{

template<typename CascadeLayerType, uint32_t CascadeCountValue, typename SampleCountType = uint16_t NBL_PRIMARY_REQUIRES(concepts::Vector<CascadeLayerType> && concepts::UnsignedIntegralScalar<SampleCountType>)
struct DefaultCascades
{
    using layer_type = CascadeLayerType;
    using sample_count_type = SampleCountType;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t CascadeCount = CascadeCountValue;

    sample_count_type cascadeSampleCounter[CascadeCount];
    CascadeLayerType data[CascadeCount];

    void clear(uint32_t cascadeIx)
    {
        cascadeSampleCounter[cascadeIx] = sample_count_type(0u);
        data[cascadeIx] = promote<CascadeLayerType, float32_t>(0.0f);
    }

    void addSampleIntoCascadeEntry(CascadeLayerType _sample, uint16_t lowerCascadeIndex, SplattingParameters::scalar_t lowerCascadeLevelWeight, SplattingParameters::scalar_t higherCascadeLevelWeight, uint32_t sampleCount)
    {
        const SplattingParameters::scalar_t reciprocalSampleCount = SplattingParameters::scalar_t(1.0f) / SplattingParameters::scalar_t(sampleCount);

        sample_count_type lowerCascadeSampleCount = cascadeSampleCounter[lowerCascadeIndex];
        data[lowerCascadeIndex] += (_sample * lowerCascadeLevelWeight - (sampleCount - lowerCascadeSampleCount) * data[lowerCascadeIndex]) * reciprocalSampleCount;
        cascadeSampleCounter[lowerCascadeIndex] = sample_count_type(sampleCount);

        uint16_t higherCascadeIndex = lowerCascadeIndex + uint16_t(1u);
        if (higherCascadeIndex < CascadeCount)
        {
            sample_count_type higherCascadeSampleCount = cascadeSampleCounter[higherCascadeIndex];
            data[higherCascadeIndex] += (_sample * higherCascadeLevelWeight - (sampleCount - higherCascadeSampleCount) * data[higherCascadeIndex]) * reciprocalSampleCount;
            cascadeSampleCounter[higherCascadeIndex] = sample_count_type(sampleCount);
        }
    }
};

template<typename CascadesType>
struct CascadeAccumulator
{
    using scalar_t = typename SplattingParameters::scalar_t;
    using input_sample_type = typename CascadesType::layer_type;
    using this_t = CascadeAccumulator<CascadesType>;
    using cascades_type = CascadesType;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t CascadeCount = cascades_type::CascadeCount;
    NBL_CONSTEXPR_STATIC_INLINE scalar_t LastCascade = scalar_t(CascadeCount - 1u);
    cascades_type accumulation;
    
    SplattingParameters splattingParameters;
    SplattingParameters::SPrecomputed splattingParametersPrecomputed;

    static this_t create(NBL_CONST_REF_ARG(SplattingParameters) settings)
    {
        this_t retval;
        for (uint32_t i = 0u; i < CascadeCount; ++i)
            retval.accumulation.clear(i);
        retval.splattingParameters = settings;
        retval.splattingParametersPrecomputed = settings.template precompute<CascadeCount>();

        return retval;
    }

    // most of this code is stolen from https://cg.ivd.kit.edu/publications/2018/rwmc/tool/split.cpp
    void addSample(uint32_t sampleCount, input_sample_type _sample)
    {
        const scalar_t luma = splattingParameters.calcLuma<input_sample_type>(_sample);
        const scalar_t log2Luma = log2<scalar_t>(luma);
        const scalar_t cascade = log2Luma * splattingParametersPrecomputed.RcpLog2Base - splattingParametersPrecomputed.Log2BaseRootOfStart;
        const scalar_t clampedCascade = clamp(cascade, scalar_t(0), LastCascade);
        const scalar_t clampedCascadeFloor = floor<scalar_t>(clampedCascade);
        // c<=0 -> 0, c>=Count-1 -> Count-1 
        uint16_t lowerCascadeIndex = uint16_t(clampedCascadeFloor);
        // 0 whenever clamped or `cascade` is integer (when `clampedCascade` is integer)
        scalar_t higherCascadeWeight = clampedCascade - clampedCascadeFloor;
        // never 0 thanks to magic of `1-fract(x)`
        scalar_t lowerCascadeWeight = scalar_t(1) - higherCascadeWeight;

        // handle super bright sample case
        if (cascade > LastCascade)
            lowerCascadeWeight = exp2(splattingParametersPrecomputed.BrightSampleLumaBias - log2Luma);

        accumulation.addSampleIntoCascadeEntry(_sample, lowerCascadeIndex, lowerCascadeWeight, higherCascadeWeight, sampleCount);
    }

    
};

}
}
}

#endif
