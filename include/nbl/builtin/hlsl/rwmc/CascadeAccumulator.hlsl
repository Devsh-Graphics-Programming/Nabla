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

template<typename CascadeLayerType, uint16_t CascadeCount, typename SampleCountType = uint16_t NBL_PRIMARY_REQUIRES(concepts::Vector<CascadeLayerType> && concepts::UnsignedIntegralScalar<SampleCountType>)
struct DefaultCascades
{
    // required public interfaces (TODO: a concept for it)
    using layer_type = CascadeLayerType;
    using sample_count_type = SampleCountType;
    using weight_t = typename SSplattingParameters::scalar_t;

    inline uint16_t getLastCascade() {return CascadeCount-uint16_t(1);}

    void clear()
    {
        for (uint16_t i=0u; i<CascadeCount; ++i)
        {
            __cascadeSampleCounter[i] = sample_count_type(0u);
            __data[i] = promote<CascadeLayerType,float32_t>(0.0f);
        }
    }

    void addSampleIntoCascadeEntry(const layer_type _sample, const uint16_t lowerCascadeIndex, const weight_t lowerCascadeLevelWeight, const weight_t higherCascadeLevelWeight, const sample_count_type sampleCount)
    {
        const weight_t reciprocalSampleCount = _static_cast<weight_t>(1)/_static_cast<weight_t>(sampleCount);

        sample_count_type lowerCascadeSampleCount = __cascadeSampleCounter[lowerCascadeIndex];
        __data[lowerCascadeIndex] += (_sample * lowerCascadeLevelWeight - (sampleCount - lowerCascadeSampleCount) * __data[lowerCascadeIndex]) * reciprocalSampleCount;
        __cascadeSampleCounter[lowerCascadeIndex] = sampleCount;

        uint16_t higherCascadeIndex = lowerCascadeIndex + uint16_t(1u);
        if (higherCascadeIndex < CascadeCount)
        {
            sample_count_type higherCascadeSampleCount = __cascadeSampleCounter[higherCascadeIndex];
            __data[higherCascadeIndex] += (_sample * higherCascadeLevelWeight - (sampleCount - higherCascadeSampleCount) * __data[higherCascadeIndex]) * reciprocalSampleCount;
            __cascadeSampleCounter[higherCascadeIndex] = sampleCount;
        }
    }
    
    // private
    sample_count_type __cascadeSampleCounter[CascadeCount];
    CascadeLayerType __data[CascadeCount];
};

template<typename CascadesType>
struct CascadeAccumulator
{
    using scalar_t = typename SSplattingParameters::scalar_t;
    using input_sample_type = typename CascadesType::layer_type;
    using sample_count_type = typename CascadesType::sample_count_type;
    using weight_t = typename CascadesType::weight_t;
    using this_t = CascadeAccumulator<CascadesType>;
    using cascades_type = CascadesType;

    cascades_type accumulation;
    SSplattingParameters splattingParameters;

    static this_t create(NBL_CONST_REF_ARG(SPackedSplattingParameters) settings, const bool clear=true)
    {
        this_t retval;
        if (clear)
            retval.accumulation.clear();
        retval.splattingParameters = settings.unpack();

        return retval;
    }

    // most of this code is stolen from https://cg.ivd.kit.edu/publications/2018/rwmc/tool/split.cpp
    void addSample(const sample_count_type sampleCount, input_sample_type _sample)
    {
        const uint16_t lastCascade = accumulation.getLastCascade();

        const weight_t luma = splattingParameters.calcLuma<input_sample_type>(_sample);
        const weight_t log2Luma = hlsl::log2<weight_t>(luma);
        const scalar_t cascade = log2Luma * splattingParameters.RcpLog2Base - splattingParameters.Log2BaseRootOfStart;
        const scalar_t clampedCascade = hlsl::clamp<scalar_t>(cascade, scalar_t(0), lastCascade);
        const scalar_t clampedCascadeFloor = hlsl::floor(clampedCascade);
        // c<=0 -> 0, c>=Count-1 -> Count-1 
        const uint16_t lowerCascadeIndex = _static_cast<uint16_t>(clampedCascadeFloor);
        // 0 whenever clamped or `cascade` is integer (when `clampedCascade` is integer)
        const weight_t higherCascadeWeight = _static_cast<weight_t>(clampedCascade - clampedCascadeFloor);
        // never 0 thanks to magic of `1-fract(x)`
        weight_t lowerCascadeWeight = weight_t(1) - higherCascadeWeight;

        // handle super bright sample case
        if (cascade > lastCascade)
            lowerCascadeWeight = hlsl::exp2(_static_cast<weight_t>(splattingParameters.BrightSampleLumaBias - log2Luma));

        accumulation.addSampleIntoCascadeEntry(_sample, lowerCascadeIndex, lowerCascadeWeight, higherCascadeWeight, sampleCount);
    }
     
    
};

}
}
}

#endif
