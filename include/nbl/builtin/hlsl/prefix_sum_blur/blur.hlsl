#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "nbl/builtin/hlsl/workgroup/arithmetic.hlsl"
#include "nbl/builtin/hlsl/device_capabilities_traits.hlsl"
#include "nbl/builtin/hlsl/enums.hlsl"

#ifndef _NBL_BUILTIN_PREFIX_SUM_BLUR_INCLUDED_
#define _NBL_BUILTIN_PREFIX_SUM_BLUR_INCLUDED_

namespace nbl
{
namespace hlsl
{
namespace prefix_sum_blur
{

// Prefix-Sum Blur using SAT (Summed Area Table) technique.
// `scanScract` and `_sampler.prefixSumAccessor` must not to alias.
template<
    typename DataAccessor,
    typename ScanSharedAccessor,
    typename Sampler,
    uint16_t WorkgroupSize,
    class device_capabilities=void> // TODO: define concepts for the Box1D and apply constraints
struct Blur1D
{
    void operator()(
        NBL_REF_ARG(DataAccessor) data,
        NBL_REF_ARG(ScanSharedAccessor) scanScratch,
        NBL_REF_ARG(Sampler) _sampler,
        const uint16_t channel)
    {
        const uint16_t end = data.linearSize();
        const uint16_t localInvocationIndex = workgroup::SubgroupContiguousIndex();

        // prefix sum
        // note the dynamically uniform loop condition 
        for (uint16_t baseIx = 0; baseIx < end;)
        {
            const uint16_t ix = localInvocationIndex + baseIx;
            float32_t input = data.template get<float32_t>(channel, ix);
            // dynamically uniform condition
            if (baseIx != 0)
            {
                // take result of previous prefix sum and add it to first element here
                if (localInvocationIndex == 0)
                    input += _sampler.prefixSumAccessor.template get<float32_t>(baseIx - 1);
            }
            const float32_t sum = workgroup::inclusive_scan<plus<float32_t>, WorkgroupSize, device_capabilities>::template __call(input, scanScratch);
            // loop increment
            baseIx += WorkgroupSize;
            // save prefix sum results
            if (ix < end)
                _sampler.prefixSumAccessor.template set<float32_t>(ix, sum);
            // previous prefix sum must have finished before we ask for results
            _sampler.prefixSumAccessor.workgroupExecutionAndMemoryBarrier();
        }

        // TODO: split this Blur1D into two separate functors:
        //     - multi-wg-wide prefix sum
        //     - the SAT sampling
        const float32_t last = end - 1;
        for (float32_t ix = localInvocationIndex; ix < end; ix += WorkgroupSize)
        {
            const float32_t result = _sampler(ix, radius, borderColor[channel]);
            data.template set<float32_t>(channel, uint16_t(ix), result);
        }
    }

    vector<float32_t, DataAccessor::Channels> borderColor;
    float32_t radius;
};

}
}
}

#endif