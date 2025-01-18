#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "nbl/builtin/hlsl/workgroup/arithmetic.hlsl"
#include "nbl/builtin/hlsl/workgroup/scratch_size.hlsl"
#include "nbl/builtin/hlsl/device_capabilities_traits.hlsl"
#include "nbl/builtin/hlsl/enums.hlsl"

namespace nbl
{
namespace hlsl
{
namespace box_blur
{

template<
    typename DataAccessor,
    typename SharedAccessor,
    typename ScanSharedAccessor,
    typename Sampler,
    uint16_t WorkgroupSize,
    class device_capabilities=void> // TODO: define concepts for the Box1D and apply constraints
struct Box1D
{
    // TODO: Generalize later on when Francesco enforces accessor-concepts in `workgroup` and adds a `SharedMemoryAccessor` concept
    struct ScanSharedAccessorWrapper
    {
        void get(const uint16_t ix, NBL_REF_ARG(float32_t) val)
        {
            val = base.template get<float32_t, uint16_t>(ix);
        }

        void set(const uint16_t ix, const float32_t val)
        {
            base.template set<float32_t, uint16_t>(ix, val);
        }

        void workgroupExecutionAndMemoryBarrier()
        {
            base.workgroupExecutionAndMemoryBarrier();
        }

        ScanSharedAccessor base;
    };

    void operator()(
        NBL_REF_ARG(DataAccessor) data,
        NBL_REF_ARG(SharedAccessor) scratch,
        NBL_REF_ARG(ScanSharedAccessor) scanScratch,
        NBL_REF_ARG(Sampler) boxSampler,
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
                    input += scratch.template get<float32_t>(baseIx - 1);
            }
            // need to copy-in / copy-out the accessor cause no references in HLSL - yay!
            ScanSharedAccessorWrapper scanScratchWrapper;
            scanScratchWrapper.base = scanScratch;
            const float32_t sum = workgroup::inclusive_scan<plus<float32_t>, WorkgroupSize, device_capabilities>::template __call(input, scanScratchWrapper);
            scanScratch = scanScratchWrapper.base;
            // loop increment
            baseIx += WorkgroupSize;
            // if doing the last prefix sum, we need to barrier to stop aliasing of temporary scratch for `inclusive_scan` and our scanline
            // TODO: might be worth adding a non-aliased mode as NSight says nr 1 hotspot is barrier waiting in this code
            if (end + ScanSharedAccessor::Size > SharedAccessor::Size)
                scratch.workgroupExecutionAndMemoryBarrier();
            // save prefix sum results
            if (ix < end)
                scratch.template set<float32_t>(ix, sum);
            // previous prefix sum must have finished before we ask for results
            scratch.workgroupExecutionAndMemoryBarrier();
        }

        const float32_t last = end - 1;
        const float32_t normalizationFactor = 1.f / (2.f * radius + 1.f);

        for (float32_t ix = localInvocationIndex; ix < end; ix += WorkgroupSize)
        {
            const float32_t result = boxSampler(scratch, ix, radius, borderColor[channel]);
            data.template set<float32_t>(channel, uint16_t(ix), result * normalizationFactor);
        }
    }

    vector<float32_t, DataAccessor::Channels> borderColor;
    float32_t radius;
};

template<typename PrefixSumAccessor, typename T>
struct BoxSampler
{
    uint16_t wrapMode;
    uint16_t linearSize;

    T operator()(NBL_REF_ARG(PrefixSumAccessor) prefixSumAccessor, float32_t ix, float32_t radius, float32_t borderColor)
    {
        const float32_t alpha = radius - floor(radius);
        const float32_t lastIdx = linearSize - 1;
        const float32_t rightIdx = float32_t(ix) + radius;
        const float32_t leftIdx = float32_t(ix) - radius;
        const int32_t rightFlIdx = (int32_t)floor(rightIdx);
        const int32_t rightClIdx = (int32_t)ceil(rightIdx);
        const int32_t leftFlIdx = (int32_t)floor(leftIdx);
        const int32_t leftClIdx = (int32_t)ceil(leftIdx);

        T result = 0;
        if (rightFlIdx < linearSize)
        {
            result += lerp(prefixSumAccessor.template get<T, uint32_t>(rightFlIdx), prefixSumAccessor.template get<T, uint32_t>(rightClIdx), alpha);
        }
        else
        {
            switch (wrapMode) {
            case ETC_REPEAT:
            {
                const T last = prefixSumAccessor.template get<T, uint32_t>(lastIdx);
                const T floored = prefixSumAccessor.template get<T, uint32_t>(rightFlIdx % linearSize) + ceil(float32_t(rightFlIdx % lastIdx) / linearSize) * last;
                const T ceiled = prefixSumAccessor.template get<T, uint32_t>(rightClIdx % linearSize) + ceil(float32_t(rightClIdx % lastIdx) / linearSize) * last;
                result += lerp(floored, ceiled, alpha);
                break;
            }
            case ETC_CLAMP_TO_BORDER:
            {
                result += prefixSumAccessor.template get<T, uint32_t>(lastIdx) + (rightIdx - lastIdx) * borderColor;
                break;
            }
            case ETC_CLAMP_TO_EDGE:
            {
                const T last = prefixSumAccessor.template get<T, uint32_t>(lastIdx);
                const T lastMinusOne = prefixSumAccessor.template get<T, uint32_t>(lastIdx - 1);
                result += (rightIdx - lastIdx) * (last - lastMinusOne) + last;
                break;
            }
            case ETC_MIRROR:
            {
                const T last = prefixSumAccessor.template get<T, uint32_t>(lastIdx);
                T floored, ceiled;
                int32_t d = rightFlIdx - lastIdx;

                if (d % (2 * linearSize) == linearSize)
                    floored = ((d + linearSize) / linearSize) * last;
                else
                {
                    const uint32_t period = uint32_t(ceil(float32_t(d) / linearSize));
                    if ((period & 0x1u) == 1)
                        floored = period * last + last - prefixSumAccessor.template get<T, uint32_t>(lastIdx - uint32_t(d % linearSize));
                    else
                        floored = period * last + prefixSumAccessor.template get<T, uint32_t>((d - 1) % linearSize);
                }
                
                d = rightClIdx - lastIdx;
                if (d % (2 * linearSize) == linearSize)
                    ceiled = ((d + linearSize) / linearSize) * last;
                else
                {
                    const uint32_t period = uint32_t(ceil(float32_t(d) / linearSize));
                    if ((period & 0x1u) == 1)
                        ceiled = period * last + last - prefixSumAccessor.template get<T, uint32_t>(lastIdx - uint32_t(d % linearSize));
                    else
                        ceiled = period * last + prefixSumAccessor.template get<T, uint32_t>((d - 1) % linearSize);
                }

                result += lerp(floored, ceiled, alpha);
                break;
            }
            case ETC_MIRROR_CLAMP_TO_EDGE:
            {
                const T last = prefixSumAccessor.template get<T, uint32_t>(lastIdx);
                const T first = prefixSumAccessor.template get<T, uint32_t>(0);
                const T firstPlusOne = prefixSumAccessor.template get<T, uint32_t>(1);
                result += (rightIdx - lastIdx) * (firstPlusOne - first) + last;
                break;
            }
            }
        }

        if (leftFlIdx >= 0)
        {
            result -= lerp(prefixSumAccessor.template get<T, uint32_t>(leftFlIdx), prefixSumAccessor.template get<T, uint32_t>(leftClIdx), alpha);
        }
        else
        {
            switch (wrapMode) {
            case ETC_REPEAT:
            {
                const T last = prefixSumAccessor.template get<T, uint32_t>(lastIdx);
                const T floored = prefixSumAccessor.template get<T, uint32_t>(abs(leftFlIdx) % linearSize) + ceil(T(leftFlIdx) / linearSize) * last;
                const T ceiled = prefixSumAccessor.template get<T, uint32_t>(abs(leftClIdx) % linearSize) + ceil(float32_t(leftClIdx) / linearSize) * last;
                result -= lerp(floored, ceiled, alpha);
                break;
            }
            case ETC_CLAMP_TO_BORDER:
            {
                result -= prefixSumAccessor.template get<T, uint32_t>(0) + leftIdx * borderColor;
                break;
            }
            case ETC_CLAMP_TO_EDGE:
            {
                result -= leftIdx * prefixSumAccessor.template get<T, uint32_t>(0);
                break;
            }
            case ETC_MIRROR:
            {
                const T last = prefixSumAccessor.template get<T, uint32_t>(lastIdx);
                T floored, ceiled;

                if (abs(leftFlIdx + 1) % (2 * linearSize) == 0)
                    floored = -(abs(leftFlIdx + 1) / linearSize) * last;
                else
                {
                    const uint32_t period = uint32_t(ceil(float32_t(abs(leftFlIdx + 1)) / linearSize));
                    if ((period & 0x1u) == 1)
                        floored = -(period - 1) * last - prefixSumAccessor.template get<T, uint32_t>((abs(leftFlIdx + 1) - 1) % linearSize);
                    else
                        floored = -(period - 1) * last - (last - prefixSumAccessor.template get<T, uint32_t>((leftFlIdx + 1) % linearSize - 1));
                }

                if (leftClIdx == 0) // Special case, wouldn't be possible for `floored` above
                    ceiled = 0;
                else if (abs(leftClIdx + 1) % (2 * linearSize) == 0)
                    ceiled = -(abs(leftClIdx + 1) / linearSize) * last;
                else
                {
                    const uint32_t period = uint32_t(ceil(float32_t(abs(leftClIdx + 1)) / linearSize));
                    if ((period & 0x1u) == 1)
                        ceiled = -(period - 1) * last - prefixSumAccessor.template get<T, uint32_t>((abs(leftClIdx + 1) - 1) % linearSize);
                    else
                        ceiled = -(period - 1) * last - (last - prefixSumAccessor.template get<T, uint32_t>((leftClIdx + 1) % linearSize - 1));
                }

                result -= lerp(floored, ceiled, alpha);
                break;
            }
            case ETC_MIRROR_CLAMP_TO_EDGE:
            {
                const T last = prefixSumAccessor.template get<T, uint32_t>(lastIdx);
                const T lastMinusOne = prefixSumAccessor.template get<T, uint32_t>(lastIdx - 1);
                result -= leftIdx * (last - lastMinusOne);
                break;
            }
            }
        }

        return result;
    }
};

}
}
}