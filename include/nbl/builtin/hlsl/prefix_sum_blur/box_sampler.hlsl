#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/enums.hlsl"
#include "nbl/builtin/hlsl/macros.h"

#ifndef _NBL_BUILTIN_BOX_SAMPLER_INCLUDED_
#define _NBL_BUILTIN_BOX_SAMPLER_INCLUDED_

namespace nbl
{
namespace hlsl
{
namespace prefix_sum_blur
{

// Requires an *inclusive* prefix sum
template<typename PrefixSumAccessor, typename T>
struct BoxSampler
{
    using prefix_sum_accessor_t = PrefixSumAccessor;

    PrefixSumAccessor prefixSumAccessor;
    uint16_t wrapMode;
    uint16_t linearSize;

    T operator()(float32_t ix, float32_t radius, float32_t borderColor)
    {
        const float32_t alpha = frac(radius);
        const float32_t rightIdx = float32_t(ix) + radius;
        const float32_t leftIdx = float32_t(ix) - radius - 1;
        const int32_t lastIdx = linearSize - 1;
        const int32_t rightFlIdx = (int32_t)floor(rightIdx);
        const int32_t rightClIdx = (int32_t)ceil(rightIdx);
        const int32_t leftFlIdx = (int32_t)floor(leftIdx);
        const int32_t leftClIdx = (int32_t)ceil(leftIdx);

        assert(linearSize > 1 && radius >= bit_cast<float32_t>(numeric_limits<float32_t>::min));
        assert(borderColor >= 0 && borderColor <= 1);

        T result = 0;
        if (rightClIdx < linearSize)
        {
            result += lerp(prefixSumAccessor.template get<T, uint32_t>(rightFlIdx), prefixSumAccessor.template get<T, uint32_t>(rightClIdx), alpha);
        }
        else
        {
            switch (wrapMode) {
            case ETC_REPEAT:
            {
                const uint32_t flooredMod = rightFlIdx % linearSize;
                const uint32_t ceiledMod = rightClIdx % linearSize;
                const T last = prefixSumAccessor.template get<T, uint32_t>(lastIdx);
                const T periodicOffset = (T(rightFlIdx) / linearSize) * last;
                const T floored = prefixSumAccessor.template get<T, uint32_t>(flooredMod);
                T ceiled = prefixSumAccessor.template get<T, uint32_t>(ceiledMod);
                if (flooredMod == lastIdx && ceiledMod == 0)
                    ceiled += last;
                result += lerp(floored, ceiled, alpha) + periodicOffset;
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
                result += (rightIdx - lastIdx) * first + last;
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
                const uint32_t flooredMod = (linearSize + leftFlIdx) % linearSize;
                const uint32_t ceiledMod = (linearSize + leftClIdx) % linearSize;
                const T last = prefixSumAccessor.template get<T, uint32_t>(lastIdx);
                const T periodicOffset = (T(linearSize + leftClIdx) / T(linearSize)) * last;
                const T floored = prefixSumAccessor.template get<T, uint32_t>(flooredMod);
                T ceiled = prefixSumAccessor.template get<T, uint32_t>(ceiledMod);
                if (flooredMod == lastIdx && ceiledMod == 0)
                    ceiled += last;
                result -= lerp(floored, ceiled, alpha) - periodicOffset;
                break;
            }
            case ETC_CLAMP_TO_BORDER:
            {
                result -= (leftIdx + 1) * borderColor;
                break;
            }
            case ETC_CLAMP_TO_EDGE:
            {
                result -= (leftIdx + 1) * prefixSumAccessor.template get<T, uint32_t>(0);
                break;
            }
            case ETC_MIRROR:
            {
                const T last = prefixSumAccessor.template get<T, uint32_t>(lastIdx);
                T floored, ceiled;

                if (leftFlIdx % (2 * linearSize) == 0)
                    floored = (T(leftFlIdx) / linearSize) * last;
                else
                {
                    const uint32_t period = uint32_t(ceil(float32_t(-leftFlIdx) / linearSize));
                    if ((period & 0x1u) == 1)
                        floored = -(period - 1) * last - prefixSumAccessor.template get<T, uint32_t>(-(leftFlIdx + 1) % linearSize);
                    else
                        floored = -(period - 1) * last - (last - prefixSumAccessor.template get<T, uint32_t>(leftFlIdx % linearSize - 1));
                }

                if (leftClIdx == 0) // Special case, wouldn't be possible for `floored` above
                    ceiled = 0;
                else if (leftClIdx % (2 * linearSize) == 0)
                    ceiled = (T(leftClIdx) / linearSize) * last;
                else
                {
                    const uint32_t period = uint32_t(ceil(float32_t(-leftClIdx) / linearSize));
                    if ((period & 0x1u) == 1)
                        ceiled = -(period - 1) * last - prefixSumAccessor.template get<T, uint32_t>(-(leftClIdx + 1) % linearSize);
                    else
                        ceiled = -(period - 1) * last - (last - prefixSumAccessor.template get<T, uint32_t>(leftClIdx % linearSize - 1));
                }

                result -= lerp(floored, ceiled, alpha);
                break;
            }
            case ETC_MIRROR_CLAMP_TO_EDGE:
            {
                const T last = prefixSumAccessor.template get<T, uint32_t>(lastIdx);
                const T lastMinusOne = prefixSumAccessor.template get<T, uint32_t>(lastIdx - 1);
                result -= (leftIdx + 1) * (last - lastMinusOne);
                break;
            }
            }
        }

        return result / (2.f * radius + 1.f);
    }
};

}
}
}

#endif