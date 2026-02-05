// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_LUMA_METER_INCLUDED_
#define _NBL_BUILTIN_HLSL_LUMA_METER_INCLUDED_

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_arithmetic.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "nbl/builtin/hlsl/workgroup/arithmetic.hlsl"
#include "nbl/builtin/hlsl/type_traits.hlsl"
#include "nbl/builtin/hlsl/morton.hlsl"
#include "nbl/builtin/hlsl/luma_meter/common.hlsl"

namespace nbl
{
namespace hlsl
{
namespace luma_meter
{

template<uint32_t WorkgroupSize, uint16_t BinCount, typename HistogramAccessor, typename SharedAccessor, typename TexAccessor>
struct median_meter
{
    using int_t = typename SharedAccessor::type;
    using float_t  = float32_t;
    using float_t2 = typename conditional<is_same_v<float_t, float32_t>, float32_t2, float16_t2>::type;
    using float_t3 = typename conditional<is_same_v<float_t, float32_t>, float32_t3, float16_t3>::type;
    using this_t = median_meter<WorkgroupSize, BinCount, HistogramAccessor, SharedAccessor, TexAccessor>;

    static this_t create(float_t lumaMin, float_t lumaMax, float_t lowerBoundPercentile, float_t upperBoundPercentile)
    {
        this_t retval;
        retval.lumaMin = lumaMin;
        retval.lumaMax = lumaMax;
        retval.lowerBoundPercentile = lowerBoundPercentile;
        retval.upperBoundPercentile = upperBoundPercentile;
        return retval;
    }

    int_t __inclusive_scan(float_t value, NBL_REF_ARG(SharedAccessor) sdata)
    {
        return workgroup::inclusive_scan < plus < int_t >, WorkgroupSize >::
            template __call <SharedAccessor>(value, sdata);
    }

    float_t __computeLuma(
        NBL_CONST_REF_ARG(MeteringWindow) window,
        NBL_REF_ARG(TexAccessor) tex,
        float_t2 shiftedCoord
    )
    {
        float_t2 uvPos = shiftedCoord * window.meteringWindowScale + window.meteringWindowOffset;
        float_t3 color = tex.get(uvPos);
        float_t luma = (float_t)TexAccessor::toXYZ(color);

        return clamp(luma, lumaMin, lumaMax);
    }

    void sampleLuma(
        NBL_CONST_REF_ARG(MeteringWindow) window,
        NBL_REF_ARG(HistogramAccessor) histo,
        NBL_REF_ARG(TexAccessor) tex,
        NBL_REF_ARG(SharedAccessor) sdata,
        float_t2 tileOffset,
        float_t2 viewportSize
    )
    {
        uint32_t tid = workgroup::SubgroupContiguousIndex();
        
        for (uint32_t vid = tid; vid < BinCount; vid += WorkgroupSize) {
            sdata.set(vid, 0);
        }

        sdata.workgroupExecutionAndMemoryBarrier();

        morton::code<false, 32, 2> mc;
        mc.value = tid;
        uint32_t2 coord = _static_cast<uint32_t2>(mc);

        float_t2 shiftedCoord = (tileOffset + (float32_t2)(coord)) / viewportSize;
        float_t luma = __computeLuma(window, tex, shiftedCoord);

        float_t scaledLogLuma = log2(luma / lumaMin) / log2(lumaMax / lumaMin);
        uint32_t binIndex = int_t(scaledLogLuma * float_t(BinCount-1u) + 0.5);
        sdata.atomicAdd(binIndex, 1u);

        sdata.workgroupExecutionAndMemoryBarrier();

        int_t histogram_value;
        sdata.get(tid, histogram_value);

        sdata.workgroupExecutionAndMemoryBarrier();

        int_t sum = __inclusive_scan(histogram_value, sdata);
        histo.atomicAdd(tid, sum);
    }

    float_t gatherLuma(
        NBL_REF_ARG(HistogramAccessor) histo,
        NBL_REF_ARG(SharedAccessor) sdata
    )
    {
        uint32_t tid = workgroup::SubgroupContiguousIndex();

        for (uint32_t vid = tid; vid < BinCount; vid += WorkgroupSize) {
            sdata.set(
                vid,
                histo.get(vid)
            );
        }
        sdata.workgroupExecutionAndMemoryBarrier();

        int_t lower, upper;
        if (tid == 0)
        {
            const uint32_t lowerPercentile = uint32_t(BinCount * lowerBoundPercentile);
            uint32_t lo = 0u;
            uint32_t hi = BinCount;
            int_t v;
            while (lo < hi)
            {
                uint32_t mid = lo + (hi - lo) / 2;
                sdata.get(mid, v);
                if (lowerPercentile <= v)
                    hi = mid;
                else
                    lo = mid + 1;
            }

            lower = lo;
        }
        if (tid == 1)
        {
            const uint32_t upperPercentile = uint32_t(BinCount * upperBoundPercentile);
            uint32_t lo = 0u;
            uint32_t hi = BinCount;
            int_t v;
            while (lo < hi)
            {
                uint32_t mid = lo + (hi - lo) / 2;
                sdata.get(mid, v);
                if (upperPercentile >= v)
                    lo = mid + 1;
                else
                    hi = mid;
            }

            upper = lo;
        }
        sdata.workgroupExecutionAndMemoryBarrier();

        lower = workgroup::Broadcast(lower, sdata, 0);
        upper = workgroup::Broadcast(upper, sdata, 1);

        return ((float_t(lower) + float_t(upper)) * 0.5 / float_t(BinCount-1u)) * log2(lumaMax/lumaMin) + log2(lumaMin);
    }

    float_t lumaMin;
    float_t lumaMax;
    float_t lowerBoundPercentile;
    float_t upperBoundPercentile;
};

}
}
}

#endif