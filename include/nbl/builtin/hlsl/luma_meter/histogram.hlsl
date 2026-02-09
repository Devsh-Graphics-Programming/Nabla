// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_LUMA_METER_INCLUDED_
#define _NBL_BUILTIN_HLSL_LUMA_METER_INCLUDED_

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_arithmetic.hlsl"
#include "nbl/builtin/hlsl/workgroup2/basic.hlsl"
#include "nbl/builtin/hlsl/workgroup2/arithmetic.hlsl"
#include "nbl/builtin/hlsl/type_traits.hlsl"
#include "nbl/builtin/hlsl/morton.hlsl"
#include "nbl/builtin/hlsl/luma_meter/common.hlsl"

namespace nbl
{
namespace hlsl
{
namespace luma_meter
{

namespace impl
{
template<typename T>
struct data_proxy
{
    template<typename AccessType, typename IndexType>
    void get(const IndexType idx, NBL_REF_ARG(AccessType) value)
    {
        value = data;
    }

    template<typename AccessType, typename IndexType>
    void set(const IndexType ix, const AccessType value)
    {
        data = value;
    }

    T data;
};
}

template<class WorkgroupConfig, uint16_t BinCount, typename HistogramAccessor, typename SharedAccessor, typename TexAccessor, class device_capabilities>
struct median_meter
{
    using int_t = typename SharedAccessor::type;
    using float_t  = float32_t;
    using float_t2 = typename conditional<is_same_v<float_t, float32_t>, float32_t2, float16_t2>::type;
    using float_t3 = typename conditional<is_same_v<float_t, float32_t>, float32_t3, float16_t3>::type;

    NBL_CONSTEXPR_STATIC_INLINE uint32_t WorkgroupSize = WorkgroupConfig::WorkgroupSize;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t ScanItemsPerInvoc = WorkgroupConfig::ItemsPerInvocation_0;
    using proxy_data_t = vector<int_t, ScanItemsPerInvoc>;
    using proxy_t = impl::data_proxy<proxy_data_t>;

    using this_t = median_meter<WorkgroupConfig, BinCount, HistogramAccessor, SharedAccessor, TexAccessor, device_capabilities>;

    static this_t create(float_t lumaMin, float_t lumaMax, float_t lowerBoundPercentile, float_t upperBoundPercentile)
    {
        this_t retval;
        retval.lumaMin = lumaMin;
        retval.lumaMax = lumaMax;
        retval.lowerBoundPercentile = lowerBoundPercentile;
        retval.upperBoundPercentile = upperBoundPercentile;
        return retval;
    }

    void __inclusive_scan(NBL_REF_ARG(proxy_t) data, NBL_REF_ARG(SharedAccessor) sdata)
    {
        workgroup2::inclusive_scan< WorkgroupConfig, plus<int_t>, device_capabilities >::
            template __call<proxy_t, SharedAccessor>(data, sdata);
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
        float_t2 tileOffset
    )
    {
        uint32_t tid = workgroup::SubgroupContiguousIndex();
        
        for (uint32_t vid = tid; vid < BinCount; vid += WorkgroupSize)
            sdata.template set<uint32_t,uint32_t>(vid, 0u);

        sdata.workgroupExecutionAndMemoryBarrier();

        morton::code<false, 32, 2> mc;
        mc.value = tid;
        uint32_t2 coord = _static_cast<uint32_t2>(mc);

        float_t2 shiftedCoord = tileOffset + float32_t2(coord);
        float_t luma = __computeLuma(window, tex, shiftedCoord);

        float_t scaledLogLuma = log2(luma / lumaMin) / log2(lumaMax / lumaMin);
        uint32_t binIndex = int_t(scaledLogLuma * float_t(BinCount-1u) + 0.5);
        sdata.atomicAdd(binIndex, 1u);

        sdata.workgroupExecutionAndMemoryBarrier();

        proxy_t histogram_data;
        NBL_UNROLL for (uint32_t i = 0; i < ScanItemsPerInvoc; i++)
            sdata.template get<uint32_t,uint32_t>(tid * ScanItemsPerInvoc + i, histogram_data.data[i]);

        sdata.workgroupExecutionAndMemoryBarrier();

        __inclusive_scan(histogram_data, sdata);
        NBL_UNROLL for (uint32_t i = 0; i < ScanItemsPerInvoc; i++)
            histo.atomicAdd(tid * ScanItemsPerInvoc + i, histogram_data.data[i]);
    }

    float_t gatherLuma(
        NBL_REF_ARG(HistogramAccessor) histo,
        NBL_REF_ARG(SharedAccessor) sdata
    )
    {
        uint32_t tid = workgroup::SubgroupContiguousIndex();

        for (uint32_t vid = tid; vid < BinCount; vid += WorkgroupSize)
            sdata.template set<uint32_t,uint32_t>(vid, histo.get(vid));
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
                sdata.template get<uint32_t,uint32_t>(mid, v);
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
                sdata.template get<uint32_t,uint32_t>(mid, v);
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