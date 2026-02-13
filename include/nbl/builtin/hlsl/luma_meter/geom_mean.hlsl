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

    T data;
};
}

template<class WorkgroupConfig, typename ValueAccessor, typename SharedAccessor, typename TexAccessor, class device_capabilities>
struct geom_meter
{
    using float_t = typename SharedAccessor::type;
    using float_t2 = typename conditional<is_same_v<float_t, float32_t>, float32_t2, float16_t2>::type;
    using float_t3 = typename conditional<is_same_v<float_t, float32_t>, float32_t3, float16_t3>::type;
    
    using proxy_data_t = float_t;
    using proxy_t = impl::data_proxy<proxy_data_t>;

    NBL_CONSTEXPR_STATIC_INLINE uint32_t WorkgroupSize = WorkgroupConfig::WorkgroupSize;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t SubgroupSize = WorkgroupConfig::SubgroupSize;
    using this_t = geom_meter<WorkgroupConfig, ValueAccessor, SharedAccessor, TexAccessor, device_capabilities>;

    NBL_CONSTEXPR_STATIC_INLINE uint32_t MaxWorkgroupIncrement = 0x1000u;

    static this_t create(float_t lumaMin, float_t lumaMax, float_t rcpFirstPassWGCount)
    {
        this_t retval;
        retval.lumaMin = lumaMin;
        retval.lumaMax = lumaMax;
        retval.log2LumaMin = log2(lumaMin);
        retval.log2LumaRange = log2(lumaMax) - retval.log2LumaMin;
        retval.rcpFirstPassWGCount = rcpFirstPassWGCount;
        return retval;
    }

    float_t __reduction(NBL_REF_ARG(proxy_t) data, NBL_REF_ARG(SharedAccessor) sdata)
    {
        return workgroup2::reduction< WorkgroupConfig, plus<float_t>, device_capabilities >::
            template __call<proxy_t, SharedAccessor>(data, sdata);
    }

    float_t __computeLumaLog2(
        NBL_CONST_REF_ARG(MeteringWindow) window,
        NBL_REF_ARG(TexAccessor) tex,
        const float_t2 shiftedCoord
    )
    {
        const float_t2 uvPos = shiftedCoord * window.meteringWindowScale + window.meteringWindowOffset;
        const float_t3 color = tex.get(uvPos);
        float_t luma = TexAccessor::toXYZ(color);

        luma = clamp(luma, lumaMin, lumaMax);

        return log2(luma);
    }

    void __uploadFloat(
        NBL_REF_ARG(ValueAccessor) val_accessor,
        float_t val
    )
    {
        const uint32_t3 workGroupCount = glsl::gl_NumWorkGroups();
        const uint32_t3 workgroupID = glsl::gl_WorkGroupID();
        const uint32_t index = (workgroupID.y * workGroupCount.x + workgroupID.x) & (SubgroupSize - 1u);
        const uint32_t lumaVal = uint32_t(val / float_t(WorkgroupSize) * float_t(MaxWorkgroupIncrement) + 0.5);
        val_accessor.atomicAdd(index, lumaVal);
    }

    float_t __downloadFloat(
        NBL_REF_ARG(ValueAccessor) val_accessor,
        uint32_t index
    )
    {
        uint32_t lumaVal = val_accessor.get(index);
        lumaVal = glsl::subgroupAdd(lumaVal);
        return float_t(lumaVal) / float_t(MaxWorkgroupIncrement) * rcpFirstPassWGCount * log2LumaRange + log2LumaMin;
    }

    void sampleLuma(
        NBL_CONST_REF_ARG(MeteringWindow) window,
        NBL_REF_ARG(ValueAccessor) val,
        NBL_REF_ARG(TexAccessor) tex,
        NBL_REF_ARG(SharedAccessor) sdata
    )
    {
        uint32_t tid = workgroup::SubgroupContiguousIndex();
        morton::code<false, 32, 2> mc;
        mc.value = tid;
        uint32_t2 coord = _static_cast<uint32_t2>(mc);

        const float_t2 tileOffset = float32_t2((glsl::gl_WorkGroupID() * SubgroupSize).xy);
        const float_t2 shiftedCoord = tileOffset + float32_t2(coord);
        float_t lumaLog2 = __computeLumaLog2(window, tex, shiftedCoord);
        lumaLog2 = (lumaLog2 - log2LumaMin) / log2LumaRange;

        proxy_t data;
        data.data = lumaLog2;
        float_t lumaLog2Sum = __reduction(data, sdata);

        if (tid == 0) {
            __uploadFloat(
                val,
                lumaLog2Sum
            );
        }
    }

    float_t gatherLuma(
        NBL_REF_ARG(ValueAccessor) val
    )
    {
        const uint32_t tid = glsl::gl_SubgroupInvocationID();
        const float_t luma = __downloadFloat(
                val,
                tid
            );

        return luma;
    }

    float_t lumaMin;
    float_t lumaMax;
    float_t log2LumaMin;
    float_t log2LumaRange;
    float_t rcpFirstPassWGCount;
};

}
}
}

#endif