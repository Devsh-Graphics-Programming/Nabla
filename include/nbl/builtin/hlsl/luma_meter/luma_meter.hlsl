// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_LUMA_METER_INCLUDED_
#define _NBL_BUILTIN_HLSL_LUMA_METER_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "nbl/builtin/hlsl/workgroup/arithmetic.hlsl"
#include "nbl/builtin/hlsl/type_traits.hlsl"
#include "nbl/builtin/hlsl/math/morton.hlsl"
#include "nbl/builtin/hlsl/colorspace/EOTF.hlsl"
#include "nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl"

namespace nbl
{
namespace hlsl
{
namespace luma_meter
{

struct LumaMeteringWindow
{
	float32_t2 meteringWindowScale;
	float32_t2 meteringWindowOffset;
};

template<uint32_t GroupSize, typename ValueAccessor, typename SharedAccessor, typename TexAccessor>
struct geom_luma_meter {
    using this_t = geom_luma_meter<GroupSize, ValueAccessor, SharedAccessor, TexAccessor>;

    static this_t create(NBL_REF_ARG(LumaMeteringWindow) window)
    {
        this_t retval;
        retval.window = window;
        return retval;
    }

    float32_t reduction(float32_t value, NBL_REF_ARG(SharedAccessor) sdata)
    {
        return workgroup::reduction < plus < float32_t >, GroupSize >::
            template __call <SharedAccessor>(value, sdata);
    }

    float32_t computeLuma(
        NBL_REF_ARG(TexAccessor) tex,
        uint32_t2 sampleCount,
        uint32_t2 sampleIndex,
        float32_t2 viewportSize
    )
    {
        float32_t2 stride = window.meteringWindowScale / (sampleCount + float32_t2(1.0f, 1.0f));
        float32_t2 samplePos = stride * sampleIndex;
        float32_t2 uvPos = (samplePos + float32_t2(0.5f, 0.5f)) / viewportSize;
        float32_t3 color = colorspace::eotf::sRGB(tex.get(uvPos));
        float32_t luma = dot(colorspace::sRGBtoXYZ[1], color);

        const float32_t minLuma = 1.0 / 4096.0;
        const float32_t maxLuma = 32768.0;

        luma = clamp(luma, minLuma, maxLuma);

        return log2(luma / minLuma) / log2(maxLuma / minLuma);
    }

    void gatherLuma(
        NBL_REF_ARG(ValueAccessor) val,
        NBL_REF_ARG(TexAccessor) tex,
        NBL_REF_ARG(SharedAccessor) sdata,
        uint32_t2 sampleCount,
        float32_t2 viewportSize
    ) {
        uint32_t2 coord = {
            morton2d_decode_x(glsl::gl_LocalInvocationIndex()),
            morton2d_decode_y(glsl::gl_LocalInvocationIndex())
        };
        uint32_t tid = workgroup::SubgroupContiguousIndex();

        uint32_t2 sampleIndex = coord * GroupSize + float32_t2(glsl::gl_SubgroupID() + 1, glsl::gl_SubgroupInvocationID() + 1);
        float32_t luma = 0.0f;

        if (sampleIndex.x <= sampleCount.x && sampleIndex.y <= sampleCount.y) {
            luma = computeLuma(tex, sampleCount, sampleIndex, viewportSize);
            float32_t lumaSum = reduction(luma, sdata);

            sdata.workgroupExecutionAndMemoryBarrier();

            if (tid == GroupSize - 1) {
                uint32_t3 workGroupCount = glsl::gl_NumWorkGroups();
                uint32_t fixedPointBitsLeft = 32 - uint32_t(ceil(log2(workGroupCount.x * workGroupCount.y * workGroupCount.z))) + glsl::gl_SubgroupSizeLog2();
                uint32_t lumaSumBitPattern = uint32_t(clamp(lumaSum, 0.f, float((1 << fixedPointBitsLeft) - 1)));
                uint32_t3 workgroupSize = glsl::gl_WorkGroupSize();
                uint32_t workgroupIndex = dot(uint32_t3(workgroupSize.y * workgroupSize.z, workgroupSize.z, 1), glsl::gl_WorkGroupID());

                val.atomicAdd(workgroupIndex & ((1 << glsl::gl_SubgroupSizeLog2()) - 1), lumaSumBitPattern);
            }
        }
    }

    LumaMeteringWindow window;
};
}
}
}

#endif