// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_LUMA_METER_INCLUDED_
#define _NBL_BUILTIN_HLSL_LUMA_METER_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
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

template<uint32_t SubgroupSize, uint32_t SubgroupCount, typename SharedAccessor, typename TexAccessor>
struct geom_luma_meter {
    using this_t = geom_luma_meter<SubgroupSize, SubgroupCount, SharedAccessor, TexAccessor>;

    static this_t create(NBL_REF_ARG(LumaMeteringWindow) window)
    {
        this_t retval;
        retval.window = window;
        return retval;
    }

    float32_t computeLuma(NBL_REF_ARG(TexAccessor) tex, uint32_t2 sampleCount, uint32_t2 sampleIndex, float32_t2 viewportSize)
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

    LumaMeteringWindow window;
};
}
}
}

#endif