// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_IES_TEXTURE_INCLUDED_
#define _NBL_BUILTIN_HLSL_IES_TEXTURE_INCLUDED_

#include "nbl/builtin/hlsl/ies/sampler.hlsl"
#include "nbl/builtin/hlsl/bda/struct_declare.hlsl"

namespace nbl
{
namespace hlsl
{
namespace ies
{
struct IESTextureInfo;
}
}
}

NBL_HLSL_DEFINE_STRUCT((::nbl::hlsl::ies::IESTextureInfo),
    ((lastTexelRcp, float32_t2))
    ((maxValueRecip, float32_t))
);

namespace nbl
{
namespace hlsl
{
namespace ies
{

struct SProceduralTexture
{
    using info_t = IESTextureInfo;
    using octahedral_t = math::OctahedralTransform<float32_t>;
    using polar_t = math::Polar<float32_t>;

    NBL_CONSTEXPR_STATIC_INLINE uint32_t MaxTextureWidth = 15360u;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t MaxTextureHeight = 8640u;

    NBL_CONSTEXPR_STATIC_INLINE uint32_t MinTextureWidth = 3u;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t MinTextureHeight = 3u;

    info_t info;

    static inline SProceduralTexture create(const float32_t maxCandelaValue, const uint32_t2 resolution)
    {
        SProceduralTexture retval;
        retval.info.lastTexelRcp = float32_t2(1.f, 1.f) / (float32_t2(resolution) - float32_t2(1.f, 1.f));
        retval.info.maxValueRecip = maxCandelaValue > 0.f ? (1.f / maxCandelaValue) : 0.f;
        return retval;
    }

    // NOTE: DXC fails overload resolution for templated operator() in HLSL, so we use templated __call instead.
    template<typename Accessor>
    inline float32_t __call(NBL_CONST_REF_ARG(Accessor) accessor, NBL_CONST_REF_ARG(float32_t2) uv) NBL_CONST_MEMBER_FUNC
    {
        const float32_t2 halfMinusHalfPixel = float32_t2(0.5f, 0.5f) / (float32_t2(1.f, 1.f) + info.lastTexelRcp);
        const float32_t2 ndc = (uv - float32_t2(0.5f, 0.5f)) / halfMinusHalfPixel;
        return __evalNDC(accessor, ndc);
    }

    template<typename Accessor>
    inline float32_t __call(NBL_CONST_REF_ARG(Accessor) accessor, NBL_CONST_REF_ARG(uint32_t2) coord) NBL_CONST_MEMBER_FUNC
    {
        const float32_t2 ndc = float32_t2(coord) * info.lastTexelRcp * float32_t2(2.f, 2.f) - float32_t2(1.f, 1.f);
        return __evalNDC(accessor, ndc);
    }

    template<typename Accessor>
    inline float32_t __evalNDC(NBL_CONST_REF_ARG(Accessor) accessor, NBL_CONST_REF_ARG(float32_t2) ndc) NBL_CONST_MEMBER_FUNC
    {
        // We don't currently support generating IES images that exploit symmetries or reduced domains,
        // all are full octahederal mappings of a sphere.
        // If we did, we'd rely on MIRROR and CLAMP samplers to do some of the work for us while handling the discontinuity due to corner sampling.
        const float32_t3 dir = octahedral_t::ndcToDir(ndc);
        const polar_t polar = polar_t::createFromCartesian(dir);
        CandelaSampler<Accessor> _sampler = CandelaSampler<Accessor>::create(info.lastTexelRcp);
        const float32_t intensity = _sampler(accessor, polar);
        return intensity * info.maxValueRecip;
    }
};

}
}
}

#endif // _NBL_BUILTIN_HLSL_IES_TEXTURE_INCLUDED_
