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

// TODO(?): should be in nbl::hlsl::ies (or in the Texutre struct) but I get
// error GA3909C62: class template specialization of 'member_count' not in a namespace enclosing 'bda'
// which I don't want to deal with rn to not (eventually) break stuff

struct IESTextureInfo;
NBL_HLSL_DEFINE_STRUCT((IESTextureInfo),
	((inv, float32_t2))
	((flatten, float32_t))
	((maxValueRecip, float32_t))
	((flattenTarget, float32_t))
	((domainLo, float32_t))
	((domainHi, float32_t))
	((fullDomainFlatten, uint16_t)) // bool
);

namespace ies
{

template<typename Accessor NBL_FUNC_REQUIRES(concepts::IsIESAccessor<Accessor>)
struct Texture
{
    using accessor_t = Accessor;
    using value_t = typename accessor_t::value_t;
	using sampler_t = CandelaSampler<accessor_t>;
	using polar_t = math::Polar<float32_t>;
	using octahedral_t = math::OctahedralTransform<float32_t>;
	using SInfo = nbl::hlsl::IESTextureInfo;

    static inline SInfo createInfo(NBL_CONST_REF_ARG(accessor_t) accessor, NBL_CONST_REF_ARG(uint32_t2) size, float32_t flatten, bool fullDomainFlatten)
    {
        SInfo retval;
		const ProfileProperties props = accessor.getProperties();

        // There is one huge issue, the IES files love to give us values for degrees 0, 90, 180 an 360
        // So standard octahedral mapping won't work, because for above data points you need corner sampled images.

        retval.inv = float32_t2(1.f, 1.f) / float32_t2(size - 1u);
        retval.flatten = flatten;
        retval.maxValueRecip = 1.0f / props.maxCandelaValue; // Late Optimization TODO: Modify the Max Value for the UNORM texture to be the Max Value after flatten blending
        retval.domainLo = radians(accessor.vAngle(0u));
        retval.domainHi = radians(accessor.vAngle(accessor.vAnglesCount() - 1u));
        retval.fullDomainFlatten = fullDomainFlatten;

		if(fullDomainFlatten)
			retval.flattenTarget = props.fullDomainAvgEmission;
		else
			retval.flattenTarget = props.avgEmmision;

        return retval;
    }

	static inline float32_t eval(NBL_CONST_REF_ARG(accessor_t) accessor, NBL_CONST_REF_ARG(SInfo) info, NBL_CONST_REF_ARG(float32_t2) uv)
    {
	    // We don't currently support generating IES images that exploit symmetries or reduced domains, all are full octahederal mappings of a sphere.
		// If we did, we'd rely on MIRROR and CLAMP samplers to do some of the work for us while handling the discontinuity due to corner sampling. 
        const float32_t3 dir = octahedral_t::uvToDir(uv);
        const polar_t polar = polar_t::createFromCartesian(dir);

		sampler_t sampler;
        const float32_t intensity = sampler.sample(accessor, polar);

		//! blend the IES texture with "flatten"
        float32_t blendV = intensity * (1.f - info.flatten);

        const bool inDomain = (info.domainLo <= polar.theta) && (polar.theta <= info.domainHi);

        if ((info.fullDomainFlatten && inDomain) || intensity > 0.0f)
            blendV += info.flattenTarget * info.flatten;

        blendV *= info.maxValueRecip;

        return blendV;
    }

	static inline float32_t eval(NBL_CONST_REF_ARG(accessor_t) accessor, NBL_CONST_REF_ARG(SInfo) info, NBL_CONST_REF_ARG(uint32_t2) position)
    {
        const float32_t2 uv = float32_t2(position) * info.inv;
        return eval(accessor, info, uv);
    }
};

}
}
}

#endif // _NBL_BUILTIN_HLSL_IES_TEXTURE_INCLUDED_
