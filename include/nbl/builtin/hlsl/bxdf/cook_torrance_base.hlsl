// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_COOK_TORRANCE_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_COOK_TORRANCE_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/config.hlsl"
#include "nbl/builtin/hlsl/bxdf/ndf.hlsl"
#include "nbl/builtin/hlsl/bxdf/fresnel.hlsl"
#include "nbl/builtin/hlsl/bxdf/ndf/microfacet_to_light_transform.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{

// N (NDF), F (fresnel), MT (measure transform, using DualMeasureQuant)
template<class Config, class N, class F, class MT NBL_PRIMARY_REQUIRES(config_concepts::MicrofacetConfiguration<Config> && ndf::NDF<N> && fresnel::Fresnel<F>)
struct SCookTorrance
{
    NBL_BXDF_CONFIG_ALIAS(scalar_type, Config);
    NBL_BXDF_CONFIG_ALIAS(vector2_type, Config);
    NBL_BXDF_CONFIG_ALIAS(vector3_type, Config);
    NBL_BXDF_CONFIG_ALIAS(isotropic_interaction_type, Config);
    NBL_BXDF_CONFIG_ALIAS(anisotropic_interaction_type, Config);
    NBL_BXDF_CONFIG_ALIAS(sample_type, Config);
    NBL_BXDF_CONFIG_ALIAS(isocache_type, Config);
    NBL_BXDF_CONFIG_ALIAS(anisocache_type, Config);

    scalar_type __D(NBL_CONST_REF_ARG(isocache_type) cache)
    {
        return ndf.template D<isocache_type>(cache);
    }
    scalar_type __D(NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        return ndf.template D<anisocache_type>(cache);
    }

    template<class Query>
    MT __DG1(NBL_CONST_REF_ARG(Query) query)
    {
        MT measure_transform;
        measure_transform.pdf = ndf.template DG1<Query>(query);
        return measure_transform;
    }
    template<class Query>
    MT __DG1(NBL_CONST_REF_ARG(Query) query, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        MT measure_transform;
        measure_transform.pdf = ndf.template DG1<Query>(query, cache);
        measure_transform.transmitted = cache.isTransmission();
        measure_transform.VdotH = cache.getVdotH();
        measure_transform.LdotH = cache.getLdotH();
        measure_transform.VdotHLdotH = cache.getVdotHLdotH();
        return measure_transform;
    }
    template<class Query>
    MT __DG1(NBL_CONST_REF_ARG(Query) query, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        MT measure_transform;
        measure_transform.pdf = ndf.template DG1<Query>(query, cache);
        measure_transform.transmitted = cache.isTransmission();
        measure_transform.VdotH = cache.getVdotH();
        measure_transform.LdotH = cache.getLdotH();
        measure_transform.VdotHLdotH = cache.getVdotHLdotH();
        return measure_transform;
    }

    template<class Query>
    MT __DG(NBL_CONST_REF_ARG(Query) query, NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        MT measure_transform;
        measure_transform.pdf = ndf.template D<isocache_type>(cache);
        NBL_IF_CONSTEXPR(MT::Type == ndf::MicrofacetTransformTypes::MTT_REFLECT_REFRACT)
            measure_transform.transmitted = cache.isTransmission();
        NBL_IF_CONSTEXPR(MT::Type == ndf::MicrofacetTransformTypes::MTT_REFRACT)
        {
            measure_transform.VdotH = cache.getVdotH();
            measure_transform.LdotH = cache.getLdotH();
            measure_transform.VdotHLdotH = cache.getVdotHLdotH();
        }
        if (any<vector<bool, 2> >(ndf.A > hlsl::promote<vector2_type>(numeric_limits<scalar_type>::min)))
        {
            measure_transform.pdf *= ndf.template correlated<Query, sample_type, isotropic_interaction_type>(query, _sample, interaction);
        }
        return measure_transform;
    }
    template<class Query>
    MT __DG(NBL_CONST_REF_ARG(Query) query, NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        MT measure_transform;
        measure_transform.pdf = ndf.template D<anisocache_type>(cache);
        NBL_IF_CONSTEXPR(MT::Type == ndf::MicrofacetTransformTypes::MTT_REFLECT_REFRACT)
            measure_transform.transmitted = cache.isTransmission();
        NBL_IF_CONSTEXPR(MT::Type == ndf::MicrofacetTransformTypes::MTT_REFRACT)
        {
            measure_transform.VdotH = cache.getVdotH();
            measure_transform.LdotH = cache.getLdotH();
            measure_transform.VdotHLdotH = cache.getVdotHLdotH();
        }
        if (any<vector<bool, 2> >(ndf.A > hlsl::promote<vector2_type>(numeric_limits<scalar_type>::min)))
        {
            measure_transform.pdf *= ndf.template correlated<Query, sample_type, anisotropic_interaction_type>(query, _sample, interaction);
        }
        return measure_transform;
    }

    N getNDF() { return ndf; }
    F getFresnel() { return fresnel; }

    N ndf;
    F fresnel;
};

}
}
}

#endif
