// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_COOK_TORRANCE_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_COOK_TORRANCE_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/config.hlsl"
#include "nbl/builtin/hlsl/bxdf/ndf.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{

template<class Config, class N, class F, class MLT NBL_PRIMARY_REQUIRES(config_concepts::MicrofacetConfiguration<Config> && ndf::NDF<N>) // TODO concepts for ndf and fresnel
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

    // TODO need to make sure ndf functions have the same args -> concept
    template<class Query>
    scalar_type __DG1(NBL_CONST_REF_ARG(Query) query)
    {
        return ndf.template DG1<Query>(dg1_query);
    }
    template<class Query>
    scalar_type __DG1(NBL_CONST_REF_ARG(Query) query, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        return ndf.template DG1<Query>(dg1_query, cache);
    }
    template<class Query>
    scalar_type __DG1(NBL_CONST_REF_ARG(Query) query, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        return ndf.template DG1<Query>(dg1_query, cache);
    }

    // TODO need to make sure ndf functions have the same args -> concept (query, sample, interaction)
    template<class Query>
    scalar_type __DG(NBL_CONST_REF_ARG(Query) g2_query, NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        scalar_type DG = ggx_ndf.template D<isocache_type>(cache);
        if (any<vector<bool, 2> >(A > (vector2_type)numeric_limits<scalar_type>::min))
        {
            DG *= ggx_ndf.template correlated_wo_numerator<Query, sample_type, isotropic_interaction_type>(g2_query, _sample, interaction);
        }
        return DG;
    }
    template<class Query>
    scalar_type __DG(NBL_CONST_REF_ARG(Query) g2_query, NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        scalar_type DG = ggx_ndf.template D<anisocache_type>(cache);
        if (any<vector<bool, 2> >(A > (vector2_type)numeric_limits<scalar_type>::min))
        {
            DG *= ggx_ndf.template correlated_wo_numerator<Query, sample_type, anisotropic_interaction_type>(g2_query, _sample, interaction);
        }
        return DG;
    }

    N getNDF() { return N; }
    F getFresnel() { return fresnel; }
    MLT getMicrofacetLightTransform( return microfacet_transform; )

    N ndf;
    F fresnel;
    MLT microfacet_transform;
};

}
}
}

#endif
