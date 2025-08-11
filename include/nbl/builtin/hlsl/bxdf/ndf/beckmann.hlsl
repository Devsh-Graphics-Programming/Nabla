// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_NDF_BECKMANN_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_NDF_BECKMANN_INCLUDED_

#include "nbl/builtin/hlsl/limits.hlsl"
#include "nbl/builtin/hlsl/bxdf/ndf/microfacet_to_light_transform.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace ndf
{

namespace beckmann_concepts
{
#define NBL_CONCEPT_NAME DG1Query
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (query, T)
NBL_CONCEPT_BEGIN(1)
#define query NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((query.getNdf()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((query.getLambdaV()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
);
#undef query
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

#define NBL_CONCEPT_NAME G2overG1Query
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (query, T)
NBL_CONCEPT_BEGIN(1)
#define query NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((query.getLambdaL()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((query.getLambdaV()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
);
#undef query
#include <nbl/builtin/hlsl/concepts/__end.hlsl>
}

template<typename T, bool IsAnisotropic=false NBL_STRUCT_CONSTRAINABLE>
struct Beckmann;

template<typename T>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<T>)
struct Beckmann<T,false NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<T>) >
{
    using scalar_type = T;

    template<class MicrofacetCache NBL_FUNC_REQUIRES(ReadableIsotropicMicrofacetCache<MicrofacetCache>)
    scalar_type D(NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        scalar_type nom = exp2<scalar_type>((cache.getNdotH2() - scalar_type(1.0)) / (log<scalar_type>(2.0) * a2 * cache.getNdotH2()));
        scalar_type denom = a2 * cache.getNdotH2() * cache.getNdotH2();
        return numbers::inv_pi<scalar_type> * nom / denom;
    }

    // brdf
    template<class Query NBL_FUNC_REQUIRES(beckmann_concepts::DG1Query<Query>)
    scalar_type DG1(NBL_CONST_REF_ARG(Query) query)
    {
        return query.getNdf() / (scalar_type(1.0) + query.getLambdaV());
    }

    // bsdf
    template<class Query, class MicrofacetCache NBL_FUNC_REQUIRES(beckmann_concepts::DG1Query<Query> && ReadableIsotropicMicrofacetCache<MicrofacetCache>)
    scalar_type DG1(NBL_CONST_REF_ARG(Query) query, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        return DG1(query);
    }

    scalar_type G1(scalar_type lambda)
    {
        return scalar_type(1.0) / (scalar_type(1.0) + lambda);
    }

    scalar_type C2(scalar_type NdotX2)
    {
        return NdotX2 / (a2 * (scalar_type(1.0) - NdotX2));
    }

    scalar_type Lambda(scalar_type c2)
    {
        scalar_type c = sqrt<scalar_type>(c2);
        scalar_type nom = scalar_type(1.0) - scalar_type(1.259) * c + scalar_type(0.396) * c2;
        scalar_type denom = scalar_type(2.181) * c2 + scalar_type(3.535) * c;
        return hlsl::mix<scalar_type>(scalar_type(0.0), nom / denom, c < scalar_type(1.6));
    }

    scalar_type LambdaC2(scalar_type NdotX2)
    {
        return Lambda(C2(NdotX2));
    }

    template<class Query, class LS, class Interaction NBL_FUNC_REQUIRES(beckmann_concepts::G2overG1Query<Query> && LightSample<LS> && surface_interactions::Isotropic<Interaction>)
    scalar_type correlated(NBL_CONST_REF_ARG(Query) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        return scalar_type(1.0) / (scalar_type(1.0) + query.getLambdaV() + query.getLambdaL());
    }

    template<class Query, class LS, class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(beckmann_concepts::G2overG1Query<Query> && LightSample<LS> && surface_interactions::Isotropic<Interaction> && ReadableIsotropicMicrofacetCache<MicrofacetCache>)
    scalar_type G2_over_G1(NBL_CONST_REF_ARG(Query) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        scalar_type onePlusLambda_V = scalar_type(1.0) + query.getLambdaV();
        return onePlusLambda_V * hlsl::mix(scalar_type(1.0)/(onePlusLambda_V + query.getLambdaL()), bxdf::beta<scalar_type>(onePlusLambda_V, scalar_type(1.0) + query.getLambdaL()), cache.isTransmission());
    }

    scalar_type a2;
};


template<typename T>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<T>)
struct Beckmann<T,true NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<T>) >
{
    using scalar_type = T;

    template<class MicrofacetCache NBL_FUNC_REQUIRES(AnisotropicMicrofacetCache<MicrofacetCache>)
    scalar_type D(NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        const scalar_type ax2 = ax*ax;
        const scalar_type ay2 = ay*ay;
        scalar_type nom = exp<scalar_type>(-(cache.getTdotH2() / ax2 + cache.getBdotH2() / ay2) / cache.getNdotH2());
        scalar_type denom = ax * ay * cache.getNdotH2() * cache.getNdotH2();
        return numbers::inv_pi<scalar_type> * nom / denom;
    }

    template<class Query NBL_FUNC_REQUIRES(beckmann_concepts::DG1Query<Query>)
    scalar_type DG1(NBL_CONST_REF_ARG(Query) query)
    {
        Beckmann<T,false> beckmann;
        scalar_type dg = beckmann.template DG1<Query>(query);
        return dg;
    }

    template<class Query, class MicrofacetCache NBL_FUNC_REQUIRES(beckmann_concepts::DG1Query<Query> && AnisotropicMicrofacetCache<MicrofacetCache>)
    scalar_type DG1(NBL_CONST_REF_ARG(Query) query, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        Beckmann<T,false> beckmann;
        scalar_type dg = beckmann.template DG1<Query, typename MicrofacetCache::isocache_type>(query, cache.iso_cache);
        return dg;
    }

    scalar_type G1(scalar_type lambda)
    {
        return scalar_type(1.0) / (scalar_type(1.0) + lambda);
    }

    scalar_type C2(scalar_type TdotX2, scalar_type BdotX2, scalar_type NdotX2)
    {
        const scalar_type ax2 = ax*ax;
        const scalar_type ay2 = ay*ay;
        return NdotX2 / (TdotX2 * ax2 + BdotX2 * ay2);
    }

    scalar_type Lambda(scalar_type c2)
    {
        scalar_type c = sqrt<scalar_type>(c2);
        scalar_type nom = scalar_type(1.0) - scalar_type(1.259) * c + scalar_type(0.396) * c2;
        scalar_type denom = scalar_type(2.181) * c2 + scalar_type(3.535) * c;
        return hlsl::mix<scalar_type>(scalar_type(0.0), nom / denom, c < scalar_type(1.6));
    }

    scalar_type LambdaC2(scalar_type TdotX2, scalar_type BdotX2, scalar_type NdotX2)
    {
        return Lambda(C2(TdotX2, BdotX2, NdotX2));
    }

    template<class Query, class LS, class Interaction NBL_FUNC_REQUIRES(beckmann_concepts::G2overG1Query<Query> && LightSample<LS> && surface_interactions::Anisotropic<Interaction>)
    scalar_type correlated(NBL_CONST_REF_ARG(Query) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        return scalar_type(1.0) / (scalar_type(1.0) + query.getLambdaV() + query.getLambdaL());
    }

    template<class Query, class LS, class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(beckmann_concepts::G2overG1Query<Query> && LightSample<LS> && surface_interactions::Anisotropic<Interaction> && AnisotropicMicrofacetCache<MicrofacetCache>)
    scalar_type G2_over_G1(NBL_CONST_REF_ARG(Query) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        scalar_type onePlusLambda_V = scalar_type(1.0) + query.getLambdaV();
        return onePlusLambda_V * hlsl::mix(scalar_type(1.0)/(onePlusLambda_V + query.getLambdaL()), bxdf::beta<scalar_type>(onePlusLambda_V, scalar_type(1.0) + query.getLambdaL()), cache.isTransmission());
    }

    scalar_type ax;
    scalar_type ay;
};

}
}
}
}

#endif
