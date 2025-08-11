// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_NDF_GGX_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_NDF_GGX_INCLUDED_

#include "nbl/builtin/hlsl/limits.hlsl"
#include "nbl/builtin/hlsl/bxdf/ndf.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace ndf
{

namespace ggx_concepts
{
#define NBL_CONCEPT_NAME DG1BrdfQuery
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (query, T)
NBL_CONCEPT_BEGIN(1)
#define query NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((query.getNdf()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((query.getG1over2NdotV()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
);
#undef query
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

#define NBL_CONCEPT_NAME DG1BsdfQuery
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (query, T)
NBL_CONCEPT_BEGIN(1)
#define query NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((query.getNdf()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((query.getG1over2NdotV()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((query.getOrientedEta()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
);
#undef query
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

#define NBL_CONCEPT_NAME G2XQuery
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (query, T)
NBL_CONCEPT_BEGIN(1)
#define query NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((query.getDevshV()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((query.getDevshL()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((query.getClampMode()), ::nbl::hlsl::is_same_v, BxDFClampMode))
);
#undef query
#include <nbl/builtin/hlsl/concepts/__end.hlsl>
}

template<typename T, bool IsAnisotropic=false NBL_STRUCT_CONSTRAINABLE>
struct GGX;

template<typename T>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<T>)
struct GGX<T,false NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<T>) >
{
    using scalar_type = T;
    using this_t = GGX<T,false>;

    // trowbridge-reitz
    template<class MicrofacetCache NBL_FUNC_REQUIRES(ReadableIsotropicMicrofacetCache<MicrofacetCache>)
    scalar_type D(NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        scalar_type denom = scalar_type(1.0) - one_minus_a2 * cache.getNdotH2();
        return a2 * numbers::inv_pi<scalar_type> / (denom * denom);
    }

    template<class Query NBL_FUNC_REQUIRES(ggx_concepts::DG1BrdfQuery<Query>)
    scalar_type DG1(NBL_REF_ARG(Query) query)
    {
        return scalar_type(0.5) * query.getNdf() * query.getG1over2NdotV();
    }

    template<class Query, class MicrofacetCache NBL_FUNC_REQUIRES(ggx_concepts::DG1BsdfQuery<Query> && ReadableIsotropicMicrofacetCache<MicrofacetCache>)
    scalar_type DG1(NBL_CONST_REF_ARG(Query) query, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        scalar_type NG = query.getNdf() * query.getG1over2NdotV();
        scalar_type factor = scalar_type(0.5);
        if (cache.isTransmission())
        {
            const scalar_type VdotH_etaLdotH = (cache.getVdotH() + query.getOrientedEta() * cache.getLdotH());
            // VdotHLdotH is negative under transmission, so this factor is negative
            factor *= -scalar_type(2.0) * cache.getVdotHLdotH() / (VdotH_etaLdotH * VdotH_etaLdotH);
        }
        return NG * factor;
    }

    scalar_type devsh_part(scalar_type NdotX2)
    {
        assert(a2 >= numeric_limits<scalar_type>::min);
        return sqrt(a2 + one_minus_a2 * NdotX2);
    }

    scalar_type G1_wo_numerator(scalar_type absNdotX, scalar_type NdotX2)
    {
        return scalar_type(1.0) / (absNdotX + devsh_part(NdotX2));
    }

    scalar_type G1_wo_numerator_devsh_part(scalar_type absNdotX, scalar_type devsh_part)
    {
        // numerator is 2 * NdotX
        return scalar_type(1.0) / (absNdotX + devsh_part);
    }

    // without numerator, numerator is 2 * NdotV * NdotL, we factor out 4 * NdotV * NdotL, hence 0.5
    template<class Query, class LS, class Interaction NBL_FUNC_REQUIRES(ggx_concepts::G2XQuery<Query> && LightSample<LS> && surface_interactions::Isotropic<Interaction>)
    scalar_type correlated(NBL_CONST_REF_ARG(Query) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        BxDFClampMode _clamp = query.getClampMode();
        assert(_clamp != BxDFClampMode::BCM_NONE);

        scalar_type Vterm = _sample.getNdotL(_clamp) * query.getDevshV();
        scalar_type Lterm = interaction.getNdotV(_clamp) * query.getDevshL();
        return scalar_type(0.5) / (Vterm + Lterm);
    }

    template<class Query, class LS, class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(ggx_concepts::G2XQuery<Query> && LightSample<LS> && surface_interactions::Isotropic<Interaction> && ReadableIsotropicMicrofacetCache<MicrofacetCache>)
    scalar_type G2_over_G1(NBL_CONST_REF_ARG(Query) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        BxDFClampMode _clamp = query.getClampMode();
        assert(_clamp != BxDFClampMode::BCM_NONE);

        scalar_type G2_over_G1;
        scalar_type NdotV = interaction.getNdotV(_clamp);
        scalar_type NdotL = _sample.getNdotL(_clamp);
        scalar_type devsh_v = query.getDevshV();
        scalar_type devsh_l = query.getDevshL();
        if (cache.isTransmission())
        {
            if (NdotV < 1e-7 || NdotL < 1e-7)
                return 0.0;
            scalar_type onePlusLambda_V = scalar_type(0.5) * (devsh_v / NdotV + scalar_type(1.0));
            scalar_type onePlusLambda_L = scalar_type(0.5) * (devsh_l / NdotL + scalar_type(1.0));
            G2_over_G1 = bxdf::beta<scalar_type>(onePlusLambda_L, onePlusLambda_V) * onePlusLambda_V;
        }
        else
        {
            G2_over_G1 = NdotL * (devsh_v + NdotV); // alternative `Vterm+NdotL*NdotV /// NdotL*NdotV could come as a parameter
            G2_over_G1 /= NdotV * devsh_l + NdotL * devsh_v;
        }

        return G2_over_G1;
    }

    scalar_type a2;
    scalar_type one_minus_a2;
};

template<typename T>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<T>)
struct GGX<T,true NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<T>) >
{
    using scalar_type = T;

    template<class MicrofacetCache NBL_FUNC_REQUIRES(AnisotropicMicrofacetCache<MicrofacetCache>)
    scalar_type D(NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        scalar_type denom = cache.getTdotH2() / ax2 + cache.getBdotH2() / ay2 + cache.getNdotH2();
        return numbers::inv_pi<scalar_type> / (a2 * denom * denom);
    }

    // burley
    scalar_type D(scalar_type a2, scalar_type TdotH, scalar_type BdotH, scalar_type NdotH, scalar_type anisotropy)
    {
        scalar_type antiAniso = scalar_type(1.0) - anisotropy;
        scalar_type atab = a2 * antiAniso;
        scalar_type anisoTdotH = antiAniso * TdotH;
        scalar_type anisoNdotH = antiAniso * NdotH;
        scalar_type w2 = antiAniso/(BdotH * BdotH + anisoTdotH * anisoTdotH + anisoNdotH * anisoNdotH * a2);
        return w2 * w2 * atab * numbers::inv_pi<scalar_type>;
    }

    template<class Query NBL_FUNC_REQUIRES(ggx_concepts::DG1BrdfQuery<Query>)
    scalar_type DG1(NBL_REF_ARG(Query) query)
    {
        GGX<T,false> ggx;
        return ggx.template DG1<Query>(query);
    }

    template<class Query, class MicrofacetCache NBL_FUNC_REQUIRES(ggx_concepts::DG1BsdfQuery<Query> && AnisotropicMicrofacetCache<MicrofacetCache>)
    scalar_type DG1(NBL_CONST_REF_ARG(Query) query, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        GGX<T,false> ggx;
        return ggx.template DG1<Query, typename MicrofacetCache::isocache_type>(query, cache.iso_cache);
    }

    scalar_type devsh_part(scalar_type TdotX2, scalar_type BdotX2, scalar_type NdotX2)
    {
        assert(ax2 >= numeric_limits<scalar_type>::min && ay2 >= numeric_limits<scalar_type>::min);
        return sqrt(TdotX2 * ax2 + BdotX2 * ay2 + NdotX2);
    }

    scalar_type G1_wo_numerator(scalar_type NdotX, scalar_type TdotX2, scalar_type BdotX2, scalar_type NdotX2)
    {
        return scalar_type(1.0) / (NdotX + devsh_part(TdotX2, BdotX2, NdotX2));
    }

    scalar_type G1_wo_numerator_devsh_part(scalar_type NdotX, scalar_type devsh_part)
    {
        return scalar_type(1.0) / (NdotX + devsh_part);
    }

    // without numerator
    template<class Query, class LS, class Interaction NBL_FUNC_REQUIRES(ggx_concepts::G2XQuery<Query> && LightSample<LS> && surface_interactions::Anisotropic<Interaction>)
    scalar_type correlated(NBL_CONST_REF_ARG(Query) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        BxDFClampMode _clamp = query.getClampMode();
        assert(_clamp != BxDFClampMode::BCM_NONE);

        scalar_type Vterm = _sample.getNdotL(_clamp) * query.getDevshV();
        scalar_type Lterm = interaction.getNdotV(_clamp) * query.getDevshL();
        return scalar_type(0.5) / (Vterm + Lterm);
    }

    template<class Query, class LS, class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(ggx_concepts::G2XQuery<Query> && LightSample<LS> && surface_interactions::Anisotropic<Interaction> && AnisotropicMicrofacetCache<MicrofacetCache>)
    scalar_type G2_over_G1(NBL_CONST_REF_ARG(Query) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        BxDFClampMode _clamp = query.getClampMode();
        assert(_clamp != BxDFClampMode::BCM_NONE);

        scalar_type G2_over_G1;
        scalar_type NdotV = interaction.getNdotV(_clamp);
        scalar_type NdotL = _sample.getNdotL(_clamp);
        scalar_type devsh_v = query.getDevshV();
        scalar_type devsh_l = query.getDevshL();
        if (cache.isTransmission())
        {
            if (NdotV < 1e-7 || NdotL < 1e-7)
                return 0.0;
            scalar_type onePlusLambda_V = scalar_type(0.5) * (devsh_v / NdotV + scalar_type(1.0));
            scalar_type onePlusLambda_L = scalar_type(0.5) * (devsh_l / NdotL + scalar_type(1.0));
            G2_over_G1 = bxdf::beta<scalar_type>(onePlusLambda_L, onePlusLambda_V) * onePlusLambda_V;
        }
        else
        {
            G2_over_G1 = NdotL * (devsh_v + NdotV);
            G2_over_G1 /= NdotV * devsh_l + NdotL * devsh_v;
        }

        return G2_over_G1;
    }

    scalar_type ax2;
    scalar_type ay2;
    scalar_type a2;
};


namespace impl
{
template<class T, class U>
struct is_ggx : bool_constant<
    is_same<T, GGX<U,false> >::value ||
    is_same<T, GGX<U,true> >::value
> {};
}

template<class T> 
struct is_ggx : impl::is_ggx<T, typename T::scalar_type> {};

template<typename T>
NBL_CONSTEXPR bool is_ggx_v = is_ggx<T>::value;

}
}
}
}

#endif
