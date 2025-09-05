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

// namespace beckmann_concepts
// {
// #define NBL_CONCEPT_NAME DG1BrdfQuery
// #define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
// #define NBL_CONCEPT_TPLT_PRM_NAMES (T)
// #define NBL_CONCEPT_PARAM_0 (query, T)
// NBL_CONCEPT_BEGIN(1)
// #define query NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
// NBL_CONCEPT_END(
//     ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
//     ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((query.getNdf()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
//     ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((query.getLambdaV()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
// );
// #undef query
// #include <nbl/builtin/hlsl/concepts/__end.hlsl>

// #define NBL_CONCEPT_NAME DG1BsdfQuery
// #define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
// #define NBL_CONCEPT_TPLT_PRM_NAMES (T)
// #define NBL_CONCEPT_PARAM_0 (query, T)
// NBL_CONCEPT_BEGIN(1)
// #define query NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
// NBL_CONCEPT_END(
//     ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
//     ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(DG1BrdfQuery, T))
//     ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((query.getOrientedEta()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
// );
// #undef query
// #include <nbl/builtin/hlsl/concepts/__end.hlsl>

// #define NBL_CONCEPT_NAME G2overG1Query
// #define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
// #define NBL_CONCEPT_TPLT_PRM_NAMES (T)
// #define NBL_CONCEPT_PARAM_0 (query, T)
// NBL_CONCEPT_BEGIN(1)
// #define query NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
// NBL_CONCEPT_END(
//     ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
//     ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((query.getLambdaL()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
//     ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((query.getLambdaV()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
// );
// #undef query
// #include <nbl/builtin/hlsl/concepts/__end.hlsl>
// }

namespace impl
{

template<typename T>
struct SBeckmannDG1Query    // TODO: need to specialize? or just ignore orientedEta if not needed
{
    using scalar_type = T;

    scalar_type getNdf() NBL_CONST_MEMBER_FUNC { return ndf; }
    scalar_type getLambdaV() NBL_CONST_MEMBER_FUNC { return lambda_V; }

    scalar_type ndf;
    scalar_type lambda_V;
};

template<typename T>
struct SBeckmannG2overG1Query
{
    using scalar_type = T;

    scalar_type getLambdaL() NBL_CONST_MEMBER_FUNC { return lambda_L; }
    scalar_type getLambdaV() NBL_CONST_MEMBER_FUNC { return lambda_V; }

    scalar_type lambda_L;
    scalar_type lambda_V;
};

template<typename T>
struct SQuantQuery
{
    using scalar_type = T;

    scalar_type getVdotHLdotH() NBL_CONST_MEMBER_FUNC { return VdotHLdotH; }
    scalar_type getVdotH_etaLdotH() NBL_CONST_MEMBER_FUNC { return VdotH_etaLdotH; }

    scalar_type VdotHLdotH;
    scalar_type VdotH_etaLdotH;
};

template<typename T, bool IsAnisotropic=false NBL_STRUCT_CONSTRAINABLE>
struct BeckmannCommon;

template<typename T>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<T>)
struct BeckmannCommon<T,false NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<T>) >
{
    using scalar_type = T;
    using dg1_query_type = SBeckmannDG1Query<scalar_type>;
    using g2g1_query_type = SBeckmannG2overG1Query<scalar_type>;

    template<class MicrofacetCache NBL_FUNC_REQUIRES(ReadableIsotropicMicrofacetCache<MicrofacetCache>)
    scalar_type D(NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        scalar_type nom = exp2<scalar_type>((cache.getNdotH2() - scalar_type(1.0)) / (log<scalar_type>(2.0) * a2 * cache.getNdotH2()));
        scalar_type denom = a2 * cache.getNdotH2() * cache.getNdotH2();
        return numbers::inv_pi<scalar_type> * nom / denom;
    }

    scalar_type DG1(NBL_CONST_REF_ARG(dg1_query_type) query)
    {
        return query.getNdf() / (scalar_type(1.0) + query.getLambdaV());
    }

    static scalar_type C2(scalar_type NdotX2)
    {
        return NdotX2 / (a2 * (scalar_type(1.0) - NdotX2));
    }

    static scalar_type Lambda(scalar_type c2)
    {
        scalar_type c = sqrt<scalar_type>(c2);
        scalar_type nom = scalar_type(1.0) - scalar_type(1.259) * c + scalar_type(0.396) * c2;
        scalar_type denom = scalar_type(2.181) * c2 + scalar_type(3.535) * c;
        return hlsl::mix<scalar_type>(scalar_type(0.0), nom / denom, c < scalar_type(1.6));
    }

    static scalar_type LambdaC2(scalar_type NdotX2)
    {
        return Lambda(C2(NdotX2));
    }

    template<class LS, class Interaction NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Isotropic<Interaction>)
    scalar_type correlated(NBL_CONST_REF_ARG(g2g1_query_type) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        return scalar_type(1.0) / (scalar_type(1.0) + query.getLambdaV() + query.getLambdaL());
    }

    template<class LS, class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Isotropic<Interaction> && ReadableIsotropicMicrofacetCache<MicrofacetCache>)
    scalar_type G2_over_G1(NBL_CONST_REF_ARG(g2g1_query_type) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        scalar_type onePlusLambda_V = scalar_type(1.0) + query.getLambdaV();
        return onePlusLambda_V * hlsl::mix(scalar_type(1.0)/(onePlusLambda_V + query.getLambdaL()), bxdf::beta<scalar_type>(onePlusLambda_V, scalar_type(1.0) + query.getLambdaL()), cache.isTransmission());
    }

    vector<scalar_type, 2> A;   // TODO: remove?
    scalar_type a2;
};

template<typename T>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<T>)
struct BeckmannCommon<T,true NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<T>) >
{
    using scalar_type = T;
    using dg1_query_type = SBeckmannDG1Query<scalar_type>;
    using g2g1_query_type = SBeckmannG2overG1Query<scalar_type>;

    template<class MicrofacetCache NBL_FUNC_REQUIRES(AnisotropicMicrofacetCache<MicrofacetCache>)
    scalar_type D(NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        scalar_type nom = exp<scalar_type>(-(cache.getTdotH2() / ax2 + cache.getBdotH2() / ay2) / cache.getNdotH2());
        scalar_type denom = A.x * A.y * cache.getNdotH2() * cache.getNdotH2();
        return numbers::inv_pi<scalar_type> * nom / denom;
    }

    scalar_type DG1(NBL_CONST_REF_ARG(dg1_query_type) query)
    {
        BeckmannCommon<T,false> beckmann;
        scalar_type dg = beckmann.DG1(query);
        return dg;
    }

    static scalar_type C2(scalar_type TdotX2, scalar_type BdotX2, scalar_type NdotX2)
    {
        return NdotX2 / (TdotX2 * ax2 + BdotX2 * ay2);
    }

    static scalar_type Lambda(scalar_type c2)
    {
        scalar_type c = sqrt<scalar_type>(c2);
        scalar_type nom = scalar_type(1.0) - scalar_type(1.259) * c + scalar_type(0.396) * c2;
        scalar_type denom = scalar_type(2.181) * c2 + scalar_type(3.535) * c;
        return hlsl::mix<scalar_type>(scalar_type(0.0), nom / denom, c < scalar_type(1.6));
    }

    static scalar_type LambdaC2(scalar_type TdotX2, scalar_type BdotX2, scalar_type NdotX2)
    {
        return Lambda(C2(TdotX2, BdotX2, NdotX2));
    }

    template<class LS, class Interaction NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Anisotropic<Interaction>)
    scalar_type correlated(NBL_CONST_REF_ARG(g2g1_query_type) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        return scalar_type(1.0) / (scalar_type(1.0) + query.getLambdaV() + query.getLambdaL());
    }

    template<class LS, class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Anisotropic<Interaction> && AnisotropicMicrofacetCache<MicrofacetCache>)
    scalar_type G2_over_G1(NBL_CONST_REF_ARG(g2g1_query_type) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        scalar_type onePlusLambda_V = scalar_type(1.0) + query.getLambdaV();
        return onePlusLambda_V * hlsl::mix(scalar_type(1.0)/(onePlusLambda_V + query.getLambdaL()), bxdf::beta<scalar_type>(onePlusLambda_V, scalar_type(1.0) + query.getLambdaL()), cache.isTransmission());
    }

    vector<scalar_type, 2> A;
    scalar_type ax2;
    scalar_type ay2;
};
}

template<typename T, bool IsAnisotropic, MicrofacetTransformTypes reflect_refract NBL_STRUCT_CONSTRAINABLE>
struct Beckmann;

// partial spec for brdf
template<typename T>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<T>)
struct Beckmann<T,false,MTT_REFLECT NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<T>) >
{
    using scalar_type = T;
    using base_type = impl::BeckmannCommon<T,false>;
    using quant_type = SDualMeasureQuant<scalar_type>;

    using dg1_query_type = typename base_type::dg1_query_type;
    using g2g1_query_type = typename base_type::g2g1_query_type;
    using quant_query_type = SQuantQuery<scalar_type>;

    template<class MicrofacetCache NBL_FUNC_REQUIRES(ReadableIsotropicMicrofacetCache<MicrofacetCache>)
    quant_query_type createQuantQuery(NBL_CONST_REF_ARG(MicrofacetCache) cache, scalar_type orientedEta)
    {
        quant_query_type dummy; // brdfs don't make use of this
        return dummy;
    }
    template<class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(surface_interactions::Isotropic<Interaction> && ReadableIsotropicMicrofacetCache<MicrofacetCache>)
    dg1_query_type createDG1Query(NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        dg1_query_type dg1_query;
        dg1_query.ndf = __base.template D<MicrofacetCache>(cache);
        dg1_query.lambda_V = base_type::LambdaC2(interaction.getNdotV2());
        return dg1_query;
    }
    template<class LS, class Interaction NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Isotropic<Interaction>)
    g2g1_query_type createG2G1Query(NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        g2g1_query_type g2_query;
        g2_query.lambda_L = base_type::LambdaC2(_sample.getNdotL2());
        g2_query.lambda_V = base_type::LambdaC2(interaction.getNdotV2());
        return g2_query;
    }

    template<class LS, class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Isotropic<Interaction> && ReadableIsotropicMicrofacetCache<MicrofacetCache>)
    quant_type D(NBL_CONST_REF_ARG(quant_query_type) quant_query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        scalar_type d = __base.template D<MicrofacetCache>(cache);
        return createDualMeasureQuantity<T>(d, interaction.getNdotV(BxDFClampMode::BCM_MAX), _sample.getNdotL(BxDFClampMode::BCM_MAX));
    }

    template<class LS, class Interaction NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Isotropic<Interaction>)
    quant_type DG1(NBL_CONST_REF_ARG(dg1_query_type) query, NBL_CONST_REF_ARG(quant_query_type) quant_query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        scalar_type dg1 = __base.DG1(query);
        return createDualMeasureQuantity<T>(dg1, interaction.getNdotV(BxDFClampMode::BCM_MAX), _sample.getNdotL(BxDFClampMode::BCM_MAX));
    }

    template<class LS, class Interaction NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Isotropic<Interaction>)
    quant_type correlated(NBL_CONST_REF_ARG(g2g1_query_type) query, NBL_CONST_REF_ARG(quant_query_type) quant_query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        scalar_type g = __base.template correlated<LS, Interaction>(query, _sample, interaction);
        return createDualMeasureQuantity<T>(g, interaction.getNdotV(BxDFClampMode::BCM_MAX), _sample.getNdotL(BxDFClampMode::BCM_MAX));
    }

    template<class LS, class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Isotropic<Interaction> && ReadableIsotropicMicrofacetCache<MicrofacetCache>)
    scalar_type G2_over_G1(NBL_CONST_REF_ARG(g2g1_query_type) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        return __base.template G2_over_G1<LS, Interaction, MicrofacetCache>(query, _sample, interaction, cache);
    }

    base_type __base;
};

template<typename T>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<T>)
struct Beckmann<T,true,MTT_REFLECT NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<T>) >
{
    using scalar_type = T;
    using base_type = impl::BeckmannCommon<T,true>;
    using quant_type = SDualMeasureQuant<scalar_type>;

    using dg1_query_type = typename base_type::dg1_query_type;
    using g2g1_query_type = typename base_type::g2g1_query_type;
    using quant_query_type = SQuantQuery<scalar_type>;

    template<class MicrofacetCache NBL_FUNC_REQUIRES(ReadableIsotropicMicrofacetCache<MicrofacetCache>)
    quant_query_type createQuantQuery(NBL_CONST_REF_ARG(MicrofacetCache) cache, scalar_type orientedEta)
    {
        quant_query_type dummy; // brdfs don't make use of this
        return dummy;
    }
    template<class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(surface_interactions::Isotropic<Interaction> && ReadableIsotropicMicrofacetCache<MicrofacetCache>)
    dg1_query_type createDG1Query(NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        dg1_query_type dg1_query;
        dg1_query.ndf = __base.template D<MicrofacetCache>(cache);
        dg1_query.lambda_V = base_type::LambdaC2(interaction.getTdotV2(), interaction.getBdotV2(), interaction.getNdotV2());
        return dg1_query;
    }
    template<class LS, class Interaction NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Isotropic<Interaction>)
    g2g1_query_type createG2G1Query(NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        g2g1_query_type g2_query;
        g2_query.lambda_L = base_type::LambdaC2(_sample.getTdotL2(), _sample.getBdotL2(), _sample.getNdotL2());
        g2_query.lambda_V = base_type::LambdaC2(interaction.getTdotV2(), interaction.getBdotV2(), interaction.getNdotV2());
        return g2_query;
    }

    template<class LS, class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Anisotropic<Interaction> && AnisotropicMicrofacetCache<MicrofacetCache>)
    quant_type D(NBL_CONST_REF_ARG(quant_query_type) quant_query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        scalar_type d = __base.template D<MicrofacetCache>(cache);
        return createDualMeasureQuantity<T>(d, interaction.getNdotV(BxDFClampMode::BCM_MAX), _sample.getNdotL(BxDFClampMode::BCM_MAX));
    }

    template<class LS, class Interaction NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Anisotropic<Interaction>)
    quant_type DG1(NBL_CONST_REF_ARG(dg1_query_type) query, NBL_CONST_REF_ARG(quant_query_type) quant_query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        scalar_type dg1 = __base.DG1(query);
        return createDualMeasureQuantity<T>(dg1, interaction.getNdotV(BxDFClampMode::BCM_MAX), _sample.getNdotL(BxDFClampMode::BCM_MAX));
    }

    template<class LS, class Interaction NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Anisotropic<Interaction>)
    quant_type correlated(NBL_CONST_REF_ARG(g2g1_query_type) query, NBL_CONST_REF_ARG(quant_query_type) quant_query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        scalar_type g = __base.template correlated<LS, Interaction>(query, _sample, interaction);
        return createDualMeasureQuantity<T>(g, interaction.getNdotV(BxDFClampMode::BCM_MAX), _sample.getNdotL(BxDFClampMode::BCM_MAX));
    }

    template<class LS, class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Anisotropic<Interaction> && AnisotropicMicrofacetCache<MicrofacetCache>)
    scalar_type G2_over_G1(NBL_CONST_REF_ARG(g2g1_query_type) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        return __base.template G2_over_G1<LS, Interaction, MicrofacetCache>(query, _sample, interaction, cache);
    }

    base_type __base;
};

// partial for bsdf
template<typename T, MicrofacetTransformTypes reflect_refract>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<T>)
struct Beckmann<T,false,reflect_refract NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<T>) >
{
    using scalar_type = T;
    using base_type = impl::BeckmannCommon<T,false>;
    using quant_type = SDualMeasureQuant<scalar_type>;

    using dg1_query_type = typename base_type::dg1_query_type;
    using g2g1_query_type = typename base_type::g2g1_query_type;
    using quant_query_type = SQuantQuery<scalar_type>;

    template<class MicrofacetCache NBL_FUNC_REQUIRES(ReadableIsotropicMicrofacetCache<MicrofacetCache>)
    quant_query_type createQuantQuery(NBL_CONST_REF_ARG(MicrofacetCache) cache, scalar_type orientedEta)
    {
        quant_query_type quant_query;
        quant_query.VdotHLdotH = cache.getVdotHLdotH();
        quant_query.VdotH_etaLdotH = cache.getVdotH() + orientedEta * cache.getLdotH();
        return quant_query;
    }
    template<class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(surface_interactions::Isotropic<Interaction> && ReadableIsotropicMicrofacetCache<MicrofacetCache>)
    dg1_query_type createDG1Query(NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        dg1_query_type dg1_query;
        dg1_query.ndf = __base.template D<MicrofacetCache>(cache);
        dg1_query.lambda_V = base_type::LambdaC2(interaction.getNdotV2());
        return dg1_query;
    }
    template<class LS, class Interaction NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Isotropic<Interaction>)
    g2g1_query_type createG2G1Query(NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        g2g1_query_type g2_query;
        g2_query.lambda_L = base_type::LambdaC2(_sample.getNdotL2());
        g2_query.lambda_V = base_type::LambdaC2(interaction.getNdotV2());
        return g2_query;
    }

    template<class LS, class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Isotropic<Interaction> && ReadableIsotropicMicrofacetCache<MicrofacetCache>)
    quant_type D(NBL_CONST_REF_ARG(quant_query_type) quant_query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        scalar_type d = __base.template D<MicrofacetCache>(cache);
        return createDualMeasureQuantity<T,reflect_refract>(d, interaction.getNdotV(BxDFClampMode::BCM_ABS), _sample.getNdotL(BxDFClampMode::BCM_ABS), quant_query.getVdotHLdotH(), quant_query.getVdotH_etaLdotH());
    }

    template<class LS, class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Isotropic<Interaction>)
    quant_type DG1(NBL_CONST_REF_ARG(dg1_query_type) query, NBL_CONST_REF_ARG(quant_query_type) quant_query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        scalar_type dg1 = __base.DG1(query);
        return createDualMeasureQuantity<T,reflect_refract>(dg1, interaction.getNdotV(BxDFClampMode::BCM_ABS), _sample.getNdotL(BxDFClampMode::BCM_ABS), quant_query.getVdotHLdotH(), quant_query.getVdotH_etaLdotH());
    }

    template<class LS, class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Isotropic<Interaction>)
    quant_type correlated(NBL_CONST_REF_ARG(g2g1_query_type) query, NBL_CONST_REF_ARG(quant_query_type) quant_query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        scalar_type g = __base.template correlated<LS, Interaction>(query, _sample, interaction);
        return createDualMeasureQuantity<T,reflect_refract>(g, interaction.getNdotV(BxDFClampMode::BCM_ABS), _sample.getNdotL(BxDFClampMode::BCM_ABS), quant_query.getVdotHLdotH(), quant_query.getVdotH_etaLdotH());
    }

    template<class LS, class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Isotropic<Interaction> && ReadableIsotropicMicrofacetCache<MicrofacetCache>)
    scalar_type G2_over_G1(NBL_CONST_REF_ARG(g2g1_query_type) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        return __base.template G2_over_G1<LS, Interaction, MicrofacetCache>(query, _sample, interaction, cache);
    }

    base_type __base;
};

template<typename T, MicrofacetTransformTypes reflect_refract>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<T>)
struct Beckmann<T,true,reflect_refract NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<T>) >
{
    using scalar_type = T;
    using base_type = impl::BeckmannCommon<T,true>;
    using quant_type = SDualMeasureQuant<scalar_type>;

    using dg1_query_type = typename base_type::dg1_query_type;
    using g2g1_query_type = typename base_type::g2g1_query_type;
    using quant_query_type = SQuantQuery<scalar_type>;

    template<class MicrofacetCache NBL_FUNC_REQUIRES(ReadableIsotropicMicrofacetCache<MicrofacetCache>)
    quant_query_type createQuantQuery(NBL_CONST_REF_ARG(MicrofacetCache) cache, scalar_type orientedEta)
    {
        quant_query_type quant_query;
        quant_query.VdotHLdotH = cache.getVdotHLdotH();
        quant_query.VdotH_etaLdotH = cache.getVdotH() + orientedEta * cache.getLdotH();
        return quant_query;
    }
    template<class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(surface_interactions::Isotropic<Interaction> && ReadableIsotropicMicrofacetCache<MicrofacetCache>)
    dg1_query_type createDG1Query(NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        dg1_query_type dg1_query;
        dg1_query.ndf = __base.template D<MicrofacetCache>(cache);
        dg1_query.lambda_V = base_type::LambdaC2(interaction.getTdotV2(), interaction.getBdotV2(), interaction.getNdotV2());
        return dg1_query;
    }
    template<class LS, class Interaction NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Isotropic<Interaction>)
    g2g1_query_type createG2G1Query(NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        g2g1_query_type g2_query;
        g2_query.lambda_L = base_type::LambdaC2(_sample.getTdotL2(), _sample.getBdotL2(), _sample.getNdotL2());
        g2_query.lambda_V = base_type::LambdaC2(interaction.getTdotV2(), interaction.getBdotV2(), interaction.getNdotV2());
        return g2_query;
    }

    template<class LS, class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Anisotropic<Interaction> && AnisotropicMicrofacetCache<MicrofacetCache>)
    quant_type D(NBL_CONST_REF_ARG(quant_query_type) quant_query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        scalar_type d = __base.template D<MicrofacetCache>(cache);
        return createDualMeasureQuantity<T>(d, interaction.getNdotV(BxDFClampMode::BCM_ABS), _sample.getNdotL(BxDFClampMode::BCM_ABS), quant_query.getVdotHLdotH(), quant_query.getVdotH_etaLdotH());
    }

    template<class LS, class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Anisotropic<Interaction>)
    quant_type DG1(NBL_CONST_REF_ARG(dg1_query_type) query, NBL_CONST_REF_ARG(quant_query_type) quant_query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        scalar_type dg1 = __base.template DG1<Query>(query);
        return createDualMeasureQuantity<T>(dg1, interaction.getNdotV(BxDFClampMode::BCM_ABS), _sample.getNdotL(BxDFClampMode::BCM_ABS), quant_query.getVdotHLdotH(), quant_query.getVdotH_etaLdotH());
    }

    template<class LS, class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(beckmann_concepts::G2overG1Query<Query> && LightSample<LS> && surface_interactions::Anisotropic<Interaction>)
    quant_type correlated(NBL_CONST_REF_ARG(g2g1_query_type) query, NBL_CONST_REF_ARG(quant_query_type) quant_query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        scalar_type g = __base.template correlated<LS, Interaction>(query, _sample, interaction);
        return createDualMeasureQuantity<T>(g, interaction.getNdotV(BxDFClampMode::BCM_ABS), _sample.getNdotL(BxDFClampMode::BCM_ABS), quant_query.getVdotHLdotH(), quant_query.getVdotH_etaLdotH());
    }

    template<class LS, class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(beckmann_concepts::G2overG1Query<Query> && LightSample<LS> && surface_interactions::Anisotropic<Interaction> && AnisotropicMicrofacetCache<MicrofacetCache>)
    scalar_type G2_over_G1(NBL_CONST_REF_ARG(g2g1_query_type) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        return __base.template G2_over_G1<LS, Interaction, MicrofacetCache>(query, _sample, interaction, cache);
    }

    base_type __base;
};

}
}
}
}

#endif
