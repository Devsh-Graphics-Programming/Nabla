// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_NDF_GGX_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_NDF_GGX_INCLUDED_

#include "nbl/builtin/hlsl/limits.hlsl"
#include "nbl/builtin/hlsl/bxdf/ndf/microfacet_to_light_transform.hlsl"
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
#define NBL_CONCEPT_NAME DG1Query
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

#define NBL_CONCEPT_NAME G2overG1Query
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (query, T)
NBL_CONCEPT_BEGIN(1)
#define query NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((query.getDevshV()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((query.getDevshL()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
);
#undef query
#include <nbl/builtin/hlsl/concepts/__end.hlsl>
}

namespace impl
{

template<typename T>
struct SGGXDG1Query
{
    using scalar_type = T;

    scalar_type getNdf() NBL_CONST_MEMBER_FUNC { return ndf; }
    scalar_type getG1over2NdotV() NBL_CONST_MEMBER_FUNC { return G1_over_2NdotV; }

    scalar_type ndf;
    scalar_type G1_over_2NdotV;
};

template<typename T>
struct SGGXG2XQuery
{
    using scalar_type = T;

    scalar_type getDevshV() NBL_CONST_MEMBER_FUNC { return devsh_v; }
    scalar_type getDevshL() NBL_CONST_MEMBER_FUNC { return devsh_l; }

    scalar_type devsh_v;
    scalar_type devsh_l;
};

template<typename T, bool IsBSDF, bool IsAnisotropic=false NBL_STRUCT_CONSTRAINABLE>
struct GGXCommon;

template<typename T, bool IsBSDF>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<T>)
struct GGXCommon<T,IsBSDF,false NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<T>) >
{
    using scalar_type = T;

    NBL_CONSTEXPR_STATIC_INLINE BxDFClampMode _clamp = IsBSDF ? BxDFClampMode::BCM_ABS : BxDFClampMode::BCM_MAX;

    // trowbridge-reitz
    template<class MicrofacetCache NBL_FUNC_REQUIRES(ReadableIsotropicMicrofacetCache<MicrofacetCache>)
    scalar_type D(NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        scalar_type denom = scalar_type(1.0) - one_minus_a2 * cache.getNdotH2();
        return a2 * numbers::inv_pi<scalar_type> / (denom * denom);
    }

    template<class Query NBL_FUNC_REQUIRES(ggx_concepts::DG1Query<Query>)
    static scalar_type DG1(NBL_CONST_REF_ARG(Query) query)
    {
        return scalar_type(0.5) * query.getNdf() * query.getG1over2NdotV();
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

    static scalar_type G1_wo_numerator_devsh_part(scalar_type absNdotX, scalar_type devsh_part)
    {
        // numerator is 2 * NdotX
        return scalar_type(1.0) / (absNdotX + devsh_part);
    }

    // without numerator, numerator is 2 * NdotV * NdotL, we factor out 4 * NdotV * NdotL, hence 0.5
    template<class Query, class LS, class Interaction NBL_FUNC_REQUIRES(ggx_concepts::G2overG1Query<Query> && LightSample<LS> && surface_interactions::Isotropic<Interaction>)
    static scalar_type correlated_wo_numerator(NBL_CONST_REF_ARG(Query) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        scalar_type Vterm = _sample.getNdotL(_clamp) * query.getDevshV();
        scalar_type Lterm = interaction.getNdotV(_clamp) * query.getDevshL();
        return scalar_type(0.5) / (Vterm + Lterm);
    }

    template<class Query, class LS, class Interaction NBL_FUNC_REQUIRES(ggx_concepts::G2overG1Query<Query> && LightSample<LS> && surface_interactions::Isotropic<Interaction>)
    static scalar_type correlated(NBL_CONST_REF_ARG(Query) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        return scalar_type(4.0) * interaction.getNdotV(_clamp) * _sample.getNdotL(_clamp) * correlated_wo_numerator<LS, Interaction>(query, _sample, interaction);
    }

    template<class Query, class LS, class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(ggx_concepts::G2overG1Query<Query> && LightSample<LS> && surface_interactions::Isotropic<Interaction> && ReadableIsotropicMicrofacetCache<MicrofacetCache>)
    static scalar_type G2_over_G1(NBL_CONST_REF_ARG(Query) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
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
            G2_over_G1 = bxdf::beta_wo_check<scalar_type>(onePlusLambda_L, onePlusLambda_V) * onePlusLambda_V;
        }
        else
        {
            G2_over_G1 = NdotL * (devsh_v + NdotV);
            G2_over_G1 /= NdotV * devsh_l + NdotL * devsh_v;
        }

        return G2_over_G1;
    }

    scalar_type a2;
    scalar_type one_minus_a2;
};

template<typename T, bool IsBSDF>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<T>)
struct GGXCommon<T,IsBSDF,true NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<T>) >
{
    using scalar_type = T;

    NBL_CONSTEXPR_STATIC_INLINE BxDFClampMode _clamp = IsBSDF ? BxDFClampMode::BCM_ABS : BxDFClampMode::BCM_MAX;

    template<class MicrofacetCache NBL_FUNC_REQUIRES(AnisotropicMicrofacetCache<MicrofacetCache>)
    scalar_type D(NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        scalar_type denom = cache.getTdotH2() / ax2 + cache.getBdotH2() / ay2 + cache.getNdotH2();
        return numbers::inv_pi<scalar_type> / (a2 * denom * denom);
    }

    // TODO: potential idea for making GGX spin using covariance matrix of sorts: https://www.desmos.com/3d/weq2ginq9o

    template<class Query NBL_FUNC_REQUIRES(ggx_concepts::DG1Query<Query>)
    static scalar_type DG1(NBL_CONST_REF_ARG(Query) query)
    {
        GGXCommon<T,false> ggx;
        return ggx.DG1(query);
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

    static scalar_type G1_wo_numerator_devsh_part(scalar_type NdotX, scalar_type devsh_part)
    {
        return scalar_type(1.0) / (NdotX + devsh_part);
    }

    // without numerator
    template<class Query, class LS, class Interaction NBL_FUNC_REQUIRES(ggx_concepts::G2overG1Query<Query> && LightSample<LS> && surface_interactions::Anisotropic<Interaction>)
    static scalar_type correlated_wo_numerator(NBL_CONST_REF_ARG(Query) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        scalar_type Vterm = _sample.getNdotL(_clamp) * query.getDevshV();
        scalar_type Lterm = interaction.getNdotV(_clamp) * query.getDevshL();
        return scalar_type(0.5) / (Vterm + Lterm);
    }

    template<class Query, class LS, class Interaction NBL_FUNC_REQUIRES(ggx_concepts::G2overG1Query<Query> && LightSample<LS> && surface_interactions::Anisotropic<Interaction>)
    static scalar_type correlated(NBL_CONST_REF_ARG(Query) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        return scalar_type(4.0) * interaction.getNdotV(_clamp) * _sample.getNdotL(_clamp) * correlated_wo_numerator<LS, Interaction>(query, _sample, interaction);
    }

    template<class Query, class LS, class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(ggx_concepts::G2overG1Query<Query> && LightSample<LS> && surface_interactions::Anisotropic<Interaction> && AnisotropicMicrofacetCache<MicrofacetCache>)
    static scalar_type G2_over_G1(NBL_CONST_REF_ARG(Query) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
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
            G2_over_G1 = bxdf::beta_wo_check<scalar_type>(onePlusLambda_L, onePlusLambda_V) * onePlusLambda_V;
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

template<typename T>
struct GGXGenerateH
{
    using scalar_type = T;
    using vector2_type = vector<T, 2>;
    using vector3_type = vector<T, 3>;

    vector3_type __call(const vector3_type localV, const vector2_type u)
    {
        vector3_type V = nbl::hlsl::normalize<vector3_type>(vector3_type(ax * localV.x, ay * localV.y, localV.z));//stretch view vector so that we're sampling as if roughness=1.0

        scalar_type lensq = V.x*V.x + V.y*V.y;
        vector3_type T1 = lensq > 0.0 ? vector3_type(-V.y, V.x, 0.0) * rsqrt<scalar_type>(lensq) : vector3_type(1.0,0.0,0.0);
        vector3_type T2 = cross<scalar_type>(V,T1);

        scalar_type r = sqrt<scalar_type>(u.x);
        scalar_type phi = 2.0 * numbers::pi<scalar_type> * u.y;
        scalar_type t1 = r * cos<scalar_type>(phi);
        scalar_type t2 = r * sin<scalar_type>(phi);
        scalar_type s = 0.5 * (1.0 + V.z);
        t2 = (1.0 - s)*sqrt<scalar_type>(1.0 - t1*t1) + s*t2;

        //reprojection onto hemisphere
        //tested, seems -t1*t1-t2*t2>-1.0
        vector3_type H = t1*T1 + t2*T2 + sqrt<scalar_type>(1.0-t1*t1-t2*t2)*V;
        //unstretch
        return nbl::hlsl::normalize<vector3_type>(vector3_type(ax*H.x, ay*H.y, H.z));
    }

    scalar_type ax;
    scalar_type ay;
};
}


template<typename T, bool _IsAnisotropic, MicrofacetTransformTypes reflect_refract  NBL_PRIMARY_REQUIRES(concepts::FloatingPointScalar<T>)
struct GGX
{
    NBL_CONSTEXPR_STATIC_INLINE bool IsAnisotropic = _IsAnisotropic;
    NBL_CONSTEXPR_STATIC_INLINE bool IsBSDF = reflect_refract != MTT_REFLECT;

    using this_t = GGX<T, _IsAnisotropic, reflect_refract>;
    using scalar_type = T;
    using base_type = impl::GGXCommon<T,IsBSDF,IsAnisotropic>;
    using quant_type = SDualMeasureQuant<scalar_type>;
    using vector2_type = vector<T, 2>;
    using vector3_type = vector<T, 3>;

    using dg1_query_type = impl::SGGXDG1Query<scalar_type>;
    using g2g1_query_type = impl::SGGXG2XQuery<scalar_type>;
    using quant_query_type = impl::NDFQuantQuery<scalar_type>;

    NBL_CONSTEXPR_STATIC_INLINE BxDFClampMode _clamp = IsBSDF ? BxDFClampMode::BCM_ABS : BxDFClampMode::BCM_NONE;
    template<class Interaction>
    NBL_CONSTEXPR_STATIC_INLINE bool RequiredInteraction = IsAnisotropic ? surface_interactions::Anisotropic<Interaction> : surface_interactions::Isotropic<Interaction>;
    template<class MicrofacetCache>
    NBL_CONSTEXPR_STATIC_INLINE bool RequiredMicrofacetCache = IsAnisotropic ? AnisotropicMicrofacetCache<MicrofacetCache> : ReadableIsotropicMicrofacetCache<MicrofacetCache>;

    template<typename C=bool_constant<!IsAnisotropic> >
    enable_if_t<C::value && !IsAnisotropic, this_t> create(scalar_type A)
    {
        this_t retval;
        retval.__ndf_base.a2 = A*A;
        retval.__ndf_base.one_minus_a2 = scalar_type(1.0) - A*A;
        retval.__generate_base.ax = A;
        retval.__generate_base.ay = A;
        return retval;
    }
    template<typename C=bool_constant<IsAnisotropic> >
    enable_if_t<C::value && IsAnisotropic, this_t> create(scalar_type ax, scalar_type ay)
    {
        this_t retval;
        retval.__ndf_base.ax2 = ax*ax;
        retval.__ndf_base.ay2 = ay*ay;
        retval.__ndf_base.a2 = ax*ay;
        retval.__generate_base.ax = ax;
        retval.__generate_base.ay = ay;
        return retval;
    }

    template<class MicrofacetCache, typename C=bool_constant<!IsBSDF> NBL_FUNC_REQUIRES(RequiredMicrofacetCache<MicrofacetCache>)
    enable_if_t<C::value && !IsBSDF, quant_query_type> createQuantQuery(NBL_CONST_REF_ARG(MicrofacetCache) cache, scalar_type orientedEta)
    {
        quant_query_type dummy; // brdfs don't make use of this
        return dummy;
    }
    template<class MicrofacetCache, typename C=bool_constant<IsBSDF> NBL_FUNC_REQUIRES(RequiredMicrofacetCache<MicrofacetCache>)
    enable_if_t<C::value && IsBSDF, quant_query_type> createQuantQuery(NBL_CONST_REF_ARG(MicrofacetCache) cache, scalar_type orientedEta)
    {
        quant_query_type quant_query;
        quant_query.VdotHLdotH = cache.getVdotHLdotH();
        quant_query.VdotH_etaLdotH = cache.getVdotH() + orientedEta * cache.getLdotH();
        return quant_query;
    }
    template<class Interaction, class MicrofacetCache, typename C=bool_constant<!IsAnisotropic> NBL_FUNC_REQUIRES(RequiredInteraction<Interaction> && RequiredMicrofacetCache<MicrofacetCache>)
    enable_if_t<C::value && !IsAnisotropic, dg1_query_type> createDG1Query(NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        dg1_query_type dg1_query;
        dg1_query.ndf = __ndf_base.template D<MicrofacetCache>(cache);
        scalar_type clampedNdotV = interaction.getNdotV(_clamp);
        dg1_query.G1_over_2NdotV = __ndf_base.G1_wo_numerator(clampedNdotV, interaction.getNdotV2());
        return dg1_query;
    }
    template<class LS, class Interaction, typename C=bool_constant<!IsAnisotropic> NBL_FUNC_REQUIRES(LightSample<LS> && RequiredInteraction<Interaction>)
    enable_if_t<C::value && !IsAnisotropic, g2g1_query_type> createG2G1Query(NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        g2g1_query_type g2_query;
        g2_query.devsh_l = __ndf_base.devsh_part(_sample.getNdotL2());
        g2_query.devsh_v = __ndf_base.devsh_part(interaction.getNdotV2());
        return g2_query;
    }
    template<class Interaction, class MicrofacetCache, typename C=bool_constant<IsAnisotropic> NBL_FUNC_REQUIRES(RequiredInteraction<Interaction> && RequiredMicrofacetCache<MicrofacetCache>)
    enable_if_t<C::value && IsAnisotropic, dg1_query_type> createDG1Query(NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        dg1_query_type dg1_query;
        dg1_query.ndf = __ndf_base.template D<MicrofacetCache>(cache);
        scalar_type clampedNdotV = interaction.getNdotV(_clamp);
        dg1_query.G1_over_2NdotV = __ndf_base.G1_wo_numerator(clampedNdotV, interaction.getTdotV2(), interaction.getBdotV2(), interaction.getNdotV2());
        return dg1_query;
    }
    template<class LS, class Interaction, typename C=bool_constant<IsAnisotropic> NBL_FUNC_REQUIRES(LightSample<LS> && RequiredInteraction<Interaction>)
    enable_if_t<C::value && IsAnisotropic, g2g1_query_type> createG2G1Query(NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        g2g1_query_type g2_query;
        g2_query.devsh_l = __ndf_base.devsh_part(_sample.getTdotL2(), _sample.getBdotL2(), _sample.getNdotL2());
        g2_query.devsh_v = __ndf_base.devsh_part(interaction.getTdotV2(), interaction.getBdotV2(), interaction.getNdotV2());
        return g2_query;
    }

    vector3_type generateH(const vector3_type localV, const vector2_type u)
    {
        return __generate_base.__call(localV, u);
    }

    template<class LS, class Interaction, class MicrofacetCache, typename C=bool_constant<!IsBSDF> NBL_FUNC_REQUIRES(LightSample<LS> && RequiredInteraction<Interaction> && RequiredMicrofacetCache<MicrofacetCache>)
    enable_if_t<C::value && !IsBSDF, quant_type> D(NBL_CONST_REF_ARG(quant_query_type) quant_query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        scalar_type d = __ndf_base.template D<MicrofacetCache>(cache);
        quant_type dmq;
        dmq.microfacetMeasure = d;
        dmq.projectedLightMeasure = d * _sample.getNdotL(BxDFClampMode::BCM_MAX);
        return dmq;
    }
    template<class LS, class Interaction, class MicrofacetCache, typename C=bool_constant<IsBSDF> NBL_FUNC_REQUIRES(LightSample<LS> && RequiredInteraction<Interaction> && RequiredMicrofacetCache<MicrofacetCache>)
    enable_if_t<C::value && IsBSDF, quant_type> D(NBL_CONST_REF_ARG(quant_query_type) quant_query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        scalar_type d = __ndf_base.template D<MicrofacetCache>(cache);
        quant_type dmq;
        dmq.microfacetMeasure = d;  // note: microfacetMeasure/2NdotV

        const scalar_type VdotHLdotH = quant_query.getVdotHLdotH();
        const scalar_type VdotH_etaLdotH = quant_query.getVdotH_etaLdotH();
        const bool transmitted = reflect_refract==MTT_REFRACT || (reflect_refract!=MTT_REFLECT && VdotHLdotH < scalar_type(0.0));
        scalar_type NdotL_over_denominator = _sample.getNdotL(BxDFClampMode::BCM_ABS);
        if (transmitted)
                NdotL_over_denominator *= -scalar_type(4.0) * VdotHLdotH / (VdotH_etaLdotH * VdotH_etaLdotH);
        dmq.projectedLightMeasure = d * NdotL_over_denominator;
        return dmq;
    }

    template<class LS, class Interaction, typename C=bool_constant<!IsBSDF> NBL_FUNC_REQUIRES(LightSample<LS> && RequiredInteraction<Interaction>)
    enable_if_t<C::value && !IsBSDF, quant_type> DG1(NBL_CONST_REF_ARG(dg1_query_type) query, NBL_CONST_REF_ARG(quant_query_type) quant_query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        scalar_type dg1 = base_type::template DG1<dg1_query_type>(query);
        quant_type dmq;
        dmq.microfacetMeasure = dg1;
        dmq.projectedLightMeasure = dg1;// TODO: figure this out * _sample.getNdotL(BxDFClampMode::BCM_MAX);
        return dmq;
    }
    template<class LS, class Interaction, typename C=bool_constant<IsBSDF> NBL_FUNC_REQUIRES(LightSample<LS> && RequiredInteraction<Interaction>)
    enable_if_t<C::value && IsBSDF, quant_type> DG1(NBL_CONST_REF_ARG(dg1_query_type) query, NBL_CONST_REF_ARG(quant_query_type) quant_query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        scalar_type dg1 = base_type::template DG1<dg1_query_type>(query);
        quant_type dmq;
        dmq.microfacetMeasure = dg1;  // note: microfacetMeasure/2NdotV

        const scalar_type VdotHLdotH = quant_query.getVdotHLdotH();
        const scalar_type VdotH_etaLdotH = quant_query.getVdotH_etaLdotH();
        const bool transmitted = reflect_refract==MTT_REFRACT || (reflect_refract!=MTT_REFLECT && VdotHLdotH < scalar_type(0.0));
        scalar_type NdotL_over_denominator = _sample.getNdotL(BxDFClampMode::BCM_ABS);
        if (transmitted)
                NdotL_over_denominator *= -scalar_type(4.0) * VdotHLdotH / (VdotH_etaLdotH * VdotH_etaLdotH);
        dmq.projectedLightMeasure = dg1;// TODO: figure this out * NdotL_over_denominator;
        return dmq;
    }

    template<class LS, class Interaction NBL_FUNC_REQUIRES(LightSample<LS> && RequiredInteraction<Interaction>)
    scalar_type correlated(NBL_CONST_REF_ARG(g2g1_query_type) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        return base_type::template correlated_wo_numerator<g2g1_query_type, LS, Interaction>(query, _sample, interaction);
    }

    template<class LS, class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(LightSample<LS> && RequiredInteraction<Interaction> && RequiredMicrofacetCache<MicrofacetCache>)
    scalar_type G2_over_G1(NBL_CONST_REF_ARG(g2g1_query_type) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        return base_type::template G2_over_G1<g2g1_query_type, LS, Interaction, MicrofacetCache>(query, _sample, interaction, cache);
    }

    base_type __ndf_base;
    impl::GGXGenerateH<scalar_type> __generate_base;
};

namespace impl
{
template<class T, class U>
struct is_ggx : bool_constant<
    is_same<T, GGX<U, false, MTT_REFLECT> >::value ||
    is_same<T, GGX<U, true, MTT_REFLECT> >::value ||
    is_same<T, GGX<U, false, MTT_REFRACT> >::value ||
    is_same<T, GGX<U, true, MTT_REFRACT> >::value ||
    is_same<T, GGX<U, false, MTT_REFLECT_REFRACT> >::value ||
    is_same<T, GGX<U, true, MTT_REFLECT_REFRACT> >::value
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
