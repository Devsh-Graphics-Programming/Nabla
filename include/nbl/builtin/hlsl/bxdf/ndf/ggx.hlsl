// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_NDF_GGX_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_NDF_GGX_INCLUDED_

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
    BxDFClampMode getClampMode() NBL_CONST_MEMBER_FUNC { return _clamp; }

    scalar_type devsh_v;
    scalar_type devsh_l;
    BxDFClampMode _clamp;
};

template<typename T>
struct SGGXQuantQuery
{
    using scalar_type = T;

    scalar_type getVdotHLdotH() NBL_CONST_MEMBER_FUNC { return VdotHLdotH; }
    scalar_type getVdotH_etaLdotH() NBL_CONST_MEMBER_FUNC { return VdotH_etaLdotH; }

    scalar_type VdotHLdotH;
    scalar_type VdotH_etaLdotH;
};

template<typename T, bool IsAnisotropic=false NBL_STRUCT_CONSTRAINABLE>
struct GGXCommon;

template<typename T>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<T>)
struct GGXCommon<T,false NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<T>) >
{
    using scalar_type = T;
    using dg1_query_type = SGGXDG1Query<scalar_type>;
    using g2g1_query_type = SGGXG2XQuery<scalar_type>;

    // trowbridge-reitz
    template<class MicrofacetCache NBL_FUNC_REQUIRES(ReadableIsotropicMicrofacetCache<MicrofacetCache>)
    scalar_type D(NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        scalar_type denom = scalar_type(1.0) - one_minus_a2 * cache.getNdotH2();
        return a2 * numbers::inv_pi<scalar_type> / (denom * denom);
    }

    scalar_type DG1(NBL_CONST_REF_ARG(dg1_query_type) query)
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
    template<class LS, class Interaction NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Isotropic<Interaction>)
    scalar_type correlated_wo_numerator(NBL_CONST_REF_ARG(g2g1_query_type) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        BxDFClampMode _clamp = query.getClampMode();
        assert(_clamp != BxDFClampMode::BCM_NONE);

        scalar_type Vterm = _sample.getNdotL(_clamp) * query.getDevshV();
        scalar_type Lterm = interaction.getNdotV(_clamp) * query.getDevshL();
        return scalar_type(0.5) / (Vterm + Lterm);
    }

    template<class LS, class Interaction NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Isotropic<Interaction>)
    scalar_type correlated(NBL_CONST_REF_ARG(g2g1_query_type) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        BxDFClampMode _clamp = query.getClampMode();
        assert(_clamp != BxDFClampMode::BCM_NONE);
        return scalar_type(4.0) * interaction.getNdotV(_clamp) * _sample.getNdotL(_clamp) * correlated_wo_numerator<LS, Interaction>(query, _sample, interaction);
    }

    template<class LS, class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Isotropic<Interaction> && ReadableIsotropicMicrofacetCache<MicrofacetCache>)
    scalar_type G2_over_G1(NBL_CONST_REF_ARG(g2g1_query_type) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
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
            G2_over_G1 = bxdf::beta_wo_check<scalar_type>(onePlusLambda_L, onePlusLambda_V) * onePlusLambda_V;
        }
        else
        {
            G2_over_G1 = NdotL * (devsh_v + NdotV);
            G2_over_G1 /= NdotV * devsh_l + NdotL * devsh_v;
        }

        return G2_over_G1;
    }

    vector<scalar_type, 2> A;   // TODO: remove?
    scalar_type a2;
    scalar_type one_minus_a2;
};

template<typename T>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<T>)
struct GGXCommon<T,true NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<T>) >
{
    using scalar_type = T;
    using dg1_query_type = SGGXDG1Query<scalar_type>;
    using g2g1_query_type = SGGXG2XQuery<scalar_type>;

    template<class MicrofacetCache NBL_FUNC_REQUIRES(AnisotropicMicrofacetCache<MicrofacetCache>)
    scalar_type D(NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        scalar_type denom = cache.getTdotH2() / ax2 + cache.getBdotH2() / ay2 + cache.getNdotH2();
        return numbers::inv_pi<scalar_type> / (a2 * denom * denom);
    }

    // TODO: potential idea for making GGX spin using covariance matrix of sorts: https://www.desmos.com/3d/weq2ginq9o

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

    scalar_type DG1(NBL_CONST_REF_ARG(dg1_query_type) query)
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
    template<class LS, class Interaction NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Anisotropic<Interaction>)
    scalar_type correlated_wo_numerator(NBL_CONST_REF_ARG(g2g1_query_type) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        BxDFClampMode _clamp = query.getClampMode();
        assert(_clamp != BxDFClampMode::BCM_NONE);

        scalar_type Vterm = _sample.getNdotL(_clamp) * query.getDevshV();
        scalar_type Lterm = interaction.getNdotV(_clamp) * query.getDevshL();
        return scalar_type(0.5) / (Vterm + Lterm);
    }

    template<class LS, class Interaction NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Anisotropic<Interaction>)
    scalar_type correlated(NBL_CONST_REF_ARG(g2g1_query_type) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        BxDFClampMode _clamp = query.getClampMode();
        assert(_clamp != BxDFClampMode::BCM_NONE);
        return scalar_type(4.0) * interaction.getNdotV(_clamp) * _sample.getNdotL(_clamp) * correlated_wo_numerator<LS, Interaction>(query, _sample, interaction);
    }

    template<class LS, class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Anisotropic<Interaction> && AnisotropicMicrofacetCache<MicrofacetCache>)
    scalar_type G2_over_G1(NBL_CONST_REF_ARG(g2g1_query_type) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
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
            G2_over_G1 = bxdf::beta_wo_check<scalar_type>(onePlusLambda_L, onePlusLambda_V) * onePlusLambda_V;
        }
        else
        {
            G2_over_G1 = NdotL * (devsh_v + NdotV);
            G2_over_G1 /= NdotV * devsh_l + NdotL * devsh_v;
        }

        return G2_over_G1;
    }

    vector<scalar_type, 2> A;
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

    static vector3_type __call(const vector2_type A, const vector3_type localV, const vector2_type u)
    {
        vector3_type V = nbl::hlsl::normalize<vector3_type>(vector3_type(A.x*localV.x, A.y*localV.y, localV.z));//stretch view vector so that we're sampling as if roughness=1.0

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
        //TODO try it wothout the max(), not sure if -t1*t1-t2*t2>-1.0
        vector3_type H = t1*T1 + t2*T2 + sqrt<scalar_type>(max<scalar_type>(0.0, 1.0-t1*t1-t2*t2))*V;
        //unstretch
        return nbl::hlsl::normalize<vector3_type>(vector3_type(A.x*H.x, A.y*H.y, H.z));
    }
};
}

template<typename T, bool IsAnisotropic, MicrofacetTransformTypes reflect_refract NBL_STRUCT_CONSTRAINABLE>
struct GGX;

// partial spec for brdf
template<typename T>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<T>)
struct GGX<T,false,MTT_REFLECT NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<T>) >
{
    using scalar_type = T;
    using base_type = impl::GGXCommon<T,false>;
    using quant_type = SDualMeasureQuant<scalar_type>;
    using vector2_type = vector<T, 2>;
    using vector3_type = vector<T, 3>;

    using dg1_query_type = typename base_type::dg1_query_type;
    using g2g1_query_type = typename base_type::g2g1_query_type;
    using quant_query_type = impl::SGGXQuantQuery<scalar_type>;

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
        scalar_type clampedNdotV = interaction.getNdotV();
        dg1_query.G1_over_2NdotV = __base.G1_wo_numerator(clampedNdotV, interaction.getNdotV2());
        return dg1_query;
    }
    template<class LS, class Interaction NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Isotropic<Interaction>)
    g2g1_query_type createG2G1Query(NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        g2g1_query_type g2_query;
        g2_query.devsh_l = __base.devsh_part(_sample.getNdotL2());
        g2_query.devsh_v = __base.devsh_part(interaction.getNdotV2());
        g2_query._clamp = BxDFClampMode::BCM_MAX;
        return g2_query;
    }

    vector<T, 3> generateH(const vector3_type localV, const vector2_type u)
    {
        return impl::GGXGenerateH<scalar_type>::__call(__base.A, localV, u);
    }

    template<class LS, class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Isotropic<Interaction> && ReadableIsotropicMicrofacetCache<MicrofacetCache>)
    quant_type D(NBL_CONST_REF_ARG(quant_query_type) quant_query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        scalar_type d = __base.template D<MicrofacetCache>(cache);
        quant_type dmq;
        dmq.microfacetMeasure = d;
        dmq.projectedLightMeasure = d * _sample.getNdotL(BxDFClampMode::BCM_MAX);
        return dmq;
    }

    template<class LS, class Interaction NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Isotropic<Interaction>)
    quant_type DG1(NBL_CONST_REF_ARG(dg1_query_type) query, NBL_CONST_REF_ARG(quant_query_type) quant_query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        scalar_type dg1 = __base.DG1(query);
        quant_type dmq;
        dmq.microfacetMeasure = dg1;
        dmq.projectedLightMeasure = dg1;// TODO: figure this out * _sample.getNdotL(BxDFClampMode::BCM_MAX);
        return dmq;
    }

    template<class LS, class Interaction NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Isotropic<Interaction>)
    scalar_type correlated(NBL_CONST_REF_ARG(g2g1_query_type) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        return __base.template correlated_wo_numerator<LS, Interaction>(query, _sample, interaction);
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
struct GGX<T,true,MTT_REFLECT NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<T>) >
{
    using scalar_type = T;
    using base_type = impl::GGXCommon<T,true>;
    using quant_type = SDualMeasureQuant<scalar_type>;
    using vector2_type = vector<T, 2>;
    using vector3_type = vector<T, 3>;

    using dg1_query_type = typename base_type::dg1_query_type;
    using g2g1_query_type = typename base_type::g2g1_query_type;
    using quant_query_type = impl::SGGXQuantQuery<scalar_type>;

    template<class MicrofacetCache NBL_FUNC_REQUIRES(AnisotropicMicrofacetCache<MicrofacetCache>)
    quant_query_type createQuantQuery(NBL_CONST_REF_ARG(MicrofacetCache) cache, scalar_type orientedEta)
    {
        quant_query_type dummy; // brdfs don't make use of this
        return dummy;
    }
    template<class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(surface_interactions::Anisotropic<Interaction> && AnisotropicMicrofacetCache<MicrofacetCache>)
    dg1_query_type createDG1Query(NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        dg1_query_type dg1_query;
        dg1_query.ndf = __base.template D<MicrofacetCache>(cache);
        scalar_type clampedNdotV = interaction.getNdotV();
        dg1_query.G1_over_2NdotV = __base.G1_wo_numerator(clampedNdotV, interaction.getTdotV2(), interaction.getBdotV2(), interaction.getNdotV2());
        return dg1_query;
    }
    template<class LS, class Interaction NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Anisotropic<Interaction>)
    g2g1_query_type createG2G1Query(NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        g2g1_query_type g2_query;
        g2_query.devsh_l = __base.devsh_part(_sample.getTdotL2(), _sample.getBdotL2(), _sample.getNdotL2());
        g2_query.devsh_v = __base.devsh_part(interaction.getTdotV2(), interaction.getBdotV2(), interaction.getNdotV2());
        g2_query._clamp = BxDFClampMode::BCM_MAX;
        return g2_query;
    }

    vector<T, 3> generateH(const vector3_type localV, const vector2_type u)
    {
        return impl::GGXGenerateH<scalar_type>::__call(__base.A, localV, u);
    }

    template<class LS, class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Anisotropic<Interaction> && AnisotropicMicrofacetCache<MicrofacetCache>)
    quant_type D(NBL_CONST_REF_ARG(quant_query_type) quant_query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        scalar_type d = __base.template D<MicrofacetCache>(cache);
        quant_type dmq;
        dmq.microfacetMeasure = d;
        dmq.projectedLightMeasure = d * _sample.getNdotL(BxDFClampMode::BCM_MAX);
        return dmq;
    }

    template<class LS, class Interaction NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Anisotropic<Interaction>)
    quant_type DG1(NBL_CONST_REF_ARG(dg1_query_type) query, NBL_CONST_REF_ARG(quant_query_type) quant_query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        scalar_type dg1 = __base.DG1(query);
        quant_type dmq;
        dmq.microfacetMeasure = dg1;
        dmq.projectedLightMeasure = dg1;// TODO: figure this out * _sample.getNdotL(BxDFClampMode::BCM_MAX);
        return dmq;
    }

    template<class LS, class Interaction NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Anisotropic<Interaction>)
    scalar_type correlated(NBL_CONST_REF_ARG(g2g1_query_type) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        return __base.template correlated_wo_numerator<LS, Interaction>(query, _sample, interaction);
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
struct GGX<T,false,reflect_refract NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<T>) >
{
    using scalar_type = T;
    using base_type = impl::GGXCommon<T,false>;
    using quant_type = SDualMeasureQuant<scalar_type>;
    using vector2_type = vector<T, 2>;
    using vector3_type = vector<T, 3>;

    using dg1_query_type = typename base_type::dg1_query_type;
    using g2g1_query_type = typename base_type::g2g1_query_type;
    using quant_query_type = impl::SGGXQuantQuery<scalar_type>;

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
        scalar_type clampedNdotV = interaction.getNdotV(BxDFClampMode::BCM_ABS);
        dg1_query.G1_over_2NdotV = __base.G1_wo_numerator(clampedNdotV, interaction.getNdotV2());
        return dg1_query;
    }
    template<class LS, class Interaction NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Isotropic<Interaction>)
    g2g1_query_type createG2G1Query(NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        g2g1_query_type g2_query;
        g2_query.devsh_l = __base.devsh_part(_sample.getNdotL2());
        g2_query.devsh_v = __base.devsh_part(interaction.getNdotV2());
        g2_query._clamp = BxDFClampMode::BCM_ABS;
        return g2_query;
    }

    vector<T, 3> generateH(const vector3_type localV, const vector2_type u)
    {
        return impl::GGXGenerateH<scalar_type>::__call(__base.A, localV, u);
    }

    template<class LS, class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Isotropic<Interaction> && ReadableIsotropicMicrofacetCache<MicrofacetCache>)
    quant_type D(NBL_CONST_REF_ARG(quant_query_type) quant_query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        scalar_type d = __base.template D<MicrofacetCache>(cache);
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

    template<class LS, class Interaction NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Isotropic<Interaction>)
    quant_type DG1(NBL_CONST_REF_ARG(dg1_query_type) query, NBL_CONST_REF_ARG(quant_query_type) quant_query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        scalar_type dg1 = __base.DG1(query);
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

    template<class LS, class Interaction NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Isotropic<Interaction>)
    scalar_type correlated(NBL_CONST_REF_ARG(g2g1_query_type) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        return __base.template correlated_wo_numerator<LS, Interaction>(query, _sample, interaction);
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
struct GGX<T,true,reflect_refract NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<T>) >
{
    using scalar_type = T;
    using base_type = impl::GGXCommon<T,true>;
    using quant_type = SDualMeasureQuant<scalar_type>;
    using vector2_type = vector<T, 2>;
    using vector3_type = vector<T, 3>;

    using dg1_query_type = typename base_type::dg1_query_type;
    using g2g1_query_type = typename base_type::g2g1_query_type;
    using quant_query_type = impl::SGGXQuantQuery<scalar_type>;

    template<class MicrofacetCache NBL_FUNC_REQUIRES(AnisotropicMicrofacetCache<MicrofacetCache>)
    quant_query_type createQuantQuery(NBL_CONST_REF_ARG(MicrofacetCache) cache, scalar_type orientedEta)
    {
        quant_query_type quant_query;
        quant_query.VdotHLdotH = cache.getVdotHLdotH();
        quant_query.VdotH_etaLdotH = cache.getVdotH() + orientedEta * cache.getLdotH();
        return quant_query;
    }
    template<class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(surface_interactions::Anisotropic<Interaction> && AnisotropicMicrofacetCache<MicrofacetCache>)
    dg1_query_type createDG1Query(NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        dg1_query_type dg1_query;
        dg1_query.ndf = __base.template D<MicrofacetCache>(cache);
        scalar_type clampedNdotV = interaction.getNdotV(BxDFClampMode::BCM_ABS);
        dg1_query.G1_over_2NdotV = __base.G1_wo_numerator(clampedNdotV, interaction.getTdotV2(), interaction.getBdotV2(), interaction.getNdotV2());
        return dg1_query;
    }
    template<class LS, class Interaction NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Anisotropic<Interaction>)
    g2g1_query_type createG2G1Query(NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        g2g1_query_type g2_query;
        g2_query.devsh_l = __base.devsh_part(_sample.getTdotL2(), _sample.getBdotL2(), _sample.getNdotL2());
        g2_query.devsh_v = __base.devsh_part(interaction.getTdotV2(), interaction.getBdotV2(), interaction.getNdotV2());
        g2_query._clamp = BxDFClampMode::BCM_ABS;
        return g2_query;
    }

    vector<T, 3> generateH(const vector3_type localV, const vector2_type u)
    {
        return impl::GGXGenerateH<scalar_type>::__call(__base.A, localV, u);
    }

    template<class LS, class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Anisotropic<Interaction> && AnisotropicMicrofacetCache<MicrofacetCache>)
    quant_type D(NBL_CONST_REF_ARG(quant_query_type) quant_query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        scalar_type d = __base.template D<MicrofacetCache>(cache);
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

    template<class LS, class Interaction NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Anisotropic<Interaction>)
    quant_type DG1(NBL_CONST_REF_ARG(dg1_query_type) query, NBL_CONST_REF_ARG(quant_query_type) quant_query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        scalar_type dg1 = __base.DG1(query);
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

    template<class LS, class Interaction NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Anisotropic<Interaction>)
    scalar_type correlated(NBL_CONST_REF_ARG(g2g1_query_type) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        return __base.template correlated_wo_numerator<LS, Interaction>(query, _sample, interaction);
    }

    template<class LS, class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(LightSample<LS> && surface_interactions::Anisotropic<Interaction> && AnisotropicMicrofacetCache<MicrofacetCache>)
    scalar_type G2_over_G1(NBL_CONST_REF_ARG(g2g1_query_type) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        return __base.template G2_over_G1<LS, Interaction, MicrofacetCache>(query, _sample, interaction, cache);
    }

    base_type __base;
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
