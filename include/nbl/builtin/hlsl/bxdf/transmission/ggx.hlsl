// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_GGX_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_GGX_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/bxdf_traits.hlsl"
#include "nbl/builtin/hlsl/sampling/cos_weighted_spheres.hlsl"
#include "nbl/builtin/hlsl/bxdf/reflection.hlsl"
#include "nbl/builtin/hlsl/bxdf/cook_torrance_base.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace transmission
{

template<class Config NBL_STRUCT_CONSTRAINABLE>
struct SGGXDielectricAnisotropicBxDF;

template<class Config NBL_PRIMARY_REQUIRES(config_concepts::MicrofacetConfiguration<Config>)
struct SGGXDielectricIsotropicBxDF
{
    using this_t = SGGXDielectricIsotropicBxDF<Config>;
    NBL_BXDF_CONFIG_ALIAS(scalar_type, Config);
    NBL_BXDF_CONFIG_ALIAS(vector2_type, Config);
    NBL_BXDF_CONFIG_ALIAS(vector3_type, Config);
    NBL_BXDF_CONFIG_ALIAS(matrix3x3_type, Config);
    NBL_BXDF_CONFIG_ALIAS(monochrome_type, Config);
    NBL_BXDF_CONFIG_ALIAS(ray_dir_info_type, Config);

    NBL_BXDF_CONFIG_ALIAS(isotropic_interaction_type, Config);
    NBL_BXDF_CONFIG_ALIAS(anisotropic_interaction_type, Config);
    NBL_BXDF_CONFIG_ALIAS(sample_type, Config);
    NBL_BXDF_CONFIG_ALIAS(spectral_type, Config);
    NBL_BXDF_CONFIG_ALIAS(quotient_pdf_type, Config);
    NBL_BXDF_CONFIG_ALIAS(isocache_type, Config);
    NBL_BXDF_CONFIG_ALIAS(anisocache_type, Config);
    using brdf_type = reflection::SGGXIsotropicBxDF<Config>;

    using ndf_type = ndf::GGX<scalar_type, false>;
    using fresnel_type = fresnel::Dielectric<monochrome_type>;
    using measure_transform_type = ndf::SDualMeasureQuant<scalar_type,true,ndf::MTT_REFLECT_REFRACT>;

    NBL_CONSTEXPR_STATIC_INLINE BxDFClampMode _clamp = BxDFClampMode::BCM_ABS;

    struct SGGXQuery
    {
        using scalar_type = scalar_type;

        scalar_type getDevshV() NBL_CONST_MEMBER_FUNC { return devsh_v; }
        scalar_type getDevshL() NBL_CONST_MEMBER_FUNC { return devsh_l; }

        scalar_type devsh_v;
        scalar_type devsh_l;
    };
    using query_type = SGGXQuery;

    static this_t create(scalar_type eta, scalar_type A)
    {
        this_t retval;
        retval.eta = eta;

        retval.__base.ndf.A = vector2_type(A, A);
        retval.__base.ndf.a2 = A*A;
        retval.__base.ndf.one_minus_a2 = scalar_type(1.0) - A*A;
        return retval;
    }

    query_type createQuery(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction)
    {
        query_type query;
        ndf_type ggx_ndf = __base.getNDF();
        query.devsh_v = ggx_ndf.devsh_part(interaction.getNdotV2());
        query.devsh_l = ggx_ndf.devsh_part(_sample.getNdotL2());
        return query;
    }

    spectral_type eval(NBL_CONST_REF_ARG(query_type) query, NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        fresnel::OrientedEtas<monochrome_type> orientedEta = fresnel::OrientedEtas<monochrome_type>::create(cache.getVdotH(), hlsl::promote<monochrome_type>(eta));
        const monochrome_type orientedEta2 = orientedEta.value * orientedEta.value;

        struct SGGXG2XQuery
        {
            using scalar_type = scalar_type;

            scalar_type getDevshV() NBL_CONST_MEMBER_FUNC { return devsh_v; }
            scalar_type getDevshL() NBL_CONST_MEMBER_FUNC { return devsh_l; }
            BxDFClampMode getClampMode() NBL_CONST_MEMBER_FUNC { return _clamp; }

            scalar_type devsh_v;
            scalar_type devsh_l;
            BxDFClampMode _clamp;
        };

        SGGXG2XQuery g2_query;
        g2_query.devsh_v = query.getDevshV();
        g2_query.devsh_l = query.getDevshL();
        g2_query._clamp = _clamp;

        measure_transform_type dualMeasure = __base.template __DG<SGGXG2XQuery>(g2_query, _sample, interaction, cache);
        dualMeasure.absNdotL = _sample.getNdotL(_clamp);
        dualMeasure.orientedEta = orientedEta.value[0];
        scalar_type DG = dualMeasure.getProjectedLightMeasure();

        fresnel_type f = __base.getFresnel();
        f.orientedEta2 = orientedEta2;
        f.absCosTheta = hlsl::abs(cache.getVdotH());
        return hlsl::promote<spectral_type>(f()[0]) * DG;
    }

    sample_type generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_REF_ARG(vector3_type) u, NBL_REF_ARG(isocache_type) cache)
    {
        SGGXDielectricAnisotropicBxDF<Config> ggx_aniso = SGGXDielectricAnisotropicBxDF<Config>::create(eta, __base.ndf.A.x, __base.ndf.A.y);
        anisocache_type anisocache;
        sample_type s = ggx_aniso.generate(anisotropic_interaction_type::create(interaction), u, anisocache);
        cache = anisocache.iso_cache;
        return s;
    }

    scalar_type pdf(NBL_CONST_REF_ARG(query_type) query, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        struct SGGXDG1Query
        {
            using scalar_type = scalar_type;

            scalar_type getNdf() NBL_CONST_MEMBER_FUNC { return ndf; }
            scalar_type getG1over2NdotV() NBL_CONST_MEMBER_FUNC { return G1_over_2NdotV; }
            scalar_type getOrientedEta() NBL_CONST_MEMBER_FUNC { return orientedEta; }

            scalar_type ndf;
            scalar_type G1_over_2NdotV;
            scalar_type orientedEta;
        };

        SGGXDG1Query dg1_query;
        fresnel::OrientedEtas<monochrome_type> orientedEta = fresnel::OrientedEtas<monochrome_type>::create(cache.getVdotH(), hlsl::promote<monochrome_type>(eta));
        const monochrome_type orientedEta2 = orientedEta.value * orientedEta.value;
        dg1_query.orientedEta = orientedEta.value[0];

        const scalar_type reflectance = fresnel::Dielectric<monochrome_type>::__call(orientedEta2, nbl::hlsl::abs<scalar_type>(cache.getVdotH()))[0];

        ndf_type ggx_ndf = __base.getNDF();
        dg1_query.ndf = __base.__D(cache);
        dg1_query.G1_over_2NdotV = ggx_ndf.G1_wo_numerator_devsh_part(interaction.getNdotV(_clamp), query.getDevshV());

        measure_transform_type dualMeasure = __base.template __DG1<SGGXDG1Query>(dg1_query, cache);
        return hlsl::mix(reflectance, scalar_type(1.0) - reflectance, cache.isTransmission()) * dualMeasure.getMicrofacetMeasure();
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(query_type) query, NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        scalar_type _pdf = pdf(query, interaction, cache);
        const bool transmitted = cache.isTransmission();

        struct SGGXG2XQuery
        {
            using scalar_type = scalar_type;

            scalar_type getDevshV() NBL_CONST_MEMBER_FUNC { return devsh_v; }
            scalar_type getDevshL() NBL_CONST_MEMBER_FUNC { return devsh_l; }
            BxDFClampMode getClampMode() NBL_CONST_MEMBER_FUNC { return _clamp; }

            scalar_type devsh_v;
            scalar_type devsh_l;
            BxDFClampMode _clamp;
        };

        ndf_type ggx_ndf = __base.getNDF();
        SGGXG2XQuery g2_query;
        g2_query.devsh_v = query.getDevshV();
        g2_query.devsh_l = query.getDevshL();
        g2_query._clamp = _clamp;

        scalar_type quo;
        quo = ggx_ndf.template G2_over_G1<SGGXG2XQuery, sample_type, isotropic_interaction_type, isocache_type>(g2_query, _sample, interaction, cache);

        return quotient_pdf_type::create(quo, _pdf);
    }

    scalar_type eta;
    SCookTorrance<Config, ndf_type, fresnel_type, measure_transform_type> __base;
};

template<class Config>
NBL_PARTIAL_REQ_TOP(config_concepts::MicrofacetConfiguration<Config>)
struct SGGXDielectricAnisotropicBxDF<Config NBL_PARTIAL_REQ_BOT(config_concepts::MicrofacetConfiguration<Config>) >
{
    using this_t = SGGXDielectricAnisotropicBxDF<Config>;
    NBL_BXDF_CONFIG_ALIAS(scalar_type, Config);
    NBL_BXDF_CONFIG_ALIAS(vector2_type, Config);
    NBL_BXDF_CONFIG_ALIAS(vector3_type, Config);
    NBL_BXDF_CONFIG_ALIAS(matrix3x3_type, Config);
    NBL_BXDF_CONFIG_ALIAS(monochrome_type, Config);
    NBL_BXDF_CONFIG_ALIAS(ray_dir_info_type, Config);

    NBL_BXDF_CONFIG_ALIAS(isotropic_interaction_type, Config);
    NBL_BXDF_CONFIG_ALIAS(anisotropic_interaction_type, Config);
    NBL_BXDF_CONFIG_ALIAS(sample_type, Config);
    NBL_BXDF_CONFIG_ALIAS(spectral_type, Config);
    NBL_BXDF_CONFIG_ALIAS(quotient_pdf_type, Config);
    NBL_BXDF_CONFIG_ALIAS(isocache_type, Config);
    NBL_BXDF_CONFIG_ALIAS(anisocache_type, Config);
    using brdf_type = reflection::SGGXAnisotropicBxDF<Config>;

    using ndf_type = ndf::GGX<scalar_type, true>;
    using fresnel_type = fresnel::Dielectric<monochrome_type>;
    using measure_transform_type = ndf::SDualMeasureQuant<scalar_type,true,ndf::MTT_REFLECT_REFRACT>;

    NBL_CONSTEXPR_STATIC_INLINE BxDFClampMode _clamp = BxDFClampMode::BCM_ABS;

    struct SGGXQuery
    {
        using scalar_type = scalar_type;

        scalar_type getDevshV() NBL_CONST_MEMBER_FUNC { return devsh_v; }
        scalar_type getDevshL() NBL_CONST_MEMBER_FUNC { return devsh_l; }

        scalar_type devsh_v;
        scalar_type devsh_l;
    };
    using query_type = SGGXQuery;

    static this_t create(scalar_type eta, scalar_type ax, scalar_type ay)
    {
        this_t retval;
        retval.eta = eta;

        retval.__base.ndf.A = vector2_type(ax, ay);
        retval.__base.ndf.ax2 = ax*ax;
        retval.__base.ndf.ay2 = ay*ay;
        retval.__base.ndf.a2 = ax*ay;
        return retval;
    }

    query_type createQuery(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction)
    {
        query_type query;
        ndf_type ggx_ndf = __base.getNDF();
        query.devsh_v = ggx_ndf.devsh_part(interaction.getTdotV2(), interaction.getBdotV2(), interaction.getNdotV2());
        query.devsh_l = ggx_ndf.devsh_part(_sample.getTdotL2(), _sample.getBdotL2(), _sample.getNdotL2());
        return query;
    }

    spectral_type eval(NBL_CONST_REF_ARG(query_type) query, NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        fresnel::OrientedEtas<monochrome_type> orientedEta = fresnel::OrientedEtas<monochrome_type>::create(cache.getVdotH(), hlsl::promote<monochrome_type>(eta));
        const monochrome_type orientedEta2 = orientedEta.value * orientedEta.value;

        struct SGGXG2XQuery
        {
            using scalar_type = scalar_type;

            scalar_type getDevshV() NBL_CONST_MEMBER_FUNC { return devsh_v; }
            scalar_type getDevshL() NBL_CONST_MEMBER_FUNC { return devsh_l; }
            BxDFClampMode getClampMode() NBL_CONST_MEMBER_FUNC { return _clamp; }

            scalar_type devsh_v;
            scalar_type devsh_l;
            BxDFClampMode _clamp;
        };

        SGGXG2XQuery g2_query;
        g2_query.devsh_v = query.getDevshV();
        g2_query.devsh_l = query.getDevshL();
        g2_query._clamp = _clamp;

        measure_transform_type dualMeasure = __base.template __DG<SGGXG2XQuery>(g2_query, _sample, interaction, cache);
        dualMeasure.absNdotL = _sample.getNdotL(_clamp);
        dualMeasure.orientedEta = orientedEta.value[0];
        scalar_type DG = dualMeasure.getProjectedLightMeasure();

        fresnel_type f = __base.getFresnel();
        f.orientedEta2 = orientedEta2;
        f.absCosTheta = hlsl::abs(cache.getVdotH());
        return hlsl::promote<spectral_type>(f()[0]) * DG;
    }

    sample_type __generate_wo_clamps(const vector3_type localV, const vector3_type H, NBL_CONST_REF_ARG(matrix3x3_type) m, NBL_REF_ARG(vector3_type) u, NBL_CONST_REF_ARG(fresnel::OrientedEtas<monochrome_type>) orientedEta, NBL_CONST_REF_ARG(fresnel::OrientedEtaRcps<monochrome_type>) rcpEta, NBL_REF_ARG(anisocache_type) cache)
    {
        const scalar_type localVdotH = nbl::hlsl::dot<vector3_type>(localV,H);
        const scalar_type reflectance = fresnel::Dielectric<monochrome_type>::__call(orientedEta.value * orientedEta.value,nbl::hlsl::abs<scalar_type>(localVdotH))[0];
        
        scalar_type rcpChoiceProb;
        bool transmitted = math::partitionRandVariable(reflectance, u.z, rcpChoiceProb);

        cache = anisocache_type::createForReflection(localV, H);

        const scalar_type VdotH = cache.iso_cache.getVdotH();
        Refract<scalar_type> r = Refract<scalar_type>::create(localV, H);
        cache.iso_cache.LdotH = hlsl::mix(VdotH, r.getNdotT(rcpEta.value2[0]), transmitted);
        ray_dir_info_type localL;
        bxdf::ReflectRefract<scalar_type> rr;
        rr.refract = r;
        localL.direction = rr(transmitted, rcpEta.value[0]);

        return sample_type::createFromTangentSpace(localL, m);
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_REF_ARG(vector3_type) u, NBL_REF_ARG(anisocache_type) cache)
    {
        const vector3_type localV = interaction.getTangentSpaceV();

        fresnel::OrientedEtas<monochrome_type> orientedEta = fresnel::OrientedEtas<monochrome_type>::create(interaction.getNdotV(), hlsl::promote<monochrome_type>(eta));
        fresnel::OrientedEtaRcps<monochrome_type> rcpEta = fresnel::OrientedEtaRcps<monochrome_type>::create(interaction.getNdotV(), hlsl::promote<monochrome_type>(eta));

        const vector3_type upperHemisphereV = hlsl::mix(localV, -localV, interaction.getNdotV() < scalar_type(0.0));

        spectral_type dummyior;
        brdf_type ggx = brdf_type::create(__base.ndf.A.x, __base.ndf.A.y, dummyior, dummyior);
        const vector3_type H = ggx.__generate(upperHemisphereV, u.xy);

        return __generate_wo_clamps(localV, H, interaction.getFromTangentSpace(), u, orientedEta, rcpEta, cache);
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_REF_ARG(vector3_type) u)
    {
        anisocache_type dummycache;
        return generate(interaction, u, dummycache);
    }

    scalar_type pdf(NBL_CONST_REF_ARG(query_type) query, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        struct SGGXDG1Query
        {
            using scalar_type = scalar_type;

            scalar_type getNdf() NBL_CONST_MEMBER_FUNC { return ndf; }
            scalar_type getG1over2NdotV() NBL_CONST_MEMBER_FUNC { return G1_over_2NdotV; }
            scalar_type getOrientedEta() NBL_CONST_MEMBER_FUNC { return orientedEta; }

            scalar_type ndf;
            scalar_type G1_over_2NdotV;
            scalar_type orientedEta;
        };

        SGGXDG1Query dg1_query;
        fresnel::OrientedEtas<monochrome_type> orientedEta = fresnel::OrientedEtas<monochrome_type>::create(cache.getVdotH(), hlsl::promote<monochrome_type>(eta));
        const monochrome_type orientedEta2 = orientedEta.value * orientedEta.value;
        dg1_query.orientedEta = orientedEta.value[0];

        const scalar_type reflectance = fresnel::Dielectric<monochrome_type>::__call(orientedEta2, nbl::hlsl::abs<scalar_type>(cache.getVdotH()))[0];

        ndf_type ggx_ndf = __base.getNDF();
        dg1_query.ndf = __base.__D(cache);
        dg1_query.G1_over_2NdotV = ggx_ndf.G1_wo_numerator_devsh_part(interaction.getNdotV(_clamp), query.getDevshV());

        measure_transform_type dualMeasure = __base.template __DG1<SGGXDG1Query>(dg1_query, cache);
        return hlsl::mix(reflectance, scalar_type(1.0) - reflectance, cache.isTransmission()) * dualMeasure.getMicrofacetMeasure();
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(query_type) query, NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        scalar_type _pdf = pdf(query, interaction, cache);
        const bool transmitted = cache.isTransmission();

        struct SGGXG2XQuery
        {
            using scalar_type = scalar_type;

            scalar_type getDevshV() NBL_CONST_MEMBER_FUNC { return devsh_v; }
            scalar_type getDevshL() NBL_CONST_MEMBER_FUNC { return devsh_l; }
            BxDFClampMode getClampMode() NBL_CONST_MEMBER_FUNC { return _clamp; }

            scalar_type devsh_v;
            scalar_type devsh_l;
            BxDFClampMode _clamp;
        };

        ndf_type ggx_ndf = __base.getNDF();
        SGGXG2XQuery g2_query;
        g2_query.devsh_v = query.getDevshV();
        g2_query.devsh_l = query.getDevshL();
        g2_query._clamp = _clamp;

        scalar_type quo;
        quo = ggx_ndf.template G2_over_G1<SGGXG2XQuery, sample_type, anisotropic_interaction_type, anisocache_type>(g2_query, _sample, interaction, cache);

        return quotient_pdf_type::create(quo, _pdf);
    }

    scalar_type eta;
    SCookTorrance<Config, ndf_type, fresnel_type, measure_transform_type> __base;
};

}

template<typename C>
struct traits<bxdf::transmission::SGGXDielectricIsotropicBxDF<C> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BSDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

template<typename C>
struct traits<bxdf::transmission::SGGXDielectricAnisotropicBxDF<C> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BSDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

}
}
}

#endif
