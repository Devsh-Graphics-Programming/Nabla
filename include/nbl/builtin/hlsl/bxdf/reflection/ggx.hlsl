// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_REFLECTION_GGX_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_REFLECTION_GGX_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/bxdf_traits.hlsl"
#include "nbl/builtin/hlsl/sampling/cos_weighted_spheres.hlsl"
#include "nbl/builtin/hlsl/bxdf/ndf/ggx.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace reflection
{

template<class Config NBL_STRUCT_CONSTRAINABLE>
struct SGGXAnisotropicBxDF;

template<class Config NBL_PRIMARY_REQUIRES(config_concepts::MicrofacetConfiguration<Config>)
struct SGGXIsotropicBxDF
{
    using this_t = SGGXIsotropicBxDF<Config>;
    NBL_BXDF_CONFIG_ALIAS(scalar_type, Config);
    NBL_BXDF_CONFIG_ALIAS(vector2_type, Config);
    NBL_BXDF_CONFIG_ALIAS(vector3_type, Config);
    NBL_BXDF_CONFIG_ALIAS(ray_dir_info_type, Config);

    NBL_BXDF_CONFIG_ALIAS(isotropic_interaction_type, Config);
    NBL_BXDF_CONFIG_ALIAS(anisotropic_interaction_type, Config);
    NBL_BXDF_CONFIG_ALIAS(sample_type, Config);
    NBL_BXDF_CONFIG_ALIAS(spectral_type, Config);
    NBL_BXDF_CONFIG_ALIAS(quotient_pdf_type, Config);
    NBL_BXDF_CONFIG_ALIAS(isocache_type, Config);
    NBL_BXDF_CONFIG_ALIAS(anisocache_type, Config);

    NBL_CONSTEXPR_STATIC_INLINE BxDFClampMode _clamp = BxDFClampMode::BCM_MAX;

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


    // iso
    static this_t create(scalar_type A, NBL_CONST_REF_ARG(spectral_type) ior0, NBL_CONST_REF_ARG(spectral_type) ior1)
    {
        this_t retval;
        retval.A = A;
        retval.ior0 = ior0;
        retval.ior1 = ior1;
        return retval;
    }

    scalar_type __eval_DG_wo_clamps(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache, BxDFClampMode _clamp)
    {
        scalar_type a2 = A*A;
        ndf::GGX<scalar_type, false> ggx_ndf;
        ggx_ndf.a2 = a2;
        ggx_ndf.one_minus_a2 = scalar_type(1.0) - a2;
        scalar_type NG = ggx_ndf.template D<isocache_type>(cache);
        if (a2 > numeric_limits<scalar_type>::min)
        {
            SGGXG2XQuery<scalar_type> g2_query;
            g2_query.devsh_v = ggx_ndf.devsh_part(interaction.getNdotV2());
            g2_query.devsh_l = ggx_ndf.devsh_part(_sample.getNdotL2());
            g2_query._clamp = _clamp;
            NG *= ggx_ndf.template correlated_wo_numerator<SGGXG2XQuery<scalar_type>, sample_type, isotropic_interaction_type>(g2_query, _sample, interaction);
        }
        return NG;
    }

    spectral_type eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        if (_sample.getNdotL() > numeric_limits<scalar_type>::min && interaction.getNdotV() > numeric_limits<scalar_type>::min)
        {
            const scalar_type scalar_part = __eval_DG_wo_clamps(_sample, interaction, cache, BxDFClampMode::BCM_MAX);
            const scalar_type microfacet_transform = ndf::microfacet_to_light_measure_transform<scalar_type,true,ndf::MTT_REFLECT>::__call(scalar_part, _sample.getNdotL(_clamp));
            fresnel::Conductor<spectral_type> f = fresnel::Conductor<spectral_type>::create(ior0, ior1, cache.getVdotH());
            return f() * microfacet_transform;
        }
        else
            return hlsl::promote<spectral_type>(0.0);
    }

    sample_type generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, const vector2_type u, NBL_REF_ARG(isocache_type) cache)
    {
        SGGXAnisotropicBxDF<Config> ggx_aniso = SGGXAnisotropicBxDF<Config>::create(A, A, ior0, ior1);
        anisocache_type anisocache;
        sample_type s = ggx_aniso.generate(anisotropic_interaction_type::create(interaction), u, anisocache);
        cache = anisocache.iso_cache;
        return s;
    }

    scalar_type pdf(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        SGGXDG1Query<scalar_type> dg1_query;
        const scalar_type a2 = A*A;
        ndf::GGX<scalar_type, false> ggx_ndf;
        ggx_ndf.a2 = a2;
        ggx_ndf.one_minus_a2 = scalar_type(1.0) - a2;
        dg1_query.ndf = ggx_ndf.template D<isocache_type>(cache);

        const scalar_type devsh_v = ggx_ndf.devsh_part(interaction.getNdotV2());
        dg1_query.G1_over_2NdotV = ggx_ndf.G1_wo_numerator_devsh_part(interaction.getNdotV(_clamp), devsh_v);

        return ggx_ndf.template DG1<SGGXDG1Query<scalar_type> >(dg1_query);
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        scalar_type _pdf = pdf(interaction, cache);

        spectral_type quo = (spectral_type)0.0;
        if (_sample.getNdotL() > numeric_limits<scalar_type>::min && interaction.getNdotV() > numeric_limits<scalar_type>::min)
        {
            const scalar_type a2 = A*A;
            ndf::GGX<scalar_type, false> ggx_ndf;
            ggx_ndf.a2 = a2;
            ggx_ndf.one_minus_a2 = scalar_type(1.0) - a2;
            
            SGGXG2XQuery<scalar_type> g2_query;
            g2_query.devsh_v = ggx_ndf.devsh_part(interaction.getNdotV2());
            g2_query.devsh_l = ggx_ndf.devsh_part(_sample.getNdotL2());
            g2_query._clamp = BxDFClampMode::BCM_MAX;
            const scalar_type G2_over_G1 = ggx_ndf.template G2_over_G1<SGGXG2XQuery<scalar_type>, sample_type, isotropic_interaction_type, isocache_type>(g2_query, _sample, interaction, cache);
        
            fresnel::Conductor<spectral_type> f = fresnel::Conductor<spectral_type>::create(ior0, ior1, cache.getVdotH());
            const spectral_type reflectance = f();
            quo = reflectance * G2_over_G1;
        }

        return quotient_pdf_type::create(quo, _pdf);
    }

    scalar_type A;
    spectral_type ior0, ior1;
};

template<class Config>
NBL_PARTIAL_REQ_TOP(config_concepts::MicrofacetConfiguration<Config>)
struct SGGXAnisotropicBxDF<Config NBL_PARTIAL_REQ_BOT(config_concepts::MicrofacetConfiguration<Config>) >
{
    using this_t = SGGXAnisotropicBxDF<Config>;
    NBL_BXDF_CONFIG_ALIAS(scalar_type, Config);
    NBL_BXDF_CONFIG_ALIAS(vector2_type, Config);
    NBL_BXDF_CONFIG_ALIAS(vector3_type, Config);
    NBL_BXDF_CONFIG_ALIAS(ray_dir_info_type, Config);

    NBL_BXDF_CONFIG_ALIAS(isotropic_interaction_type, Config);
    NBL_BXDF_CONFIG_ALIAS(anisotropic_interaction_type, Config);
    NBL_BXDF_CONFIG_ALIAS(sample_type, Config);
    NBL_BXDF_CONFIG_ALIAS(spectral_type, Config);
    NBL_BXDF_CONFIG_ALIAS(quotient_pdf_type, Config);
    NBL_BXDF_CONFIG_ALIAS(isocache_type, Config);
    NBL_BXDF_CONFIG_ALIAS(anisocache_type, Config);

    NBL_CONSTEXPR_STATIC_INLINE BxDFClampMode _clamp = BxDFClampMode::BCM_MAX;

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


    // aniso
    static this_t create(scalar_type ax, scalar_type ay, NBL_CONST_REF_ARG(spectral_type) ior0, NBL_CONST_REF_ARG(spectral_type) ior1)
    {
        this_t retval;
        retval.A = vector2_type(ax,ay);
        retval.ior0 = ior0;
        retval.ior1 = ior1;
        return retval;
    }

    scalar_type __eval_DG_wo_clamps(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache, BxDFClampMode _clamp)
    {
        ndf::GGX<scalar_type, true> ggx_ndf;
        ggx_ndf.ax2 = A.x*A.x;
        ggx_ndf.ay2 = A.y*A.y;
        ggx_ndf.a2 = A.x*A.y;
        scalar_type NG = ggx_ndf.template D<anisocache_type>(cache);
        if (any<vector<bool, 2> >(A > (vector2_type)numeric_limits<scalar_type>::min))
        {
            SGGXG2XQuery<scalar_type> g2_query;
            g2_query.devsh_v = ggx_ndf.devsh_part(interaction.getTdotV2(), interaction.getBdotV2(), interaction.getNdotV2());
            g2_query.devsh_l = ggx_ndf.devsh_part(_sample.getTdotL2(), _sample.getBdotL2(), _sample.getNdotL2());
            g2_query._clamp = _clamp;
            NG *= ggx_ndf.template correlated_wo_numerator<SGGXG2XQuery<scalar_type>, sample_type, anisotropic_interaction_type>(g2_query, _sample, interaction);
        }
        return NG;
    }

    spectral_type eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        if (_sample.getNdotL() > numeric_limits<scalar_type>::min && interaction.getNdotV() > numeric_limits<scalar_type>::min)
        {
            const scalar_type scalar_part = __eval_DG_wo_clamps(_sample, interaction, cache, BxDFClampMode::BCM_MAX);
            const scalar_type microfacet_transform = ndf::microfacet_to_light_measure_transform<scalar_type,true,ndf::MTT_REFLECT>::__call(scalar_part, _sample.getNdotL(_clamp));
            fresnel::Conductor<spectral_type> f = fresnel::Conductor<spectral_type>::create(ior0, ior1, cache.getVdotH());
            return f() * microfacet_transform;
        }
        else
            return hlsl::promote<spectral_type>(0.0);
    }

    vector3_type __generate(NBL_CONST_REF_ARG(vector3_type) localV, const vector2_type u)
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

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, const vector2_type u, NBL_REF_ARG(anisocache_type) cache)
    {
        const vector3_type localV = interaction.getTangentSpaceV();
        const vector3_type H = __generate(localV, u);

        cache = anisocache_type::createForReflection(localV, H);
        ray_dir_info_type localL;
        bxdf::Reflect<scalar_type> r = bxdf::Reflect<scalar_type>::create(localV, H);
        localL.direction = r(cache.iso_cache.getVdotH());

        return sample_type::createFromTangentSpace(localL, interaction.getFromTangentSpace());
    }

    scalar_type pdf(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        SGGXDG1Query<scalar_type> dg1_query;
        ndf::GGX<scalar_type, true> ggx_ndf;
        ggx_ndf.ax2 = A.x*A.x;
        ggx_ndf.ay2 = A.y*A.y;
        ggx_ndf.a2 = A.x*A.y;
        dg1_query.ndf = ggx_ndf.template D<anisocache_type>(cache);

        const scalar_type devsh_v = ggx_ndf.devsh_part(interaction.getTdotV2(), interaction.getBdotV2(), interaction.getNdotV2());
        dg1_query.G1_over_2NdotV = ggx_ndf.G1_wo_numerator_devsh_part(interaction.getNdotV(_clamp), devsh_v);

        return ggx_ndf.template DG1<SGGXDG1Query<scalar_type> >(dg1_query);
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        scalar_type _pdf = pdf(interaction, cache);

        spectral_type quo = (spectral_type)0.0;
        if (_sample.getNdotL() > numeric_limits<scalar_type>::min && interaction.getNdotV() > numeric_limits<scalar_type>::min)
        {
            ndf::GGX<scalar_type, true> ggx_ndf;
            ggx_ndf.ax2 = A.x*A.x;
            ggx_ndf.ay2 = A.y*A.y;
            ggx_ndf.a2 = A.x*A.y;

            SGGXG2XQuery<scalar_type> g2_query;
            g2_query.devsh_v = ggx_ndf.devsh_part(interaction.getTdotV2(), interaction.getBdotV2(), interaction.getNdotV2());
            g2_query.devsh_l = ggx_ndf.devsh_part(_sample.getTdotL2(), _sample.getBdotL2(), _sample.getNdotL2());
            g2_query._clamp = BxDFClampMode::BCM_MAX;
            const scalar_type G2_over_G1 = ggx_ndf.template G2_over_G1<SGGXG2XQuery<scalar_type>, sample_type, anisotropic_interaction_type, anisocache_type>(g2_query, _sample, interaction, cache);

            fresnel::Conductor<spectral_type> f = fresnel::Conductor<spectral_type>::create(ior0, ior1, cache.getVdotH());
            const spectral_type reflectance = f();
            quo = reflectance * G2_over_G1;
        }

        return quotient_pdf_type::create(quo, _pdf);
    }

    vector2_type A;
    spectral_type ior0, ior1;
};

}

template<typename C>
struct traits<bxdf::reflection::SGGXIsotropicBxDF<C> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BRDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

template<typename C>
struct traits<bxdf::reflection::SGGXAnisotropicBxDF<C> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BRDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

}
}
}

#endif
