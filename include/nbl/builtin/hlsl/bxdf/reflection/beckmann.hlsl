// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_REFLECTION_BECKMANN_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_REFLECTION_BECKMANN_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/bxdf_traits.hlsl"
#include "nbl/builtin/hlsl/sampling/cos_weighted_spheres.hlsl"
#include "nbl/builtin/hlsl/bxdf/ndf/beckmann.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace reflection
{

template<class Config NBL_STRUCT_CONSTRAINABLE>
struct SBeckmannAnisotropicBxDF;

template<class Config NBL_PRIMARY_REQUIRES(config_concepts::MicrofacetConfiguration<Config>)
struct SBeckmannIsotropicBxDF
{
    using this_t = SBeckmannIsotropicBxDF<Config>;
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
    struct SBeckmannDG1Query
    {
        using scalar_type = T;

        scalar_type getNdf() NBL_CONST_MEMBER_FUNC { return ndf; }
        BxDFClampMode getClampMode() NBL_CONST_MEMBER_FUNC { return _clamp; }
        scalar_type getLambdaV() NBL_CONST_MEMBER_FUNC { return lambda_V; }
        scalar_type getOnePlusLambdaV() NBL_CONST_MEMBER_FUNC { return onePlusLambda_V; }

        scalar_type ndf;
        BxDFClampMode _clamp;
        scalar_type lambda_V;
        scalar_type onePlusLambda_V;
    };

    template<typename T>
    struct SBeckmannG2overG1Query
    {
        using scalar_type = T;

        scalar_type getLambdaL() NBL_CONST_MEMBER_FUNC { return lambda_L; }
        scalar_type getOnePlusLambdaV() NBL_CONST_MEMBER_FUNC { return onePlusLambda_V; }

        scalar_type lambda_L;
        scalar_type onePlusLambda_V;
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

    scalar_type __eval_DG_wo_clamps(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        scalar_type a2 = A*A;
        ndf::Beckmann<scalar_type, false> beckmann_ndf;
        beckmann_ndf.a2 = a2;
        scalar_type NG = beckmann_ndf.template D<isocache_type>(cache);
        if (a2 > numeric_limits<scalar_type>::min)
        {
            SBeckmannG2overG1Query<scalar_type> query;
            query.lambda_L = beckmann_ndf.LambdaC2(_sample.getNdotL2());
            query.onePlusLambda_V = scalar_type(1.0) + beckmann_ndf.LambdaC2(interaction.getNdotV2());
            NG *= beckmann_ndf.template correlated<SBeckmannG2overG1Query<scalar_type> >(query);
        }
        return NG;
    }

    spectral_type eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        if (interaction.getNdotV() > numeric_limits<scalar_type>::min)
        {
            const scalar_type scalar_part = __eval_DG_wo_clamps(_sample, interaction, cache);
            const scalar_type microfacet_transform = ndf::microfacet_to_light_measure_transform<scalar_type,false,ndf::MTT_REFLECT>::__call(scalar_part, interaction.getNdotV());
            fresnel::Conductor<spectral_type> f = fresnel::Conductor<spectral_type>::create(ior0, ior1, cache.getVdotH());
            return f() * microfacet_transform;
        }
        else
            return hlsl::promote<spectral_type>(0.0);
    }

    sample_type generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, const vector2_type u, NBL_REF_ARG(isocache_type) cache)
    {
        SBeckmannAnisotropicBxDF<Config> beckmann_aniso = SBeckmannAnisotropicBxDF<Config>::create(A, A, ior0, ior1);
        anisocache_type anisocache;
        sample_type s = beckmann_aniso.generate(anisotropic_interaction_type::create(interaction), u, anisocache);
        cache = anisocache.iso_cache;
        return s;
    }

    scalar_type pdf(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache, NBL_REF_ARG(scalar_type) onePlusLambda_V)
    {
        SBeckmannDG1Query<scalar_type> dg1_query;
        dg1_query._clamp = BxDFClampMode::BCM_MAX;
    
        scalar_type a2 = A*A;
        ndf::Beckmann<scalar_type, false> beckmann_ndf;
        beckmann_ndf.a2 = a2;
        dg1_query.ndf = beckmann_ndf.template D<isocache_type>(cache);

        dg1_query.lambda_V = beckmann_ndf.LambdaC2(interaction.getNdotV2());

        scalar_type dg1 = beckmann_ndf.template DG1<SBeckmannDG1Query<scalar_type>, isotropic_interaction_type>(dg1_query, interaction);
        onePlusLambda_V = dg1_query.getOnePlusLambdaV();
        return dg1;
    }

    scalar_type pdf(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        scalar_type dummy;
        return pdf(interaction, cache, dummy);
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        scalar_type onePlusLambda_V;
        scalar_type _pdf = pdf(interaction, cache, onePlusLambda_V);

        ndf::Beckmann<scalar_type, false> beckmann_ndf;
        spectral_type quo = (spectral_type)0.0;
        if (_sample.getNdotL() > numeric_limits<scalar_type>::min && interaction.getNdotV() > numeric_limits<scalar_type>::min)
        {
            beckmann_ndf.a2 = A*A;
            SBeckmannG2overG1Query<scalar_type> query;
            query.lambda_L = beckmann_ndf.LambdaC2(_sample.getNdotL2());
            query.onePlusLambda_V = onePlusLambda_V;
            scalar_type G2_over_G1 = beckmann_ndf.template G2_over_G1<SBeckmannG2overG1Query<scalar_type>, isocache_type>(query, cache);
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
struct SBeckmannAnisotropicBxDF<Config NBL_PARTIAL_REQ_BOT(config_concepts::MicrofacetConfiguration<Config>) >
{
    using this_t = SBeckmannAnisotropicBxDF<Config>;
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
    struct SBeckmannDG1Query
    {
        using scalar_type = T;

        scalar_type getNdf() NBL_CONST_MEMBER_FUNC { return ndf; }
        BxDFClampMode getClampMode() NBL_CONST_MEMBER_FUNC { return _clamp; }
        scalar_type getLambdaV() NBL_CONST_MEMBER_FUNC { return lambda_V; }
        scalar_type getOnePlusLambdaV() NBL_CONST_MEMBER_FUNC { return onePlusLambda_V; }

        scalar_type ndf;
        BxDFClampMode _clamp;
        scalar_type lambda_V;
        scalar_type onePlusLambda_V;
    };

    template<typename T>
    struct SBeckmannG2overG1Query
    {
        using scalar_type = T;

        scalar_type getLambdaL() NBL_CONST_MEMBER_FUNC { return lambda_L; }
        scalar_type getOnePlusLambdaV() NBL_CONST_MEMBER_FUNC { return onePlusLambda_V; }

        scalar_type lambda_L;
        scalar_type onePlusLambda_V;
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

    scalar_type __eval_DG_wo_clamps(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        ndf::Beckmann<scalar_type, true> beckmann_ndf;
        beckmann_ndf.ax = A.x;
        beckmann_ndf.ay = A.y;
        scalar_type NG = beckmann_ndf.template D<anisocache_type>(cache);
        if (hlsl::any<vector<bool, 2> >(A > (vector2_type)numeric_limits<scalar_type>::min))
        {
            SBeckmannG2overG1Query<scalar_type> query;
            query.lambda_L = beckmann_ndf.LambdaC2(_sample.getTdotL2(), _sample.getBdotL2(), _sample.getNdotL2());
            query.onePlusLambda_V = scalar_type(1.0) + beckmann_ndf.LambdaC2(interaction.getTdotV2(), interaction.getBdotV2(), interaction.getNdotV2());
            NG *= beckmann_ndf.template correlated<SBeckmannG2overG1Query<scalar_type> >(query);
        }
        return NG;
    }

    spectral_type eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        if (interaction.getNdotV() > numeric_limits<scalar_type>::min)
        {
            const scalar_type scalar_part = __eval_DG_wo_clamps(_sample, interaction, cache);
            const scalar_type microfacet_transform = ndf::microfacet_to_light_measure_transform<scalar_type,false,ndf::MTT_REFLECT>::__call(scalar_part, interaction.getNdotV());
            fresnel::Conductor<spectral_type> f = fresnel::Conductor<spectral_type>::create(ior0, ior1, cache.getVdotH());
            return f() * microfacet_transform;
        }
        else
            return hlsl::promote<spectral_type>(0.0);
    }

    vector3_type __generate(NBL_CONST_REF_ARG(vector3_type) localV, const vector2_type u)
    {
        //stretch
        vector3_type V = nbl::hlsl::normalize<vector3_type>(vector3_type(A.x * localV.x, A.y * localV.y, localV.z));

        vector2_type slope;
        if (V.z > 0.9999)//V.z=NdotV=cosTheta in tangent space
        {
            scalar_type r = sqrt<scalar_type>(-log<scalar_type>(1.0 - u.x));
            scalar_type sinPhi = sin<scalar_type>(2.0 * numbers::pi<scalar_type> * u.y);
            scalar_type cosPhi = cos<scalar_type>(2.0 * numbers::pi<scalar_type> * u.y);
            slope = (vector2_type)r * vector2_type(cosPhi,sinPhi);
        }
        else
        {
            scalar_type cosTheta = V.z;
            scalar_type sinTheta = sqrt<scalar_type>(1.0 - cosTheta * cosTheta);
            scalar_type tanTheta = sinTheta / cosTheta;
            scalar_type cotTheta = 1.0 / tanTheta;

            scalar_type a = -1.0;
            scalar_type c = erf<scalar_type>(cosTheta);
            scalar_type sample_x = max<scalar_type>(u.x, 1.0e-6);
            scalar_type theta = acos<scalar_type>(cosTheta);
            scalar_type fit = 1.0 + theta * (-0.876 + theta * (0.4265 - 0.0594*theta));
            scalar_type b = c - (1.0 + c) * pow<scalar_type>(1.0-sample_x, fit);

            scalar_type normalization = 1.0 / (1.0 + c + numbers::inv_sqrtpi<scalar_type> * tanTheta * exp<scalar_type>(-cosTheta*cosTheta));

            const int ITER_THRESHOLD = 10;
            const float MAX_ACCEPTABLE_ERR = 1.0e-5;
            int it = 0;
            float value=1000.0;
            while (++it < ITER_THRESHOLD && nbl::hlsl::abs<scalar_type>(value) > MAX_ACCEPTABLE_ERR)
            {
                if (!(b >= a && b <= c))
                    b = 0.5 * (a + c);

                float invErf = erfInv<scalar_type>(b);
                value = normalization * (1.0 + b + numbers::inv_sqrtpi<scalar_type> * tanTheta * exp<scalar_type>(-invErf * invErf)) - sample_x;
                float derivative = normalization * (1.0 - invErf * cosTheta);

                if (value > 0.0)
                    c = b;
                else
                    a = b;

                b -= value/derivative;
            }
            // TODO: investigate if we can replace these two erf^-1 calls with a box muller transform
            slope.x = erfInv<scalar_type>(b);
            slope.y = erfInv<scalar_type>(2.0 * max<scalar_type>(u.y, 1.0e-6) - 1.0);
        }

        scalar_type sinTheta = sqrt<scalar_type>(1.0 - V.z*V.z);
        scalar_type cosPhi = sinTheta==0.0 ? 1.0 : clamp<scalar_type>(V.x/sinTheta, -1.0, 1.0);
        scalar_type sinPhi = sinTheta==0.0 ? 0.0 : clamp<scalar_type>(V.y/sinTheta, -1.0, 1.0);
        //rotate
        scalar_type tmp = cosPhi*slope.x - sinPhi*slope.y;
        slope.y = sinPhi*slope.x + cosPhi*slope.y;
        slope.x = tmp;

        //unstretch
        slope = vector2_type(A.x,A.y)*slope;

        return nbl::hlsl::normalize<vector3_type>(vector3_type(-slope, 1.0));
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

    scalar_type pdf(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache, NBL_REF_ARG(scalar_type) onePlusLambda_V)
    {
        SBeckmannDG1Query<scalar_type> dg1_query;
        dg1_query._clamp = BxDFClampMode::BCM_MAX;

        scalar_type ndf, lambda;
        ndf::Beckmann<scalar_type, true> beckmann_ndf;
        beckmann_ndf.ax = A.x;
        beckmann_ndf.ay = A.y;
        dg1_query.ndf = beckmann_ndf.template D<anisocache_type>(cache);

        dg1_query.lambda_V = beckmann_ndf.LambdaC2(interaction.getTdotV2(), interaction.getBdotV2(), interaction.getNdotV2());

        scalar_type dg1 = beckmann_ndf.template DG1<SBeckmannDG1Query<scalar_type>, anisotropic_interaction_type>(dg1_query, interaction);
        onePlusLambda_V = dg1_query.getOnePlusLambdaV();
        return dg1;
    }

    scalar_type pdf(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        scalar_type dummy;
        return pdf(interaction, cache, dummy);
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        scalar_type onePlusLambda_V;
        scalar_type _pdf = pdf(interaction, cache, onePlusLambda_V);

        ndf::Beckmann<scalar_type, true> beckmann_ndf;
        spectral_type quo = (spectral_type)0.0;
        if (_sample.getNdotL() > numeric_limits<scalar_type>::min && interaction.getNdotV() > numeric_limits<scalar_type>::min)
        {
            beckmann_ndf.ax = A.x;
            beckmann_ndf.ay = A.y;
            SBeckmannG2overG1Query<scalar_type> query;
            query.lambda_L = beckmann_ndf.LambdaC2(_sample.getTdotL2(), _sample.getBdotL2(), _sample.getNdotL2());
            query.onePlusLambda_V = onePlusLambda_V;
            scalar_type G2_over_G1 = beckmann_ndf.template G2_over_G1<SBeckmannG2overG1Query<scalar_type>, anisocache_type>(query, cache);
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
struct traits<bxdf::reflection::SBeckmannIsotropicBxDF<C> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BRDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

template<typename C>
struct traits<bxdf::reflection::SBeckmannAnisotropicBxDF<C> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BRDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

}
}
}

#endif
