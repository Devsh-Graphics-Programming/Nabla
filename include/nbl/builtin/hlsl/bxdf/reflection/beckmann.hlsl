// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_REFLECTION_BECKMANN_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_REFLECTION_BECKMANN_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/bxdf_traits.hlsl"
#include "nbl/builtin/hlsl/sampling/cos_weighted_spheres.hlsl"
#include "nbl/builtin/hlsl/bxdf/ndf/beckmann.hlsl"
#include "nbl/builtin/hlsl/bxdf/cook_torrance_base.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace reflection
{

template<class Config NBL_STRUCT_CONSTRAINABLE>
struct SBeckmannAnisotropic;

template<class Config NBL_PRIMARY_REQUIRES(config_concepts::MicrofacetConfiguration<Config>)
struct SBeckmannIsotropic
{
    using this_t = SBeckmannIsotropic<Config>;
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

    using ndf_type = ndf::Beckmann<scalar_type, false>;
    using fresnel_type = fresnel::Conductor<spectral_type>;
    using measure_transform_type = ndf::SDualMeasureQuant<scalar_type,false,ndf::MTT_REFLECT>;

    NBL_CONSTEXPR_STATIC_INLINE BxDFClampMode _clamp = BxDFClampMode::BCM_MAX;

    struct SCreationParams
    {
        scalar_type A;
        spectral_type ior0;
        spectral_type ior1;
    };
    using creation_type = SCreationParams;

    struct SBeckmannQuery
    {
        using scalar_type = scalar_type;

        scalar_type getLambdaL() NBL_CONST_MEMBER_FUNC { return lambda_L; }
        scalar_type getLambdaV() NBL_CONST_MEMBER_FUNC { return lambda_V; }

        scalar_type lambda_L;
        scalar_type lambda_V;
    };
    using query_type = SBeckmannQuery;

    static this_t create(scalar_type A, NBL_CONST_REF_ARG(spectral_type) ior0, NBL_CONST_REF_ARG(spectral_type) ior1)
    {
        this_t retval;
        retval.__base.ndf.A = vector2_type(A, A);
        retval.__base.ndf.a2 = A*A;
        retval.__base.fresnel.eta = ior0;
        retval.__base.fresnel.etak = ior1;
        retval.__base.fresnel.etak2 = ior1*ior1;
        return retval;
    }
    static this_t create(NBL_CONST_REF_ARG(creation_type) params)
    {
        return create(params.A, params.ior0, params.ior1);
    }

    query_type createQuery(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction)
    {
        query_type query;
        ndf_type beckmann_ndf = __base.ndf;
        query.lambda_L = beckmann_ndf.LambdaC2(_sample.getNdotL2());
        query.lambda_V = beckmann_ndf.LambdaC2(interaction.getNdotV2());
        return query;
    }

    spectral_type eval(NBL_CONST_REF_ARG(query_type) query, NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        if (interaction.getNdotV() > numeric_limits<scalar_type>::min)
        {
            struct SBeckmannG2overG1Query
            {
                using scalar_type = scalar_type;

                scalar_type getLambdaL() NBL_CONST_MEMBER_FUNC { return lambda_L; }
                scalar_type getLambdaV() NBL_CONST_MEMBER_FUNC { return lambda_V; }

                scalar_type lambda_L;
                scalar_type lambda_V;
            };

            SBeckmannG2overG1Query g2_query;
            g2_query.lambda_L = query.getLambdaL();
            g2_query.lambda_V = query.getLambdaV();

            measure_transform_type dualMeasure = __base.template __DG<SBeckmannG2overG1Query>(g2_query, _sample, interaction, cache);
            dualMeasure.maxNdotV = interaction.getNdotV(_clamp);
            scalar_type DG = dualMeasure.getProjectedLightMeasure();
            return __base.fresnel(cache.getVdotH()) * DG;
        }
        else
            return hlsl::promote<spectral_type>(0.0);
    }

    sample_type generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, const vector2_type u, NBL_REF_ARG(isocache_type) cache)
    {
        SBeckmannAnisotropic<Config> beckmann_aniso = SBeckmannAnisotropic<Config>::create(__base.ndf.A.x, __base.ndf.A.y, __base.fresnel.eta, __base.fresnel.etak);
        anisocache_type anisocache;
        sample_type s = beckmann_aniso.generate(anisotropic_interaction_type::create(interaction), u, anisocache);
        cache = anisocache.iso_cache;
        return s;
    }

    scalar_type pdf(NBL_CONST_REF_ARG(query_type) query, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        struct SBeckmannDG1Query
        {
            using scalar_type = scalar_type;

            scalar_type getNdf() NBL_CONST_MEMBER_FUNC { return ndf; }
            scalar_type getLambdaV() NBL_CONST_MEMBER_FUNC { return lambda_V; }

            scalar_type ndf;
            scalar_type lambda_V;
        };
    
        ndf_type beckmann_ndf = __base.ndf;

        SBeckmannDG1Query dg1_query;
        dg1_query.ndf = __base.__D(cache);
        dg1_query.lambda_V = query.getLambdaV();

        measure_transform_type dualMeasure = __base.template __DG1<SBeckmannDG1Query>(dg1_query);
        dualMeasure.maxNdotV = interaction.getNdotV(_clamp);
        return dualMeasure.getProjectedLightMeasure();
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(query_type) query, NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        scalar_type _pdf = pdf(query, interaction, cache);

        spectral_type quo = hlsl::promote<spectral_type>(0.0);
        if (_sample.getNdotL() > numeric_limits<scalar_type>::min && interaction.getNdotV() > numeric_limits<scalar_type>::min)
        {
            struct SBeckmannG2overG1Query
            {
                using scalar_type = scalar_type;

                scalar_type getLambdaL() NBL_CONST_MEMBER_FUNC { return lambda_L; }
                scalar_type getLambdaV() NBL_CONST_MEMBER_FUNC { return lambda_V; }

                scalar_type lambda_L;
                scalar_type lambda_V;
            };

            ndf_type beckmann_ndf = __base.ndf;
            SBeckmannG2overG1Query g2_query;
            g2_query.lambda_L = query.getLambdaL();
            g2_query.lambda_V = query.getLambdaV();
            scalar_type G2_over_G1 = beckmann_ndf.template G2_over_G1<SBeckmannG2overG1Query, sample_type, isotropic_interaction_type, isocache_type>(g2_query, _sample, interaction, cache);
            const spectral_type reflectance = __base.fresnel(cache.getVdotH());
            quo = reflectance * G2_over_G1;
        }

        return quotient_pdf_type::create(quo, _pdf);
    }

    SCookTorrance<Config, ndf_type, fresnel_type, measure_transform_type> __base;
};

template<class Config>
NBL_PARTIAL_REQ_TOP(config_concepts::MicrofacetConfiguration<Config>)
struct SBeckmannAnisotropic<Config NBL_PARTIAL_REQ_BOT(config_concepts::MicrofacetConfiguration<Config>) >
{
    using this_t = SBeckmannAnisotropic<Config>;
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

    using ndf_type = ndf::Beckmann<scalar_type, true>;
    using fresnel_type = fresnel::Conductor<spectral_type>;
    using measure_transform_type = ndf::SDualMeasureQuant<scalar_type,false,ndf::MTT_REFLECT>;

    NBL_CONSTEXPR_STATIC_INLINE BxDFClampMode _clamp = BxDFClampMode::BCM_MAX;

    struct SCreationParams
    {
        scalar_type ax;
        scalar_type ay;
        spectral_type ior0;
        spectral_type ior1;
    };
    using creation_type = SCreationParams;

    struct SBeckmannQuery
    {
        using scalar_type = scalar_type;

        scalar_type getLambdaL() NBL_CONST_MEMBER_FUNC { return lambda_L; }
        scalar_type getLambdaV() NBL_CONST_MEMBER_FUNC { return lambda_V; }

        scalar_type lambda_L;
        scalar_type lambda_V;
    };
    using query_type = SBeckmannQuery;

    static this_t create(scalar_type ax, scalar_type ay, NBL_CONST_REF_ARG(spectral_type) ior0, NBL_CONST_REF_ARG(spectral_type) ior1)
    {
        this_t retval;
        retval.__base.ndf.A = vector2_type(ax, ay);
        retval.__base.ndf.ax2 = ax*ax;
        retval.__base.ndf.ay2 = ay*ay;
        retval.__base.fresnel.eta = ior0;
        retval.__base.fresnel.etak = ior1;
        retval.__base.fresnel.etak2 = ior1*ior1;
        return retval;
    }
    static this_t create(NBL_CONST_REF_ARG(creation_type) params)
    {
        return create(params.ax, params.ay, params.ior0, params.ior1);
    }

    query_type createQuery(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction)
    {
        query_type query;
        ndf_type beckmann_ndf = __base.ndf;
        query.lambda_L = beckmann_ndf.LambdaC2(_sample.getTdotL2(), _sample.getBdotL2(), _sample.getNdotL2());
        query.lambda_V = beckmann_ndf.LambdaC2(interaction.getTdotV2(), interaction.getBdotV2(), interaction.getNdotV2());
        return query;
    }

    spectral_type eval(NBL_CONST_REF_ARG(query_type) query, NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        if (interaction.getNdotV() > numeric_limits<scalar_type>::min)
        {
            struct SBeckmannG2overG1Query
            {
                using scalar_type = scalar_type;

                scalar_type getLambdaL() NBL_CONST_MEMBER_FUNC { return lambda_L; }
                scalar_type getLambdaV() NBL_CONST_MEMBER_FUNC { return lambda_V; }

                scalar_type lambda_L;
                scalar_type lambda_V;
            };

            SBeckmannG2overG1Query g2_query;
            g2_query.lambda_L = query.getLambdaL();
            g2_query.lambda_V = query.getLambdaV();

            measure_transform_type dualMeasure = __base.template __DG<SBeckmannG2overG1Query>(g2_query, _sample, interaction, cache);
            dualMeasure.maxNdotV = interaction.getNdotV(_clamp);
            scalar_type DG = dualMeasure.getProjectedLightMeasure();
            return __base.fresnel(cache.getVdotH()) * DG;
        }
        else
            return hlsl::promote<spectral_type>(0.0);
    }

    vector3_type __generate(NBL_CONST_REF_ARG(vector3_type) localV, const vector2_type u)
    {
        vector2_type A = __base.ndf.A;
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

    scalar_type pdf(NBL_CONST_REF_ARG(query_type) query, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        struct SBeckmannDG1Query
        {
            using scalar_type = scalar_type;

            scalar_type getNdf() NBL_CONST_MEMBER_FUNC { return ndf; }
            scalar_type getLambdaV() NBL_CONST_MEMBER_FUNC { return lambda_V; }

            scalar_type ndf;
            scalar_type lambda_V;
        };

        ndf_type beckmann_ndf = __base.ndf;

        SBeckmannDG1Query dg1_query;
        dg1_query.ndf = __base.__D(cache);
        dg1_query.lambda_V = query.getLambdaV();

        measure_transform_type dualMeasure = __base.template __DG1<SBeckmannDG1Query>(dg1_query);
        dualMeasure.maxNdotV = interaction.getNdotV(_clamp);
        return dualMeasure.getProjectedLightMeasure();
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(query_type) query, NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        scalar_type _pdf = pdf(query, interaction, cache);

        spectral_type quo = hlsl::promote<spectral_type>(0.0);
        if (_sample.getNdotL() > numeric_limits<scalar_type>::min && interaction.getNdotV() > numeric_limits<scalar_type>::min)
        {
            struct SBeckmannG2overG1Query
            {
                using scalar_type = scalar_type;

                scalar_type getLambdaL() NBL_CONST_MEMBER_FUNC { return lambda_L; }
                scalar_type getLambdaV() NBL_CONST_MEMBER_FUNC { return lambda_V; }

                scalar_type lambda_L;
                scalar_type lambda_V;
            };

            ndf_type beckmann_ndf = __base.ndf;
            SBeckmannG2overG1Query g2_query;
            g2_query.lambda_L = query.getLambdaL();
            g2_query.lambda_V = query.getLambdaV();
            scalar_type G2_over_G1 = beckmann_ndf.template G2_over_G1<SBeckmannG2overG1Query, sample_type, anisotropic_interaction_type, anisocache_type>(g2_query, _sample, interaction, cache);
            const spectral_type reflectance = __base.fresnel(cache.getVdotH());
            quo = reflectance * G2_over_G1;
        }

        return quotient_pdf_type::create(quo, _pdf);
    }

    SCookTorrance<Config, ndf_type, fresnel_type, measure_transform_type> __base;
};

}

template<typename C>
struct traits<bxdf::reflection::SBeckmannIsotropic<C> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BRDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

template<typename C>
struct traits<bxdf::reflection::SBeckmannAnisotropic<C> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BRDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

}
}
}

#endif
