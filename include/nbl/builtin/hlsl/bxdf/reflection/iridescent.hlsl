// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_REFLECTION_IRIDESCENT_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_REFLECTION_IRIDESCENT_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/reflection/ggx.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace reflection
{

template<class Config NBL_PRIMARY_REQUIRES(config_concepts::MicrofacetConfiguration<Config>)
struct SIridescent
{
    using this_t = SIridescent<Config>;
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

    using ndf_type = ndf::GGX<scalar_type, false>;
    using fresnel_type = fresnel::Iridescent<spectral_type>;
    using measure_transform_type = ndf::SDualMeasureQuant<scalar_type,true,ndf::MTT_REFLECT>;

    NBL_CONSTEXPR_STATIC_INLINE BxDFClampMode _clamp = BxDFClampMode::BCM_MAX;

    struct SCreationParams
    {
        scalar_type A;
        scalar_type thickness;  // thin-film thickness in nm
        spectral_type ior0;
        spectral_type ior1;
        spectral_type ior2;
        spectral_type iork2;
    };
    using creation_type = SCreationParams;

    struct SIridQuery
    {
        using scalar_type = scalar_type;

        scalar_type getDevshV() NBL_CONST_MEMBER_FUNC { return devsh_v; }
        scalar_type getDevshL() NBL_CONST_MEMBER_FUNC { return devsh_l; }

        scalar_type devsh_v;
        scalar_type devsh_l;
    };
    using query_type = SIridQuery;

    static this_t create(scalar_type A, scalar_type thickness, NBL_CONST_REF_ARG(spectral_type) ior0, NBL_CONST_REF_ARG(spectral_type) ior1, NBL_CONST_REF_ARG(spectral_type) ior2, NBL_CONST_REF_ARG(spectral_type) iork2)
    {
        this_t retval;
        retval.__base.ndf.A = vector2_type(A, A);
        retval.__base.ndf.a2 = A*A;
        retval.__base.ndf.one_minus_a2 = scalar_type(1.0) - A*A;
        retval.__base.fresnel.Dinc = thickness;
        retval.__base.fresnel.ior1 = ior0;
        retval.__base.fresnel.ior2 = ior1;
        retval.__base.fresnel.ior3 = ior2;
        retval.__base.fresnel.iork3 = iork2;
        return retval;
    }
    static this_t create(NBL_CONST_REF_ARG(creation_type) params)
    {
        return create(params.A, params.thickness, params.ior0, params.ior1, params.ior2, params.iork2);
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
        if (_sample.getNdotL() > numeric_limits<scalar_type>::min && interaction.getNdotV() > numeric_limits<scalar_type>::min)
        {
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
            dualMeasure.maxNdotL = _sample.getNdotL(_clamp);
            scalar_type DG = dualMeasure.getProjectedLightMeasure();
            fresnel_type f = __base.getFresnel();
            f.absCosTheta = cache.getLdotH();
            return f() * DG;
        }
        else
            return hlsl::promote<spectral_type>(0.0);
    }

    sample_type generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, const vector2_type u, NBL_REF_ARG(isocache_type) cache)
    {
        SGGXAnisotropic<Config> ggx_aniso = SGGXAnisotropic<Config>::create(__base.ndf.A.x, __base.ndf.A.y, __base.fresnel.ior3/__base.fresnel.ior2, __base.fresnel.iork3/__base.fresnel.ior2);
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

            scalar_type ndf;
            scalar_type G1_over_2NdotV;
        };

        SGGXDG1Query dg1_query;
        ndf_type ggx_ndf = __base.getNDF();
        dg1_query.ndf = __base.__D(cache);

        const scalar_type devsh_v = query.getDevshV();
        dg1_query.G1_over_2NdotV = ggx_ndf.G1_wo_numerator_devsh_part(interaction.getNdotV(_clamp), devsh_v);

        measure_transform_type dualMeasure = __base.template __DG1<SGGXDG1Query>(dg1_query);
        return dualMeasure.getMicrofacetMeasure();
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(query_type) query, NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        scalar_type _pdf = pdf(query, interaction, cache);

        spectral_type quo = hlsl::promote<spectral_type>(0.0);
        if (_sample.getNdotL() > numeric_limits<scalar_type>::min && interaction.getNdotV() > numeric_limits<scalar_type>::min)
        {
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
            const scalar_type G2_over_G1 = ggx_ndf.template G2_over_G1<SGGXG2XQuery, sample_type, isotropic_interaction_type, isocache_type>(g2_query, _sample, interaction, cache);
        
            fresnel_type f = __base.getFresnel();
            f.absCosTheta = cache.getLdotH();
            const spectral_type reflectance = f();
            quo = reflectance * G2_over_G1;
        }

        return quotient_pdf_type::create(quo, _pdf);
    }

    SCookTorrance<Config, ndf_type, fresnel_type, measure_transform_type> __base;
};

}

template<typename C>
struct traits<bxdf::reflection::SIridescent<C> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BRDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

}
}
}

#endif
