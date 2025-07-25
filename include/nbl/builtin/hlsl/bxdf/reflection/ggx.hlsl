// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_REFLECTION_GGX_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_REFLECTION_GGX_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
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

template<class LS, class SI, class MC, typename Scalar NBL_STRUCT_CONSTRAINABLE>
struct GGXParams;

template<class LS, class SI, class MC, typename Scalar>
NBL_PARTIAL_REQ_TOP(!surface_interactions::Anisotropic<SI> && !AnisotropicMicrofacetCache<MC>)
struct GGXParams<LS, SI, MC, Scalar NBL_PARTIAL_REQ_BOT(!surface_interactions::Anisotropic<SI> && !AnisotropicMicrofacetCache<MC>) >
{
    using this_t = GGXParams<LS, SI, MC, Scalar>;

    static this_t create(NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(SI) interaction, NBL_CONST_REF_ARG(MC) cache, BxDFClampMode _clamp)
    {
        this_t retval;
        retval._sample = _sample;
        retval.interaction = interaction;
        retval.cache = cache;
        retval._clamp = _clamp;
        return retval;
    }

    // iso
    Scalar getNdotV() NBL_CONST_MEMBER_FUNC { return interaction.getNdotV(_clamp); }
    Scalar getNdotVUnclamped() NBL_CONST_MEMBER_FUNC { return interaction.getNdotV(BxDFClampMode::BCM_NONE); }
    Scalar getNdotV2() NBL_CONST_MEMBER_FUNC { return interaction.getNdotV2(); }
    Scalar getNdotL() NBL_CONST_MEMBER_FUNC { return _sample.getNdotL(_clamp); }
    Scalar getNdotLUnclamped() NBL_CONST_MEMBER_FUNC { return _sample.getNdotL(BxDFClampMode::BCM_NONE); }
    Scalar getNdotL2() NBL_CONST_MEMBER_FUNC { return _sample.getNdotL2(); }
    Scalar getVdotL() NBL_CONST_MEMBER_FUNC { return cache.getVdotL(); }
    Scalar getNdotH() NBL_CONST_MEMBER_FUNC { return cache.getNdotH(); }
    Scalar getNdotH2() NBL_CONST_MEMBER_FUNC { return cache.getNdotH2(); }
    Scalar getVdotH() NBL_CONST_MEMBER_FUNC { return cache.getVdotH(); }
    Scalar getLdotH() NBL_CONST_MEMBER_FUNC { return cache.getLdotH(); }

    LS _sample;
    SI interaction;
    MC cache;
    BxDFClampMode _clamp;
};
template<class LS, class SI, class MC, typename Scalar>
NBL_PARTIAL_REQ_TOP(surface_interactions::Anisotropic<SI> && AnisotropicMicrofacetCache<MC>)
struct GGXParams<LS, SI, MC, Scalar NBL_PARTIAL_REQ_BOT(surface_interactions::Anisotropic<SI> && AnisotropicMicrofacetCache<MC>) >
{
    using this_t = GGXParams<LS, SI, MC, Scalar>;

    static this_t create(NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(SI) interaction, NBL_CONST_REF_ARG(MC) cache, BxDFClampMode _clamp)
    {
        this_t retval;
        retval._sample = _sample;
        retval.interaction = interaction;
        retval.cache = cache;
        retval._clamp = _clamp;
        return retval;
    }

    // iso
    Scalar getNdotV() NBL_CONST_MEMBER_FUNC { return interaction.getNdotV(_clamp); }
    Scalar getNdotVUnclamped() NBL_CONST_MEMBER_FUNC { return interaction.getNdotV(BxDFClampMode::BCM_NONE); }
    Scalar getNdotV2() NBL_CONST_MEMBER_FUNC { return interaction.getNdotV2(); }
    Scalar getNdotL() NBL_CONST_MEMBER_FUNC { return _sample.getNdotL(_clamp); }
    Scalar getNdotLUnclamped() NBL_CONST_MEMBER_FUNC { return _sample.getNdotL(BxDFClampMode::BCM_NONE); }
    Scalar getNdotL2() NBL_CONST_MEMBER_FUNC { return _sample.getNdotL2(); }
    Scalar getVdotL() NBL_CONST_MEMBER_FUNC { return cache.getVdotL(); }
    Scalar getNdotH() NBL_CONST_MEMBER_FUNC { return cache.getNdotH(); }
    Scalar getNdotH2() NBL_CONST_MEMBER_FUNC { return cache.getNdotH2(); }
    Scalar getVdotH() NBL_CONST_MEMBER_FUNC { return cache.getVdotH(); }
    Scalar getLdotH() NBL_CONST_MEMBER_FUNC { return cache.getLdotH(); }

    // aniso
    Scalar getTdotL2() NBL_CONST_MEMBER_FUNC { return _sample.getTdotL2(); }
    Scalar getBdotL2() NBL_CONST_MEMBER_FUNC { return _sample.getBdotL2(); }
    Scalar getTdotV2() NBL_CONST_MEMBER_FUNC { return interaction.getTdotV2(); }
    Scalar getBdotV2() NBL_CONST_MEMBER_FUNC { return interaction.getBdotV2(); }
    Scalar getTdotH2() NBL_CONST_MEMBER_FUNC { return cache.getTdotH2(); }
    Scalar getBdotH2() NBL_CONST_MEMBER_FUNC { return cache.getBdotH2(); }

    LS _sample;
    SI interaction;
    MC cache;
    BxDFClampMode _clamp;
};

template<class LS, class Iso, class Aniso, class IsoCache, class AnisoCache, class Spectrum NBL_PRIMARY_REQUIRES(LightSample<LS> && surface_interactions::Isotropic<Iso> && surface_interactions::Anisotropic<Aniso> && CreatableIsotropicMicrofacetCache<IsoCache> && AnisotropicMicrofacetCache<AnisoCache>)
struct SGGXBxDF
{
    using this_t = SGGXBxDF<LS, Iso, Aniso, IsoCache, AnisoCache, Spectrum>;
    using scalar_type = typename LS::scalar_type;
    using ray_dir_info_type = typename LS::ray_dir_info_type;
    using vector2_type = vector<scalar_type, 2>;
    using vector3_type = vector<scalar_type, 3>;
    using matrix2x3_type = matrix<scalar_type,3,2>;

    using isotropic_interaction_type = Iso;
    using anisotropic_interaction_type = Aniso;
    using sample_type = LS;
    using spectral_type = Spectrum;
    using quotient_pdf_type = sampling::quotient_and_pdf<spectral_type, scalar_type>;
    using isocache_type = IsoCache;
    using anisocache_type = AnisoCache;

    using params_isotropic_t = GGXParams<LS, Iso, IsoCache, scalar_type>;
    using params_anisotropic_t = GGXParams<LS, Aniso, AnisoCache, scalar_type>;


    // iso
    static this_t create(scalar_type A, NBL_CONST_REF_ARG(spectral_type) ior0, NBL_CONST_REF_ARG(spectral_type) ior1)
    {
        this_t retval;
        retval.A = vector2_type(A,A);
        retval.ior0 = ior0;
        retval.ior1 = ior1;
        return retval;
    }

    // aniso
    static this_t create(scalar_type ax, scalar_type ay, NBL_CONST_REF_ARG(spectral_type) ior0, NBL_CONST_REF_ARG(spectral_type) ior1)
    {
        this_t retval;
        retval.A = vector2_type(ax,ay);
        retval.ior0 = ior0;
        retval.ior1 = ior1;
        return retval;
    }

    scalar_type __eval_DG_wo_clamps(NBL_CONST_REF_ARG(params_isotropic_t) params)
    {
        scalar_type a2 = A.x*A.x;
        ndf::GGX<scalar_type, false> ggx_ndf;
        ggx_ndf.a2 = a2;
        ggx_ndf.one_minus_a2 = scalar_type(1.0) - a2;
        scalar_type NG = ggx_ndf.D(params.getNdotH2());
        if (a2 > numeric_limits<scalar_type>::min)
        {
            NG *= ggx_ndf.correlated_wo_numerator(params.getNdotV(), params.getNdotV2(), params.getNdotL(), params.getNdotL2());
        }
        return NG;
    }
    scalar_type __eval_DG_wo_clamps(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        ndf::GGX<scalar_type, true> ggx_ndf;
        ggx_ndf.ax = A.x;
        ggx_ndf.ay = A.y;
        scalar_type NG = ggx_ndf.D(params.getTdotH2(), params.getBdotH2(), params.getNdotH2());
        if (any<vector<bool, 2> >(A > (vector2_type)numeric_limits<scalar_type>::min))
        {
            NG *= ggx_ndf.correlated_wo_numerator(params.getNdotV(), params.getTdotV2(), params.getBdotV2(), params.getNdotV2(), params.getNdotL(), params.getTdotL2(), params.getBdotL2(), params.getNdotL2());
        }
        return NG;
    }

    spectral_type eval(NBL_CONST_REF_ARG(params_isotropic_t) params)
    {
        if (params.getNdotLUnclamped() > numeric_limits<scalar_type>::min && params.getNdotVUnclamped() > numeric_limits<scalar_type>::min)
        {
            const scalar_type scalar_part = __eval_DG_wo_clamps(params);
            const scalar_type microfacet_transform = ndf::microfacet_to_light_measure_transform<scalar_type,true,ndf::MTT_REFLECT>::__call(scalar_part, params.getNdotL());
            fresnel::Conductor<spectral_type> f = fresnel::Conductor<spectral_type>::create(ior0, ior1, params.getVdotH());
            return f() * microfacet_transform;
        }
        else
            return hlsl::promote<spectral_type>(0.0);
    }
    spectral_type eval(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        if (params.getNdotLUnclamped() > numeric_limits<scalar_type>::min && params.getNdotVUnclamped() > numeric_limits<scalar_type>::min)
        {
            const scalar_type scalar_part = __eval_DG_wo_clamps(params);
            const scalar_type microfacet_transform = ndf::microfacet_to_light_measure_transform<scalar_type,true,ndf::MTT_REFLECT>::__call(scalar_part, params.getNdotL());
            fresnel::Conductor<spectral_type> f = fresnel::Conductor<spectral_type>::create(ior0, ior1, params.getVdotH());
            return f() * microfacet_transform;
        }
        else
            return hlsl::promote<spectral_type>(0.0);
    }

    vector3_type __generate(NBL_CONST_REF_ARG(vector3_type) localV, NBL_CONST_REF_ARG(vector2_type) u)
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

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(vector2_type) u, NBL_REF_ARG(anisocache_type) cache)
    {
        const vector3_type localV = interaction.getTangentSpaceV();
        const vector3_type H = __generate(localV, u);

        cache = anisocache_type::createForReflection(localV, H);
        ray_dir_info_type localL;
        bxdf::Reflect<scalar_type> r = bxdf::Reflect<scalar_type>::create(localV, H);
        localL.direction = r(cache.iso_cache.getVdotH());

        return sample_type::createFromTangentSpace(localV, localL, interaction.getFromTangentSpace());
    }

    sample_type generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(vector2_type) u, NBL_REF_ARG(isocache_type) cache)
    {
        anisocache_type anisocache;
        sample_type s = generate(anisotropic_interaction_type::create(interaction), u, anisocache);
        cache = anisocache.iso_cache;
        return s;
    }

    scalar_type pdf(NBL_CONST_REF_ARG(params_isotropic_t) params)
    {
        scalar_type ndf, G1_over_2NdotV;
        const scalar_type a2 = A.x*A.x;
        ndf::GGX<scalar_type, false> ggx_ndf;
        ggx_ndf.a2 = a2;
        ggx_ndf.one_minus_a2 = scalar_type(1.0) - a2;
        ndf = ggx_ndf.D(params.getNdotH2());

        const scalar_type devsh_v = ggx_ndf.devsh_part(params.getNdotV2());
        G1_over_2NdotV = ggx_ndf.G1_wo_numerator_devsh_part(params.getNdotVUnclamped(), devsh_v);

        return ggx_ndf.DG1(ndf, G1_over_2NdotV);
    }
    scalar_type pdf(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        scalar_type ndf, G1_over_2NdotV;
        const scalar_type ax2 = A.x*A.x;
        const scalar_type ay2 = A.y*A.y;
        ndf::GGX<scalar_type, true> ggx_ndf;
        ggx_ndf.ax = A.x;
        ggx_ndf.ay = A.y;
        ndf = ggx_ndf.D(params.getTdotH2(), params.getBdotH2(), params.getNdotH2());

        const scalar_type devsh_v = ggx_ndf.devsh_part(params.getTdotV2(), params.getBdotV2(), params.getNdotV2());
        G1_over_2NdotV = ggx_ndf.G1_wo_numerator_devsh_part(params.getNdotVUnclamped(), devsh_v);

        return ggx_ndf.DG1(ndf, G1_over_2NdotV);
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(params_isotropic_t) params)
    {
        scalar_type _pdf = pdf(params);

        spectral_type quo = (spectral_type)0.0;
        if (params.getNdotLUnclamped() > numeric_limits<scalar_type>::min && params.getNdotVUnclamped() > numeric_limits<scalar_type>::min)
        {
            const scalar_type a2 = A.x*A.x;
            ndf::GGX<scalar_type, false> ggx_ndf;
            ggx_ndf.a2 = a2;
            ggx_ndf.one_minus_a2 = scalar_type(1.0) - a2;
            
            const scalar_type G2_over_G1 = ggx_ndf.G2_over_G1(params.getNdotVUnclamped(), params.getNdotV2(), params.getNdotLUnclamped(), params.getNdotL2());
        
            fresnel::Conductor<spectral_type> f = fresnel::Conductor<spectral_type>::create(ior0, ior1, params.getVdotH());
            const spectral_type reflectance = f();
            quo = reflectance * G2_over_G1;
        }

        return quotient_pdf_type::create(quo, _pdf);
    }
    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        scalar_type _pdf = pdf(params);

        spectral_type quo = (spectral_type)0.0;
        if (params.getNdotLUnclamped() > numeric_limits<scalar_type>::min && params.getNdotVUnclamped() > numeric_limits<scalar_type>::min)
        {
            ndf::GGX<scalar_type, true> ggx_ndf;
            ggx_ndf.ax = A.x;
            ggx_ndf.ay = A.y;

            const scalar_type G2_over_G1 = ggx_ndf.G2_over_G1(params.getNdotVUnclamped(), params.getTdotV2(), params.getBdotV2(), params.getNdotV2(), params.getNdotLUnclamped(), params.getTdotL2(), params.getBdotL2(), params.getNdotL2());

            fresnel::Conductor<spectral_type> f = fresnel::Conductor<spectral_type>::create(ior0, ior1, params.getVdotH());
            const spectral_type reflectance = f();
            quo = reflectance * G2_over_G1;
        }

        return quotient_pdf_type::create(quo, _pdf);
    }

    vector2_type A;
    spectral_type ior0, ior1;
};

}
}
}
}

#endif
