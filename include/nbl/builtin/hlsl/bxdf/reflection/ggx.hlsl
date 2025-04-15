// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_REFLECTION_GGX_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_REFLECTION_GGX_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/sampling/cos_weighted.hlsl"
#include "nbl/builtin/hlsl/bxdf/geom_smith.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace reflection
{

template<class LightSample, class IsoCache, class AnisoCache, class Spectrum NBL_FUNC_REQUIRES(Sample<LightSample> && IsotropicMicrofacetCache<IsoCache> && AnisotropicMicrofacetCache<AnisoCache>)
struct SGGXBxDF
{
    using this_t = SGGXBxDF<LightSample, IsoCache, AnisoCache, Spectrum>;
    using scalar_type = typename LightSample::scalar_type;
    using ray_dir_info_type = typename LightSample::ray_dir_info_type;
    using vector2_type = vector<scalar_type, 2>;
    using vector3_type = vector<scalar_type, 3>;
    using matrix2x3_type = matrix<scalar_type,3,2>;
    using params_t = SBxDFParams<scalar_type>;

    using isotropic_type = typename IsoCache::isotropic_type;
    using anisotropic_type = typename AnisoCache::anisotropic_type;
    using sample_type = LightSample;
    using spectral_type = Spectrum;
    using quotient_pdf_type = sampling::quotient_and_pdf<spectral_type, scalar_type>;
    using isocache_type = IsoCache;
    using anisocache_type = AnisoCache;

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

    static this_t create(NBL_CONST_REF_ARG(SBxDFCreationParams<scalar_type, spectral_type>) params)
    {
        if (params.is_aniso)
            return create(params.A.x, params.A.y, params.ior0, params.ior1);
        else
            return create(params.A.x, params.ior0, params.ior1);
    }

    void init(NBL_CONST_REF_ARG(SBxDFCreationParams<scalar_type, spectral_type>) params)
    {
        A = params.A;
        ior0 = params.ior0;
        ior1 = params.ior1;
    }

    scalar_type __eval_DG_wo_clamps(NBL_CONST_REF_ARG(params_t) params)
    {
        if (params.is_aniso)
        {
            const scalar_type ax2 = A.x*A.x;
            const scalar_type ay2 = A.y*A.y;
            ndf::SAnisotropicParams<scalar_type> ndfparams = ndf::SAnisotropicParams<scalar_type>::create(A.x, A.y, ax2, ay2, params.TdotH2, params.BdotH2, params.NdotH2);
            ndf::GGX<scalar_type> ggx_ndf;
            scalar_type NG = ggx_ndf(ndfparams);
            if (any<vector<bool, 2> >(A > (vector2_type)numeric_limits<scalar_type>::min))
            {
                smith::SAnisotropicParams<scalar_type> smithparams = smith::SAnisotropicParams<scalar_type>::create(ax2, ay2, params.NdotV, params.TdotV2, params.BdotV2, params.NdotV2, params.NdotL, params.TdotL2, params.BdotL2, params.NdotL2);
                smith::GGX<scalar_type> ggx_smith;
                NG *= ggx_smith.correlated_wo_numerator(smithparams);
            }
            return NG;
        }
        else
        {
            scalar_type a2 = A.x*A.x;
            ndf::SIsotropicParams<scalar_type> ndfparams = ndf::SIsotropicParams<scalar_type>::create(a2, params.NdotH, params.NdotH2);
            ndf::GGX<scalar_type> ggx_ndf;
            scalar_type NG = ggx_ndf(ndfparams);
            if (a2 > numeric_limits<scalar_type>::min)
            {
                smith::SIsotropicParams<scalar_type> smithparams = smith::SIsotropicParams<scalar_type>::create(a2, params.NdotV, params.NdotV2, params.NdotL, params.NdotL2);
                smith::GGX<scalar_type> ggx_smith;
                NG *= ggx_smith.correlated_wo_numerator(smithparams);
            }
            return NG;
        }
    }

    spectral_type eval(NBL_CONST_REF_ARG(params_t) params)
    {
        if (params.uNdotL > numeric_limits<scalar_type>::min && params.uNdotV > numeric_limits<scalar_type>::min)
        {
            scalar_type scalar_part = __eval_DG_wo_clamps(params);
            ndf::microfacet_to_light_measure_transform<ndf::GGX<scalar_type>,ndf::REFLECT_BIT> microfacet_transform = ndf::microfacet_to_light_measure_transform<ndf::GGX<scalar_type>,ndf::REFLECT_BIT>::create(scalar_part, params.NdotL);
            fresnel::Conductor<spectral_type> f = fresnel::Conductor<spectral_type>::create(ior0, ior1, params.VdotH);
            return f() * microfacet_transform();
        }
        else
            return (spectral_type)0.0;
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

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_type) interaction, NBL_CONST_REF_ARG(vector2_type) u, NBL_REF_ARG(anisocache_type) cache)
    {
        const vector3_type localV = interaction.getTangentSpaceV();
        const vector3_type H = __generate(localV, u);

        cache = anisocache_type::create(localV, H);
        ray_dir_info_type localL;
        bxdf::Reflect<scalar_type> r = bxdf::Reflect<scalar_type>::create(localV, H, cache.iso_cache.getVdotH());
        localL.direction = r();

        return sample_type::createFromTangentSpace(localV, localL, interaction.getFromTangentSpace());
    }

    scalar_type pdf(NBL_CONST_REF_ARG(params_t) params)
    {
        scalar_type ndf, G1_over_2NdotV;
        if (params.is_aniso)
        {
            const scalar_type ax2 = A.x*A.x;
            const scalar_type ay2 = A.y*A.y;
            ndf::SAnisotropicParams<scalar_type> ndfparams = ndf::SAnisotropicParams<scalar_type>::create(A.x, A.y, ax2, ay2, params.TdotH2, params.BdotH2, params.NdotH2);
            ndf::GGX<scalar_type> ggx_ndf;
            ndf = ggx_ndf(ndfparams);

            smith::GGX<scalar_type> ggx_smith;
            const scalar_type devsh_v = ggx_smith.devsh_part(params.TdotV2, params.BdotV2, params.NdotV2, ax2, ay2);
            G1_over_2NdotV = ggx_smith.G1_wo_numerator(params.uNdotV, devsh_v);
        }
        else
        {
            const scalar_type a2 = A.x*A.x;
            ndf::SIsotropicParams<scalar_type> ndfparams = ndf::SIsotropicParams<scalar_type>::create(a2, params.NdotH, params.NdotH2);
            ndf::GGX<scalar_type> ggx_ndf;
            ndf = ggx_ndf(ndfparams);

            smith::GGX<scalar_type> ggx_smith;
            const scalar_type devsh_v = ggx_smith.devsh_part(params.NdotV2, a2, 1.0-a2);
            G1_over_2NdotV = ggx_smith.G1_wo_numerator(params.uNdotV, devsh_v);
        }
        return smith::VNDF_pdf_wo_clamps<scalar_type>(ndf, G1_over_2NdotV);
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(params_t) params)
    {
        scalar_type _pdf = pdf(params);

        spectral_type quo = (spectral_type)0.0;
        if (params.uNdotL > numeric_limits<scalar_type>::min && params.uNdotV > numeric_limits<scalar_type>::min)
        {
            scalar_type G2_over_G1;
            smith::GGX<scalar_type> ggx_smith;
            if (params.is_aniso)
            {
                const scalar_type ax2 = A.x*A.x;
                const scalar_type ay2 = A.y*A.y;
                smith::SAnisotropicParams<scalar_type> smithparams = smith::SAnisotropicParams<scalar_type>::create(ax2, ay2, params.uNdotV, params.TdotV2, params.BdotV2, params.NdotV2, params.uNdotL, params.TdotL2, params.BdotL2, params.NdotL2);
                G2_over_G1 = ggx_smith.G2_over_G1(smithparams);
            }
            else
            {
                const scalar_type a2 = A.x*A.x;
                smith::SIsotropicParams<scalar_type> smithparams = smith::SIsotropicParams<scalar_type>::create(a2, params.uNdotV, params.NdotV2, params.uNdotL, params.NdotL2);
                G2_over_G1 = ggx_smith.G2_over_G1(smithparams);
            }
            fresnel::Conductor<spectral_type> f = fresnel::Conductor<spectral_type>::create(ior0, ior1, params.VdotH);
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
