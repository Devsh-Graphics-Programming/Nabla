// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_REFLECTION_BECKMANN_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_REFLECTION_BECKMANN_INCLUDED_

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
struct SBeckmannBxDF
{
    using this_t = SBeckmannBxDF<LightSample, IsoCache, AnisoCache, Spectrum>;
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
            ndf::Beckmann<scalar_type> beckmann_ndf;
            scalar_type NG = beckmann_ndf(ndfparams);
            if (any<vector<bool, 2> >(A > (vector2_type)numeric_limits<scalar_type>::min))
            {
                smith::SAnisotropicParams<scalar_type> smithparams = smith::SAnisotropicParams<scalar_type>::create(ax2, ay2, params.TdotV2, params.BdotV2, params.NdotV2, params.TdotL2, params.BdotL2, params.NdotL2, 0);
                smith::Beckmann<scalar_type> beckmann_smith;
                NG *= beckmann_smith.correlated(smithparams);
            }
            return NG;
        }
        else
        {
            scalar_type a2 = A.x*A.x;
            ndf::SIsotropicParams<scalar_type> ndfparams = ndf::SIsotropicParams<scalar_type>::create(a2, params.NdotH, params.NdotH2);
            ndf::Beckmann<scalar_type> beckmann_ndf;
            scalar_type NG = beckmann_ndf(ndfparams);
            if (a2 > numeric_limits<scalar_type>::min)
            {
                smith::SIsotropicParams<scalar_type> smithparams = smith::SIsotropicParams<scalar_type>::create(a2, params.NdotV2, params.NdotL2, 0);
                smith::Beckmann<scalar_type> beckmann_smith;
                NG *= beckmann_smith.correlated(smithparams);
            }
            return NG;
        }
    }

    spectral_type eval(params_t params)
    {
        if (params.uNdotV > numeric_limits<scalar_type>::min)
        {
            scalar_type scalar_part = __eval_DG_wo_clamps(params);
            ndf::microfacet_to_light_measure_transform<ndf::Beckmann<scalar_type>,ndf::REFLECT_BIT> microfacet_transform = ndf::microfacet_to_light_measure_transform<ndf::Beckmann<scalar_type>,ndf::REFLECT_BIT>::create(scalar_part, params.uNdotV);
            fresnel::Conductor<spectral_type> f = fresnel::Conductor<spectral_type>::create(ior0, ior1, params.VdotH);
            return f() * microfacet_transform();
        }
        else
            return (spectral_type)0.0;
    }

    vector3_type __generate(NBL_CONST_REF_ARG(vector3_type) localV, NBL_CONST_REF_ARG(vector2_type) u)
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

    scalar_type pdf(NBL_CONST_REF_ARG(params_t) params, NBL_REF_ARG(scalar_type) onePlusLambda_V)
    {
        scalar_type ndf, lambda;
        if (params.is_aniso)
        {
            ndf::SAnisotropicParams<scalar_type> ndfparams = ndf::SAnisotropicParams<scalar_type>::create(A.x, A.y, A.x*A.x, A.y*A.y, params.TdotH2, params.BdotH2, params.NdotH2);
            ndf::Beckmann<scalar_type> beckmann_ndf;
            ndf = beckmann_ndf(ndfparams);

            smith::Beckmann<scalar_type> beckmann_smith;
            const scalar_type c2 = beckmann_smith.C2(params.TdotV2, params.BdotV2, params.NdotV2, A.x, A.y);
            lambda = beckmann_smith.Lambda(c2);
        }
        else
        {
            scalar_type a2 = A.x*A.x;
            ndf::SIsotropicParams<scalar_type> ndfparams = ndf::SIsotropicParams<scalar_type>::create(a2, params.NdotH, params.NdotH2);
            ndf::Beckmann<scalar_type> beckmann_ndf;
            ndf = beckmann_ndf(ndfparams);

            smith::Beckmann<scalar_type> beckmann_smith;
            lambda = beckmann_smith.Lambda(params.NdotV2, a2);
        }

        smith::brdf::VNDF_pdf<ndf::Beckmann<scalar_type> > vndf = smith::brdf::VNDF_pdf<ndf::Beckmann<scalar_type> >::create(ndf, params.uNdotV);
        scalar_type _pdf = vndf(lambda);
        onePlusLambda_V = vndf.onePlusLambda_V;

        return _pdf;
    }

    scalar_type pdf(NBL_CONST_REF_ARG(params_t) params)
    {
        scalar_type dummy;
        return pdf(params, dummy);
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(params_t) params)
    {
        scalar_type onePlusLambda_V;
        scalar_type _pdf = pdf(params, onePlusLambda_V);

        smith::Beckmann<scalar_type> beckmann_smith;
        spectral_type quo = (spectral_type)0.0;
        if (params.uNdotL > numeric_limits<scalar_type>::min && params.uNdotV > numeric_limits<scalar_type>::min)
        {
            scalar_type G2_over_G1;
            if (params.is_aniso)
            {
                smith::SAnisotropicParams<scalar_type> smithparams = smith::SAnisotropicParams<scalar_type>::create(A.x*A.x, A.y*A.y, params.TdotV2, params.BdotV2, params.NdotV2, params.TdotL2, params.BdotL2, params.NdotL2, onePlusLambda_V);
                G2_over_G1 = beckmann_smith.G2_over_G1(smithparams);
            }
            else
            {
                smith::SIsotropicParams<scalar_type> smithparams = smith::SIsotropicParams<scalar_type>::create(A.x*A.x, params.NdotV2, params.NdotL2, onePlusLambda_V);
                G2_over_G1 = beckmann_smith.G2_over_G1(smithparams);
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
