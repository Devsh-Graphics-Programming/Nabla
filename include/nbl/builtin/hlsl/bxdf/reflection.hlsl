// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_REFLECTION_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_REFLECTION_INCLUDED_

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

// still need these?
template<class LightSample, class Iso, class Aniso, class RayDirInfo, typename Scalar 
    NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso> && surface_interactions::Anisotropic<Aniso> && ray_dir_info::Basic<RayDirInfo> && is_scalar_v<Scalar>)
LightSample cos_generate(NBL_CONST_REF_ARG(Iso) interaction)
{
    return LightSample(interaction.V.reflect(interaction.N,interaction.NdotV),interaction.NdotV,interaction.N);
}
template<class LightSample, class Iso, class Aniso, class RayDirInfo, typename Scalar 
    NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso> && surface_interactions::Anisotropic<Aniso> && ray_dir_info::Basic<RayDirInfo> && is_scalar_v<Scalar>)
LightSample cos_generate(NBL_CONST_REF_ARG(Aniso) interaction)
{
    return LightSample(interaction.V.reflect(interaction.N,interaction.NdotV),interaction.NdotV,interaction.T,interaction.B,interaction.N);
}

// for information why we don't check the relation between `V` and `L` or `N` and `H`, see comments for `nbl::hlsl::transmission::cos_quotient_and_pdf`
template<typename SpectralBins, typename Pdf NBL_FUNC_REQUIRES(spectral_of<SpectralBins,Pdf> && is_floating_point_v<Pdf>)
quotient_and_pdf<SpectralBins, Pdf> cos_quotient_and_pdf()
{
    return quotient_and_pdf<SpectralBins, Pdf>::create(SpectralBins(1.f), numeric_limits<Pdf>::infinity);
}

// basic bxdfs
template<class LightSample, class Iso, class Aniso, class Spectrum NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso> && surface_interactions::Anisotropic<Aniso>)
struct SLambertianBxDF
{
    using this_t = SLambertianBxDF<LightSample, Iso, Aniso, Spectrum>;
    using scalar_type = typename LightSample::scalar_type;
    using ray_dir_info_type = typename LightSample::ray_dir_info_type;
    using isotropic_type = Iso;
    using anisotropic_type = Aniso;
    using sample_type = LightSample;
    using spectral_type = Spectrum;
    using quotient_pdf_type = quotient_and_pdf<spectral_type, scalar_type>;
    using params_t = SBxDFParams<scalar_type>;

    static this_t create()
    {
        this_t retval;
        // nothing here, just keeping in convention with others
        return retval;
    }

    scalar_type __eval_pi_factored_out(scalar_type maxNdotL)
    {
        return maxNdotL;
    }

    scalar_type eval(NBL_CONST_REF_ARG(params_t) params)
    {
        return __eval_pi_factored_out(params.NdotL) * numbers::inv_pi<scalar_type>;
    }

    sample_type generate_wo_clamps(anisotropic_type interaction, vector<scalar_type, 2> u)
    {
        ray_dir_info_type L;
        L.direction = projected_hemisphere_generate<scalar_type>(u);
        return sample_type::createFromTangentSpace(interaction.getTangentSpaceV(), L, interaction.getFromTangentSpace());
    }

    sample_type generate(anisotropic_type interaction, vector<scalar_type, 2> u)
    {
        return generate_wo_clamps(interaction, u);
    }

    scalar_type pdf(NBL_CONST_REF_ARG(params_t) params)
    {
        return projected_hemisphere_pdf<scalar_type>(params.NdotL);
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(params_t) params)
    {
        scalar_type _pdf;
        scalar_type q = projected_hemisphere_quotient_and_pdf<scalar_type>(_pdf, params.NdotL);
        return quotient_pdf_type::create((spectral_type)(q), _pdf);
    }
};


template<class LightSample, class Iso, class Aniso, class Spectrum NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso> && surface_interactions::Anisotropic<Aniso>)
struct SOrenNayarBxDF
{
    using this_t = SOrenNayarBxDF<LightSample, Iso, Aniso, Spectrum>;
    using scalar_type = typename LightSample::scalar_type;
    using vector2_type = vector<scalar_type, 2>;
    using ray_dir_info_type = typename LightSample::ray_dir_info_type;

    using isotropic_type = Iso;
    using anisotropic_type = Aniso;
    using sample_type = LightSample;
    using spectral_type = Spectrum;
    using quotient_pdf_type = quotient_and_pdf<spectral_type, scalar_type>;
    using params_t = SBxDFParams<scalar_type>;

    static this_t create(scalar_type A)
    {
        this_t retval;
        retval.A = A;
        return retval;
    }

    scalar_type __rec_pi_factored_out_wo_clamps(scalar_type VdotL, scalar_type maxNdotL, scalar_type maxNdotV)
    {
        scalar_type A2 = A * 0.5;
        vector2_type AB = vector2_type(1.0, 0.0) + vector2_type(-0.5, 0.45) * vector2_type(A2, A2) / vector2_type(A2 + 0.33, A2 + 0.09);
        scalar_type C = 1.0 / max<scalar_type>(maxNdotL, maxNdotV);

        scalar_type cos_phi_sin_theta = max<scalar_type>(VdotL - maxNdotL * maxNdotV, 0.0);
        return (AB.x + AB.y * cos_phi_sin_theta * C);
    }

    scalar_type eval(params_t params)
    {
        return params.NdotL * numbers::inv_pi<scalar_type> * __rec_pi_factored_out_wo_clamps(params.VdotL, params.NdotL, params.NdotV);
    }

    sample_type generate_wo_clamps(anisotropic_type interaction, vector2_type u)
    {
        ray_dir_info_type L;
        L.direction = projected_hemisphere_generate<scalar_type>(u);
        return sample_type::createFromTangentSpace(interaction.getTangentSpaceV(), L, interaction.getFromTangentSpace());
    }

    sample_type generate(anisotropic_type interaction, vector2_type u)
    {
        return generate_wo_clamps(interaction, u);
    }

    scalar_type pdf(params_t params)
    {
        return projected_hemisphere_pdf<scalar_type>(params.NdotL);
    }

    quotient_pdf_type quotient_and_pdf(params_t params)
    {
        scalar_type _pdf;
        projected_hemisphere_quotient_and_pdf<scalar_type>(_pdf, params.NdotL);
        scalar_type q = __rec_pi_factored_out_wo_clamps(params.VdotL, params.NdotL, params.NdotV);
        return quotient_pdf_type::create((spectral_type)(q), _pdf);
    }

    scalar_type A;
};


// microfacet bxdfs

// do not use, not tested, also shit
template<class LightSample, class IsoCache, class AnisoCache, class Spectrum NBL_FUNC_REQUIRES(Sample<LightSample> && IsotropicMicrofacetCache<IsoCache> && AnisotropicMicrofacetCache<AnisoCache>)
struct SBlinnPhongBxDF
{
    using this_t = SBlinnPhongBxDF<LightSample, IsoCache, AnisoCache, Spectrum>;
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
    using quotient_pdf_type = quotient_and_pdf<spectral_type, scalar_type>;
    using isocache_type = IsoCache;
    using anisocache_type = AnisoCache;

    static this_t create(vector2_type n, spectral_type ior0, spectral_type ior1)
    {
        this_t retval;
        retval.n = n;
        retval.ior0 = ior0;
        retval.ior1 = ior1;
        return retval;
    }

    template <typename T>
    static T phong_exp_to_alpha2(T n)
    {
        return 2.0 / (n + 2.0);
    }

    template <typename T>
    static T alpha2_to_phong_exp(T a2)
    {
        return 2.0 / a2 - 2.0;
    }

    template<bool aniso>    // this or specialize?
    scalar_type __eval_DG_wo_clamps(params_t params, vector2_type a2)
    {
        if (aniso)
        {
            ndf::SAnisotropicParams<scalar_type> ndfparams = ndf::SAnisotropicParams<scalar_type>::create(params.NdotH, 1.0 / (1.0 - params.NdotH2), params.TdotH2, params.BdotH2, n.x, n.y);
            ndf::BlinnPhong<scalar_type> blinn_phong;
            scalar_type DG = blinn_phong(ndfparams);
            if (any<vector<bool, 2>>(a2 > (vector2_type)numeric_limits<scalar_type>::min))
            {
                smith::SAnisotropicParams<scalar_type> smithparams = smith::SAnisotropicParams<scalar_type>::create(a2.x, a2.y, params.TdotV2, params.BdotV2, params.NdotV2, params.TdotL2, params.BdotL2, params.NdotL2, 0);
                smith::Beckmann<scalar_type> beckmann;
                DG *= beckmann.correlated(smithparams);
            }
            return DG;
        }
        else
        {
            ndf::SIsotropicParams<scalar_type> ndfparams = ndf::SIsotropicParams<scalar_type>::create(n, params.NdotH, params.NdotH2);
            ndf::BlinnPhong<scalar_type> blinn_phong;
            scalar_type NG = blinn_phong(ndfparams);
            if (any<vector<bool, 2>>(a2 > (vector2_type)numeric_limits<scalar_type>::min))
            {
                smith::SIsotropicParams<scalar_type> smithparams = smith::SIsotropicParams<scalar_type>::create(a2.x, params.NdotV2, params.NdotL2, 0);
                smith::Beckmann<scalar_type> beckmann;
                NG *= beckmann.correlated(smithparams);
            }
            return NG;
        }
    }

    template<bool aniso>
    vector3_type __eval_wo_clamps(params_t params)
    {
        scalar_type scalar_part;
        if (aniso)
        {
            vector2_type a2 = phong_exp_to_alpha2<vector2_type>(n);
            scalar_part = __eval_DG_wo_clamps<aniso>(params, a2);
        }
        else
        {
            vector2_type a2 = (vector2_type)phong_exp_to_alpha2<scalar_type>(n);
            scalar_part = __eval_DG_wo_clamps<aniso>(params, a2);
        }
        ndf::microfacet_to_light_measure_transform<ndf::BlinnPhong<scalar_type>,ndf::REFLECT_BIT> microfacet_transform = ndf::microfacet_to_light_measure_transform<ndf::BlinnPhong<scalar_type>,ndf::REFLECT_BIT>::create(scalar_part, params.NdotV);
        return fresnelConductor<scalar_type>(ior0, ior1, params.VdotH) * microfacet_transform();
    }

    vector3_type eval(sample_type _sample, isotropic_type interaction, isocache_type cache)
    {
        if (interaction.NdotV > numeric_limits<scalar_type>::min)
        {
            params_t params = params_t::template create<sample_type, isotropic_type, isocache_type>(_sample, interaction, cache);
            return __eval_wo_clamps<false>(params);
        }
        else
            return (vector3_type)0.0;
    }

    vector3_type eval(sample_type _sample, anisotropic_type interaction, anisocache_type cache)
    {
        if (interaction.NdotV > numeric_limits<scalar_type>::min)
        {
            params_t params = params_t::template create<sample_type, anisotropic_type, anisocache_type>(_sample, interaction, cache);
            return __eval_wo_clamps<true>(params);
        }
        else
            return (vector3_type)0.0;
    }

    vector3_type generate(vector2_type u, scalar_type n)
    {
        scalar_type phi = 2.0 * numbers::pi<scalar_type> * u.y;
        scalar_type cosTheta = pow<scalar_type>(u.x, 1.0/(n+1.0));
        scalar_type sinTheta = sqrt<scalar_type>(1.0 - cosTheta * cosTheta);
        scalar_type cosPhi = cos<scalar_type>(phi);
        scalar_type sinPhi = sin<scalar_type>(phi);
        return vector3_type(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
    }

    sample_type generate(anisotropic_type interaction, vector2_type u, NBL_REF_ARG(anisocache_type) cache)
    {
        const vector3_type H = generate(u, n.x);
        const vector3_type localV = interaction.getTangentSpaceV();

        cache = anisocache_type::create(localV, H);
        ray_dir_info_type localL;
        localL.direction = math::reflect<scalar_type>(localV, H, cache.VdotH);

        return sample_type::createFromTangentSpace(localV, localL, interaction.getFromTangentSpace());
    }

    // where pdf?

    vector2_type n;
    spectral_type ior0, ior1;
};

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
    using quotient_pdf_type = quotient_and_pdf<spectral_type, scalar_type>;
    using isocache_type = IsoCache;
    using anisocache_type = AnisoCache;

    // iso
    static this_t create(scalar_type A, spectral_type ior0, spectral_type ior1)
    {
        this_t retval;
        retval.A = vector2_type(A,A);
        retval.ior0 = ior0;
        retval.ior1 = ior1;
        return retval;
    }

    // aniso
    static this_t create(scalar_type ax, scalar_type ay, spectral_type ior0, spectral_type ior1)
    {
        this_t retval;
        retval.A = vector2_type(ax,ay);
        retval.ior0 = ior0;
        retval.ior1 = ior1;
        return retval;
    }

    scalar_type __eval_DG_wo_clamps(params_t params)
    {
        if (params.is_aniso)
        {
            const scalar_type ax2 = A.x*A.x;
            const scalar_type ay2 = A.y*A.y;
            ndf::SAnisotropicParams<scalar_type> ndfparams = ndf::SAnisotropicParams<scalar_type>::create(A.x, A.y, ax2, ay2, params.TdotH2, params.BdotH2, params.NdotH2);
            ndf::Beckmann<scalar_type> beckmann_ndf;
            scalar_type NG = beckmann_ndf(ndfparams);
            if (any<vector<bool, 2>>(A > (vector2_type)numeric_limits<scalar_type>::min))
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
        if (params.NdotV > numeric_limits<scalar_type>::min)
        {
            scalar_type scalar_part = __eval_DG_wo_clamps(params);
            ndf::microfacet_to_light_measure_transform<ndf::Beckmann<scalar_type>,ndf::REFLECT_BIT> microfacet_transform = ndf::microfacet_to_light_measure_transform<ndf::Beckmann<scalar_type>,ndf::REFLECT_BIT>::create(scalar_part, params.uNdotV);
            return fresnelConductor<spectral_type>(ior0, ior1, params.VdotH) * microfacet_transform();
        }
        else
            return (spectral_type)0.0;
    }

    vector3_type __generate(vector3_type localV, vector2_type u)
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
            while (++it<ITER_THRESHOLD && nbl::hlsl::abs<scalar_type>(value)>MAX_ACCEPTABLE_ERR)
            {
                if (!(b>=a && b<=c))
                    b = 0.5 * (a+c);

                float invErf = erfInv<scalar_type>(b);
                value = normalization * (1.0 + b + numbers::inv_sqrtpi<scalar_type> * tanTheta * exp<scalar_type>(-invErf*invErf)) - sample_x;
                float derivative = normalization * (1.0 - invErf*cosTheta);

                if (value > 0.0)
                    c = b;
                else
                    a = b;

                b -= value/derivative;
            }
            // TODO: investigate if we can replace these two erf^-1 calls with a box muller transform
            slope.x = erfInv<scalar_type>(b);
            slope.y = erfInv<scalar_type>(2.0 * max<scalar_type>(u.y,1.0e-6) - 1.0);
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

    sample_type generate(anisotropic_type interaction, vector2_type u, NBL_REF_ARG(anisocache_type) cache)
    {
        const vector3_type localV = interaction.getTangentSpaceV();
        const vector3_type H = __generate(localV, u);
        
        cache = anisocache_type::create(localV, H);
        ray_dir_info_type localL;
        localL.direction = math::reflect<scalar_type>(localV, H, cache.VdotH);

        return sample_type::createFromTangentSpace(localV, localL, interaction.getFromTangentSpace());
    }

    scalar_type pdf(params_t params, NBL_REF_ARG(scalar_type) onePlusLambda_V)
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

        return smith::VNDF_pdf_wo_clamps<smith::Beckmann<scalar_type> >(ndf, lambda, params.uNdotV, onePlusLambda_V);
    }

    scalar_type pdf(params_t params)
    {
        scalar_type dummy;
        return pdf(params, dummy);
    }

    quotient_pdf_type quotient_and_pdf(params_t params)
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
            const spectral_type reflectance = fresnelConductor<spectral_type>(ior0, ior1, params.VdotH);
            quo = reflectance * G2_over_G1;
        }
        
        return quotient_pdf_type::create(quo, _pdf);
    }

    vector2_type A;
    spectral_type ior0, ior1;
};

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
    using quotient_pdf_type = quotient_and_pdf<spectral_type, scalar_type>;
    using isocache_type = IsoCache;
    using anisocache_type = AnisoCache;

    // iso
    static this_t create(scalar_type A, spectral_type ior0, spectral_type ior1)
    {
        this_t retval;
        retval.A = vector2_type(A,A);
        retval.ior0 = ior0;
        retval.ior1 = ior1;
        return retval;
    }

    // aniso
    static this_t create(scalar_type ax, scalar_type ay, spectral_type ior0, spectral_type ior1)
    {
        this_t retval;
        retval.A = vector2_type(ax,ay);
        retval.ior0 = ior0;
        retval.ior1 = ior1;
        return retval;
    }

    scalar_type __eval_DG_wo_clamps(params_t params)
    {
        if (params.is_aniso)
        {
            const scalar_type ax2 = A.x*A.x;
            const scalar_type ay2 = A.y*A.y;
            ndf::SAnisotropicParams<scalar_type> ndfparams = ndf::SAnisotropicParams<scalar_type>::create(A.x, A.y, ax2, ay2, params.TdotH2, params.BdotH2, params.NdotH2);
            ndf::GGX<scalar_type> ggx_ndf;
            scalar_type NG = ggx_ndf(ndfparams);
            if (any<vector<bool, 2>>(A > (vector2_type)numeric_limits<scalar_type>::min))
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

    spectral_type eval(params_t params)
    {
        if (params.uNdotL > numeric_limits<scalar_type>::min && params.uNdotV > numeric_limits<scalar_type>::min)
        {
            scalar_type scalar_part = __eval_DG_wo_clamps(params);
            ndf::microfacet_to_light_measure_transform<ndf::GGX<scalar_type>,ndf::REFLECT_BIT> microfacet_transform = ndf::microfacet_to_light_measure_transform<ndf::GGX<scalar_type>,ndf::REFLECT_BIT>::create(scalar_part, params.NdotL);
            return fresnelConductor<spectral_type>(ior0, ior1, params.VdotH) * microfacet_transform();
        }
        else
            return (spectral_type)0.0;
    }

    vector3_type __generate(vector3_type localV, vector2_type u)
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

    sample_type generate(anisotropic_type interaction, vector2_type u, NBL_REF_ARG(anisocache_type) cache)
    {
        const vector3_type localV = interaction.getTangentSpaceV();
        const vector3_type H = __generate(localV, u);
        
        cache = anisocache_type::create(localV, H);
        ray_dir_info_type localL;
        localL.direction = math::reflect<scalar_type>(localV, H, cache.VdotH);

        return sample_type::createFromTangentSpace(localV, localL, interaction.getFromTangentSpace());
    }

    scalar_type pdf(params_t params)
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

    quotient_pdf_type quotient_and_pdf(params_t params)
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
                scalar_type G2_over_G1 = ggx_smith.G2_over_G1(smithparams);
            }
            const spectral_type reflectance = fresnelConductor<spectral_type>(ior0, ior1, params.VdotH);
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
