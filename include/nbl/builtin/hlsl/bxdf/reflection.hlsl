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
    return quotient_and_pdf<SpectralBins, Pdf>::create(SpectralBins(1.f), numeric_limits<float>::infinity);
}


template<typename Scalar NBL_PRIMARY_REQUIRES(is_scalar_v<Scalar>)
struct SLambertianBxDF
{
    template<class LightSample, class Iso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso>)    // maybe put template in struct vs function?
    Scalar eval(LightSample _sample, Iso interaction)
    {
        return max(_sample.NdotL, 0.0) * numbers::inv_pi<Scalar>;
    }

    template<class LightSample, class Aniso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Anisotropic<Aniso>)
    LightSample generate(Aniso interaction, vector<Scalar, 2> u)
    {
        vector<Scalar, 3> L = projected_hemisphere_generate<Scalar>(u);
        return LightSample::createTangentSpace(interaction.getTangentSpaceV(), L, interaction.getTangentFrame());
    }

    template<class LightSample, class Iso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso>)
    Scalar pdf(LightSample _sample, Iso interaction)
    {
        return projected_hemisphere_pdf(max(_sample.NdotL, 0.0));
    }

    // TODO: check math here
    template<typename SpectralBins, class LightSample, class Iso NBL_FUNC_REQUIRES(spectral_of<SpectralBins,Pdf> && Sample<LightSample> && surface_interactions::Anisotropic<Aniso>)
    quotient_and_pdf<SpectralBins, Scalar> quotient_and_pdf(LightSample _sample, Iso interaction)
    {
        Scalar pdf;
        Scalar q = projected_hemisphere_quotient_and_pdf<Scalar>(pdf, max(_sample.NdotL, 0.0));
        return quotient_and_pdf<SpectralBins, Pdf>::create(SpectralBins(q), pdf);
    }
};


template<typename Scalar NBL_PRIMARY_REQUIRES(is_scalar_v<Scalar>)
struct SOrenNayarBxDF
{
    using vector_t2 = vector<Scalar, 2>;

    Scalar rec_pi_factored_out_wo_clamps(Scalar VdotL, Scalar maxNdotL, Scalar maxNdotV)
    {
        Scalar A2 = A * 0.5;
        vector_t2 AB = vector_t2(1.0, 0.0) + vector_t2(-0.5, 0.45) * vector_t2(A2, A2) / vector_t2(A2 + 0.33, A2 + 0.09);
        Scalar C = 1.0 / max(maxNdotL, maxNdotV);

        Scalar cos_phi_sin_theta = max(VdotL - maxNdotL * maxNdotV, 0.0);
        return (AB.x + AB.y * cos_phi_sin_theta * C);
    }

    template<class LightSample, class Iso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso>)    // maybe put template in struct vs function?
    Scalar eval(LightSample _sample, Iso interaction)
    {
        return maxNdotL * numbers::inv_pi<Scalar> * rec_pi_factored_out_wo_clamps(_sample.VdotL, max(_sample.NdotL,0.0), max(interaction.NdotV,0.0));
    }

    template<class LightSample, class Aniso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Anisotropic<Aniso>)
    LightSample generate(Aniso interaction, vector<Scalar, 2> u)
    {
        vector<Scalar, 3> L = projected_hemisphere_generate<Scalar>(u);
        return LightSample::createTangentSpace(interaction.getTangentSpaceV(), L, interaction.getTangentFrame());
    }

    template<class LightSample, class Iso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso>)
    Scalar pdf(LightSample _sample, Iso interaction)
    {
        return projected_hemisphere_pdf(max(_sample.NdotL, 0.0));
    }

    // TODO: check math here
    template<typename SpectralBins, class LightSample, class Iso NBL_FUNC_REQUIRES(spectral_of<SpectralBins,Pdf> && Sample<LightSample> && surface_interactions::Anisotropic<Aniso>)
    quotient_and_pdf<SpectralBins, Scalar> quotient_and_pdf(LightSample _sample, Iso interaction)
    {
        Scalar pdf;
        projected_hemisphere_quotient_and_pdf<Scalar>(pdf, max(_sample.NdotL, 0.0));
        Scalar q = rec_pi_factored_out_wo_clamps(_sample.VdotL, max(_sample.NdotL,0.0), max(interaction.NdotV,0.0));
        return quotient_and_pdf<SpectralBins, Pdf>::create(SpectralBins(q), pdf);
    }

    Scalar A;   // set A first before eval
};

template<typename Scalar NBL_PRIMARY_REQUIRES(is_scalar_v<Scalar>)
struct SBlinnPhongBxDF
{
    using vector_t2 = vector<Scalar, 2>;
    using vector_t3 = vector<Scalar, 3>;

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
    Scalar eval_DG_wo_clamps(vector_t2 a2)
    {
        if (aniso)
        {
            Scalar DG = ndf::blinn_phong<Scalar>(NdotH, 1.0 / (1.0 - NdotH2), TdotH2, BdotH2, n.x, n.y);
            if (any(a2 > numeric_limits<float>::min))
                DG *= smith::beckmann_smith_correlated<Scalar>(TdotV2, BdotV2, NdotV2, TdotL2, BdotL2, NdotL2, a2.x, a2.y);
            return DG;
        }
        else
        {
            Scalar NG = ndf::blinn_phong<Scalar>(NdotH, n);
            if (any(a2 > numeric_limits<float>::min))
                NG *= smith::beckmann_smith_correlated<Scalar>(NdotV2, NdotL2, a2.x);
            return NG;
        }
    }

    template<bool aniso>
    vector_t3 eval_wo_clamps()
    {
        Scalar scalar_part;
        if (aniso)
        {
            vector_t2 a2 = phong_exp_to_alpha2<vector_t2>(n);
            scalar_part = eval_DG_wo_clamps<aniso>(NdotH, NdotV2, NdotL2, a2);
        }
        else
        {
            vector_t2 a2 = (vector_t2)phong_exp_to_alpha2<Scalar>(n);
            scalar_part = eval_DG_wo_clamps<aniso>(NdotH, NdotV2, NdotL2, a2);
        }
        return fresnelConductor<Scalar>(ior[0], ior[1], VdotH) * microfacet_to_light_measure_transform<Scalar>(scalar_part, NdotV);        
    }

    template<class LightSample, class Iso, class Cache NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso> && IsotropicMicrofacetCache<Cache>)    // maybe put template in struct vs function?
    vector_t3 eval(LightSample _sample, Iso interaction, Cache cache)
    {
        if (interaction.NdotV > numeric_limits<float>::min)
        {
            // maybe make in static method?
            NdotH = cache.NdotH;
            NdotV = interaction.NdotV;
            NdotV2 = interaction.NdotV2;
            NdotL2 = _sample.NdotL2;
            VdotH = cache.VdotH;
            return eval_wo_clamps<false>();
        }
        else
            return (vector_t3)0.0;
    }

    template<class LightSample, class Aniso, class Cache NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Anisotropic<Iso> && AnisotropicMicrofacetCache<Cache>)    // maybe put template in struct vs function?
    vector_t3 eval(LightSample _sample, Aniso interaction, Cache cache)
    {
        if (interaction.NdotV > numeric_limits<float>::min)
        {
            // maybe make in static method?
            NdotH = cache.NdotH;
            NdotV = interaction.NdotV;
            NdotV2 = interaction.NdotV2;
            NdotL2 = _sample.NdotL2;
            VdotH = cache.VdotH;
            NdotH2 = cache.NdotH2;
            TdotH2 = cache.TdotH * cache.TdotH;
            BdotH2 = cache.BdotH * cache.BdotH;
            TdotL2 = _sample.TdotL * _sample.TdotL;
            BdotL2 = _sample.BdotL * _sample.BdotL;
            TdotV2 = interaction.TdotV * interaction.TdotV;
            BdotV2 = interaction.BdotV * interaction.BdotV;
            return eval_wo_clamps<true>();
        }
        else
            return (vector_t3)0.0;
    }

    vector_t3 generate(vector<Scalar, 2> u, Scalar n)
    {
        Scalar phi = 2.0 * numbers::pi<Scalar>; * u.y;
        Scalar cosTheta = pow(u.x, 1.0/(n+1.0));
        Scalar sinTheta = sqrt(1.0 - cosTheta * cosTheta);
        Scalar cosPhi = cos(phi);
        Scalar sinPhi = sin(phi);
        return vector_t3(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
    }

    template<class LightSample, class Aniso, class Cache NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Anisotropic<Aniso> && AnisotropicMicrofacetCache<Cache>)
    LightSample generate(Aniso interaction, vector<Scalar, 2> u, out Cache cache)
    {
        const vector_t3 H = generate(u, n.x);
        const vector_t3 localV = interaction.getTangentSpaceV();

        cache = Aniso<Scalar>::create(localV, H);
        vector_t3 localL = math::reflect<Scalar>(localV, H, cache.VdotH);

        return LightSample::createTangentSpace(localV, localL, interaction.getTangentFrame());
    }

    // where pdf?

    // set these first before eval
    vector_t2 n;
    matrix<Scalar,3,2> ior;

    // iso
    Scalar NdotH;
    Scalar NdotV;
    Scalar NdotV2;
    Scalar NdotL2;
    Scalar VdotH;

    // aniso
    Scalar NdotH2;
    Scalar TdotH2;
    Scalar BdotH2;
    Scalar TdotL2;
    Scalar BdotL2;
    Scalar TdotV2;
    Scalar BdotV2;
};

template<typename Scalar NBL_PRIMARY_REQUIRES(is_scalar_v<Scalar>)
struct SBeckmannBxDF
{
    using vector_t2 = vector<T,2>;
    using vector_t3 = vector<T,3>;

    template<class LightSample, class Iso, class Cache NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso> && IsotropicMicrofacetCache<Cache>)    // maybe put template in struct vs function?
    vector_t3 eval(LightSample _sample, Iso interaction, Cache cache)
    {
        // TODO
    }

    template<class LightSample, class Aniso, class Cache NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Anisotropic<Iso> && AnisotropicMicrofacetCache<Cache>)    // maybe put template in struct vs function?
    vector_t3 eval(LightSample _sample, Aniso interaction, Cache cache)
    {
        // TODO
    }

    vector_t3 generate(vector_t3 localV, vector_t2 u)
    {
        //stretch
        vector_t3 V = normalize(vector_t3(A.x * localV.x, A.y * localV.y, localV.z));

        vector_t2 slope;
        if (V.z > 0.9999)//V.z=NdotV=cosTheta in tangent space
        {
            Scalar r = sqrt(-log(1.0 - u.x));
            Scalar sinPhi = sin(2.0 * numbers::pi<Scalar> * u.y);
            Scalar cosPhi = cos(2.0 * numbers::pi<Scalar> * u.y);
            slope = (vector_t2)r * vector_t2(cosPhi,sinPhi);
        }
        else
        {
            Scalar cosTheta = V.z;
            Scalar sinTheta = sqrt(1.0 - cosTheta * cosTheta);
            Scalar tanTheta = sinTheta / cosTheta;
            Scalar cotTheta = 1.0 / tanTheta;
            
            Scalar a = -1.0;
            Scalar c = math::erf<Scalar>(cosTheta);
            Scalar sample_x = max(u.x, 1.0e-6);
            Scalar theta = acos(cosTheta);
            Scalar fit = 1.0 + theta * (-0.876 + theta * (0.4265 - 0.0594*theta));
            Scalar b = c - (1.0 + c) * pow(1.0-sample_x, fit);
            
            Scalar normalization = 1.0 / (1.0 + c + numbers::inv_sqrtpi<Scalar> * tanTheta * exp(-cosTheta*cosTheta));

            const int ITER_THRESHOLD = 10;
            const float MAX_ACCEPTABLE_ERR = 1.0e-5;
            int it = 0;
            float value=1000.0;
            while (++it<ITER_THRESHOLD && abs(value)>MAX_ACCEPTABLE_ERR)
            {
                if (!(b>=a && b<=c))
                    b = 0.5 * (a+c);

                float invErf = math::erfInv<Scalar>(b);
                value = normalization * (1.0 + b + numbers::inv_sqrtpi<Scalar> * tanTheta * exp(-invErf*invErf)) - sample_x;
                float derivative = normalization * (1.0 - invErf*cosTheta);

                if (value > 0.0)
                    c = b;
                else
                    a = b;

                b -= value/derivative;
            }
            // TODO: investigate if we can replace these two erf^-1 calls with a box muller transform
            slope.x = math::erfInv<Scalar>(b);
            slope.y = math::erfInv<Scalar>(2.0 * max(u.y,1.0e-6) - 1.0);
        }
        
        Scalar sinTheta = sqrt(1.0 - V.z*V.z);
        Scalar cosPhi = sinTheta==0.0 ? 1.0 : clamp(V.x/sinTheta, -1.0, 1.0);
        Scalar sinPhi = sinTheta==0.0 ? 0.0 : clamp(V.y/sinTheta, -1.0, 1.0);
        //rotate
        Scalar tmp = cosPhi*slope.x - sinPhi*slope.y;
        slope.y = sinPhi*slope.x + cosPhi*slope.y;
        slope.x = tmp;

        //unstretch
        slope = vector_t2(ax,ay)*slope;

        return normalize(vector_t3(-slope, 1.0));
    }

    template<class LightSample, class Aniso, class Cache NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Anisotropic<Aniso> && AnisotropicMicrofacetCache<Cache>)
    LightSample generate(Aniso interaction, vector<Scalar, 2> u, out Cache cache)
    {
        const vector_t3 localV = interaction.getTangentSpaceV();
        const vector_t3 H = generate(localV, u);
        
        cache = Aniso<Scalar>::create(localV, H);
        vector_t3 localL = math::reflect<Scalar>(localV, H, cache.VdotH);

        return LightSample::createTangentSpace(localV, localL, interaction.getTangentFrame());
    }

    template<class LightSample, class Iso, class Cache NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso>, && IsotropicMicrofacetCache<Cache>)
    Scalar pdf(LightSample _sample, Iso interaction, Cache cache)
    {
        Scalar NdotH2 = cache.NdotH2;
        Scalar ndf = ndf::beckmann<Scalar>(A.x*A.x, NdotH2);

        const Scalar lambda = smith::beckmann_Lambda<Scalar>(interaction.NdotV2, A.x*A.x);
        Scalar dummy;
        return smith::VNDF_pdf_wo_clamps<Scalar>(ndf, lambda, interaction.NdotV, dummy);
    }

    template<class LightSample, class Aniso, class Cache NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Anisotropic<Aniso>, && AnisotropicMicrofacetCache<Cache>)
    Scalar pdf(LightSample _sample, Aniso interaction, Cache cache)
    {
        Scalar NdotH2 = cache.NdotH2;
        Scalar ndf = ndf::beckmann<Scalar>(A.x, A.y, A.x*A.x, A.y*A.y, cache.TdotH * cache.TdotH, cache.BdotH * cache.BdotH, NdotH2);

        const Scalar c2 = smith::beckmann_C2<Scalar>(interaction.TdotV * interaction.TdotV, interaction.BdotV * interaction.BdotV, interaction.NdotV2, A.x, A.y);
        Scalar lambda = smith::beckmann_Lambda<Scalar>(c2);
        Scalar dummy;
        return smith::VNDF_pdf_wo_clamps<Scalar>(ndf, lambda, interaction.NdotV, dummy);
    }

    // TODO: remainder_and_pdf funcs

    // set these first before eval
    vector_t2 A;
};

}
}
}
}

#endif
