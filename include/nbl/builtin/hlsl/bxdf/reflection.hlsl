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

// new bxdf structure
// static create() method, takes light sample and interaction (and cache) as argument --> fill in _dot_ variables used in later calculations, return bxdf struct
// store them as well?


template<typename Scalar NBL_PRIMARY_REQUIRES(is_scalar_v<Scalar>)
struct SLambertianBxDF
{
    static SLambertianBxDF<Scalar> create()
    {
        SLambertianBxDF<Scalar> retval;
        // nothing here, just keeping in convention with others
        return retval;
    }

    Scalar __eval_pi_factored_out(Scalar maxNdotL)
    {
        return maxNdotL;
    }

    template<class LightSample, class Iso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso>)    // maybe put template in struct vs function?
    Scalar __eval_wo_clamps(LightSample _sample, Iso interaction)
    {
        // probably doesn't need to use the param struct
        return __eval_pi_factored_out(_sample.NdotL) * numbers::inv_pi<Scalar>;
    }

    template<class LightSample, class Iso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso>)    // maybe put template in struct vs function?
    Scalar eval(LightSample _sample, Iso interaction)
    {
        // probably doesn't need to use the param struct
        return __eval_pi_factored_out(max(_sample.NdotL, 0.0)) * numbers::inv_pi<Scalar>;
    }

    template<class LightSample, class Aniso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Anisotropic<Aniso>)
    LightSample generate_wo_clamps(Aniso interaction, vector<Scalar, 2> u)
    {
        vector<Scalar, 3> L = projected_hemisphere_generate<Scalar>(u);
        return LightSample::createTangentSpace(interaction.getTangentSpaceV(), L, interaction.getTangentFrame());
    }

    template<class LightSample, class Aniso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Anisotropic<Aniso>)
    LightSample generate(Aniso interaction, vector<Scalar, 2> u)
    {
        return generate_wo_clamps<LightSample, Aniso>(interaction, u);
    }

    template<class LightSample, class Iso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso>)
    Scalar pdf_wo_clamps(LightSample _sample, Iso interaction)
    {
        return projected_hemisphere_pdf<Scalar>(_sample.NdotL);
    }

    template<class LightSample, class Iso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso>)
    Scalar pdf(LightSample _sample, Iso interaction)
    {
        return projected_hemisphere_pdf<Scalar>(max(_sample.NdotL, 0.0));
    }

    template<typename SpectralBins, class LightSample, class Iso NBL_FUNC_REQUIRES(spectral_of<SpectralBins,Pdf> && Sample<LightSample> && surface_interactions::Anisotropic<Aniso>)
    quotient_and_pdf<SpectralBins, Scalar> quotient_and_pdf_wo_clamps(LightSample _sample, Iso interaction)
    {
        Scalar pdf;
        Scalar q = projected_hemisphere_quotient_and_pdf<Scalar>(pdf, _sample.NdotL);
        return quotient_and_pdf<SpectralBins, Scalar>::create(SpectralBins(q), pdf);
    }

    template<typename SpectralBins, class LightSample, class Iso NBL_FUNC_REQUIRES(spectral_of<SpectralBins,Pdf> && Sample<LightSample> && surface_interactions::Anisotropic<Aniso>)
    quotient_and_pdf<SpectralBins, Scalar> quotient_and_pdf(LightSample _sample, Iso interaction)
    {
        Scalar pdf;
        Scalar q = projected_hemisphere_quotient_and_pdf<Scalar>(pdf, max(_sample.NdotL, 0.0));
        return quotient_and_pdf<SpectralBins, Scalar>::create(SpectralBins(q), pdf);
    }
};


template<typename Scalar NBL_PRIMARY_REQUIRES(is_scalar_v<Scalar>)
struct SOrenNayarBxDF
{
    using this_t = SOrenNayarBxDF<Scalar>;
    using vector_t2 = vector<Scalar, 2>;

    static this_t create(Scalar A)
    {
        this_t retval;
        retval.A = A;
        return retval;
    }

    Scalar __rec_pi_factored_out_wo_clamps(Scalar VdotL, Scalar maxNdotL, Scalar maxNdotV)
    {
        Scalar A2 = A * 0.5;
        vector_t2 AB = vector_t2(1.0, 0.0) + vector_t2(-0.5, 0.45) * vector_t2(A2, A2) / vector_t2(A2 + 0.33, A2 + 0.09);
        Scalar C = 1.0 / max(maxNdotL, maxNdotV);

        Scalar cos_phi_sin_theta = max(VdotL - maxNdotL * maxNdotV, 0.0);
        return (AB.x + AB.y * cos_phi_sin_theta * C);
    }

    template<class LightSample, class Iso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso>)    // maybe put template in struct vs function?
    Scalar __eval_wo_clamps(LightSample _sample, Iso interaction)
    {
        return maxNdotL * numbers::inv_pi<Scalar> * __rec_pi_factored_out_wo_clamps(_sample.VdotL, _sample.NdotL, interaction.NdotV);
    }

    template<class LightSample, class Iso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso>)    // maybe put template in struct vs function?
    Scalar eval(LightSample _sample, Iso interaction)
    {
        return maxNdotL * numbers::inv_pi<Scalar> * __rec_pi_factored_out_wo_clamps(_sample.VdotL, max(_sample.NdotL,0.0), max(interaction.NdotV,0.0));
    }

    template<class LightSample, class Aniso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Anisotropic<Aniso>)
    LightSample generate_wo_clamps(Aniso interaction, vector<Scalar, 2> u)
    {
        vector<Scalar, 3> L = projected_hemisphere_generate<Scalar>(u);
        return LightSample::createTangentSpace(interaction.getTangentSpaceV(), L, interaction.getTangentFrame());
    }

    template<class LightSample, class Aniso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Anisotropic<Aniso>)
    LightSample generate(Aniso interaction, vector<Scalar, 2> u)
    {
        return generate_wo_clamps<LightSample, Aniso>(interaction, u);
    }

    template<class LightSample, class Iso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso>)
    Scalar pdf_wo_clamps(LightSample _sample, Iso interaction)
    {
        return projected_hemisphere_pdf<Scalar>(_sample.NdotL, 0.0);
    }

    template<class LightSample, class Iso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso>)
    Scalar pdf(LightSample _sample, Iso interaction)
    {
        return projected_hemisphere_pdf<Scalar>(max(_sample.NdotL, 0.0));
    }

    // pdf type same as scalar?
    template<typename SpectralBins, class LightSample, class Iso NBL_FUNC_REQUIRES(spectral_of<SpectralBins,Scalar> && Sample<LightSample> && surface_interactions::Anisotropic<Aniso>)
    quotient_and_pdf<SpectralBins, Scalar> quotient_and_pdf_wo_clamps(LightSample _sample, Iso interaction)
    {
        Scalar pdf;
        projected_hemisphere_quotient_and_pdf<Scalar>(pdf, _sample.NdotL);
        Scalar q = __rec_pi_factored_out_wo_clamps(_sample.VdotL, _sample.NdotL, interaction.NdotV);
        return quotient_and_pdf<SpectralBins, Pdf>::create(SpectralBins(q), pdf);
    }

    template<typename SpectralBins, class LightSample, class Iso NBL_FUNC_REQUIRES(spectral_of<SpectralBins,Scalar> && Sample<LightSample> && surface_interactions::Anisotropic<Aniso>)
    quotient_and_pdf<SpectralBins, Scalar> quotient_and_pdf(LightSample _sample, Iso interaction)
    {
        Scalar pdf;
        projected_hemisphere_quotient_and_pdf<Scalar>(pdf, max(_sample.NdotL, 0.0));
        Scalar q = __rec_pi_factored_out_wo_clamps(_sample.VdotL, max(_sample.NdotL,0.0), max(interaction.NdotV,0.0));
        return quotient_and_pdf<SpectralBins, Pdf>::create(SpectralBins(q), pdf);
    }

    Scalar A;
};

template<typename Scalar NBL_PRIMARY_REQUIRES(is_scalar_v<Scalar>)
struct SBlinnPhongBxDF
{
    using this_t = SBlinnPhongBxDF<Scalar>;
    using vector_t2 = vector<Scalar, 2>;
    using vector_t3 = vector<Scalar, 3>;
    using params_t = SBxDFParams<Scalar>;

    static this_t create(vector_t2 n, matrix<Scalar,3,2> ior)
    {
        this_t retval;
        retval.n = n;
        retval.ior = ior;
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
    Scalar __eval_DG_wo_clamps(params_t params, vector_t2 a2)
    {
        if (aniso)
        {
            Scalar DG = ndf::blinn_phong<Scalar>(params.NdotH, 1.0 / (1.0 - params.NdotH2), params.TdotH2, params.BdotH2, n.x, n.y);
            if (any(a2 > numeric_limits<Scalar>::min))
                DG *= smith::beckmann_smith_correlated<Scalar>(params.TdotV2, params.BdotV2, params.NdotV2, params.TdotL2, params.BdotL2, params.NdotL2, a2.x, a2.y);
            return DG;
        }
        else
        {
            Scalar NG = ndf::blinn_phong<Scalar>(params.NdotH, n);
            if (any(a2 > numeric_limits<Scalar>::min))
                NG *= smith::beckmann_smith_correlated<Scalar>(params.NdotV2, params.NdotL2, a2.x);
            return NG;
        }
    }

    template<bool aniso>
    vector_t3 __eval_wo_clamps(params_t params)
    {
        Scalar scalar_part;
        if (aniso)
        {
            vector_t2 a2 = phong_exp_to_alpha2<vector_t2>(n);
            scalar_part = __eval_DG_wo_clamps<aniso>(params, a2);
        }
        else
        {
            vector_t2 a2 = (vector_t2)phong_exp_to_alpha2<Scalar>(n);
            scalar_part = __eval_DG_wo_clamps<aniso>(params, a2);
        }
        return fresnelConductor<Scalar>(ior[0], ior[1], params.VdotH) * microfacet_to_light_measure_transform<Scalar,false>(scalar_part, params.NdotV);        
    }

    template<class LightSample, class Iso, class Cache NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso> && IsotropicMicrofacetCache<Cache>)    // maybe put template in struct vs function?
    vector_t3 eval(LightSample _sample, Iso interaction, Cache cache)
    {
        if (interaction.NdotV > numeric_limits<Scalar>::min)
        {
            params_t params = params_t::template create<LightSample, Iso, Cache>(_sample, interaction, cache);
            return __eval_wo_clamps<false>(params);
        }
        else
            return (vector_t3)0.0;
    }

    template<class LightSample, class Aniso, class Cache NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Anisotropic<Iso> && AnisotropicMicrofacetCache<Cache>)    // maybe put template in struct vs function?
    vector_t3 eval(LightSample _sample, Aniso interaction, Cache cache)
    {
        if (interaction.NdotV > numeric_limits<Scalar>::min)
        {
            params_t params = params_t::template create<LightSample, Aniso, Cache>(_sample, interaction, cache);
            return __eval_wo_clamps<true>(params);
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

    vector_t2 n;
    matrix<Scalar,3,2> ior;
};

template<typename Scalar NBL_PRIMARY_REQUIRES(is_scalar_v<Scalar>)
struct SBeckmannBxDF
{
    using this_t = SBeckmannBxDF<Scalar>;
    using vector_t2 = vector<Scalar,2>;
    using vector_t3 = vector<Scalar,3>;
    using params_t = SBxDFParams<Scalar>;

    // iso
    static this_t create(Scalar A,matrix<Scalar,3,2> ior)
    {
        this_t retval;
        retval.A = vector_t2(A,A);
        retval.ior = ior;
        return retval;
    }

    // aniso
    static this_t create(Scalar ax,Scalar ay,matrix<Scalar,3,2> ior)
    {
        this_t retval;
        retval.A = vector_t2(ax,ay);
        retval.ior = ior;
        return retval;
    }

    template<bool aniso>    // this or specialize?
    Scalar __eval_DG_wo_clamps(params_t params)
    {
        if (aniso)
        {
            const Scalar ax2 = A.x*A.x;
            const Scalar ay2 = A.y*A.y;
            Scalar NG = ndf::beckmann<Scalar>(A.x, A.y, ax2, ay2, params.TdotH2, params.BdotH2, params.NdotH2);
            if (any(A > numeric_limits<Scalar>::min))
                NG *= smith::beckmann_smith_correlated<Scalar>(params.TdotV2, params.BdotV2, params.NdotV2, params.TdotL2, params.BdotL2, params.NdotL2, ax2, ay2);
            return NG;
        }
        else
        {
            Scalar a2 = A.x*A.x;
            Scalar NG = ndf::beckmann<Scalar>(a2, params.NdotH2);
            if (a2 > numeric_limits<Scalar>::min)
                NG *= smith::beckmann_smith_correlated<Scalar>(params.NdotV2, params.NdotL2, a2.x);
            return NG;
        }
    }

    template<bool aniso>
    vector_t3 __eval_wo_clamps(params_t params)
    {
        Scalar scalar_part = __eval_DG_wo_clamps<aniso>(params);
        return fresnelConductor<Scalar>(ior[0], ior[1], params.VdotH) * microfacet_to_light_measure_transform<Scalar,false>(scalar_part, params.NdotV);        
    }

    template<class LightSample, class Iso, class Cache NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso> && IsotropicMicrofacetCache<Cache>)    // maybe put template in struct vs function?
    vector_t3 eval(LightSample _sample, Iso interaction, Cache cache)
    {
        if (interaction.NdotV > numeric_limits<Scalar>::min)
        {
            params_t params = params_t::template create<LightSample, Iso, Cache>(_sample, interaction, cache);
            return __eval_wo_clamps<false>(params);
        }
        else
            return (vector_t3)0.0;
    }

    template<class LightSample, class Aniso, class Cache NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Anisotropic<Iso> && AnisotropicMicrofacetCache<Cache>)    // maybe put template in struct vs function?
    vector_t3 eval(LightSample _sample, Aniso interaction, Cache cache)
    {
        if (interaction.NdotV > numeric_limits<Scalar>::min)
        {
            params_t params = params_t::template create<LightSample, Aniso, Cache>(_sample, interaction, cache);
            return __eval_wo_clamps<true>(params);
        }
        else
            return (vector_t3)0.0;
    }

    vector_t3 __generate(vector_t3 localV, vector_t2 u)
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
        const vector_t3 H = __generate(localV, u);
        
        cache = Aniso<Scalar>::create(localV, H);
        vector_t3 localL = math::reflect<Scalar>(localV, H, cache.VdotH);

        return LightSample::createTangentSpace(localV, localL, interaction.getTangentFrame());
    }

    template<class LightSample, class Iso, class Cache NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso> && IsotropicMicrofacetCache<Cache>)
    Scalar pdf(LightSample _sample, Iso interaction, Cache cache)
    {
        Scalar NdotH2 = cache.NdotH2;
        Scalar ndf = ndf::beckmann<Scalar>(A.x*A.x, NdotH2);

        const Scalar lambda = smith::beckmann_Lambda<Scalar>(interaction.NdotV2, A.x*A.x);
        Scalar dummy;
        return smith::VNDF_pdf_wo_clamps<Scalar>(ndf, lambda, interaction.NdotV, dummy);
    }

    template<class LightSample, class Aniso, class Cache NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Anisotropic<Aniso> && AnisotropicMicrofacetCache<Cache>)
    Scalar pdf(LightSample _sample, Aniso interaction, Cache cache)
    {
        Scalar NdotH2 = cache.NdotH2;
        Scalar ndf = ndf::beckmann<Scalar>(A.x, A.y, A.x*A.x, A.y*A.y, cache.TdotH * cache.TdotH, cache.BdotH * cache.BdotH, NdotH2);

        const Scalar c2 = smith::beckmann_C2<Scalar>(interaction.TdotV * interaction.TdotV, interaction.BdotV * interaction.BdotV, interaction.NdotV2, A.x, A.y);
        Scalar lambda = smith::beckmann_Lambda<Scalar>(c2);
        Scalar dummy;
        return smith::VNDF_pdf_wo_clamps<Scalar>(ndf, lambda, interaction.NdotV, dummy);
    }

    template<typename SpectralBins, class LightSample, class Iso, class Cache NBL_FUNC_REQUIRES(spectral_of<SpectralBins,Scalar> && Sample<LightSample> && surface_interactions::Isotropic<Iso> && IsotropicMicrofacetCache<Cache>)
    quotient_and_pdf<SpectralBins, Scalar> quotient_and_pdf(LightSample _sample, Iso interaction, Cache cache)
    {
        const Scalar ndf = ndf::beckmann<Scalar>(A.x*A.x, cache.NdotH2);
        const Scalar lambda = smith::beckmann_Lambda<Scalar>(interaction.NdotV2, A.x*A.x);

        Scalar onePlusLambda_V;
        Scalar pdf = smith::VNDF_pdf_wo_clamps<Scalar>(ndf, lambda, interaction.NdotV, onePlusLambda_V);
        vector_t3 quo = (vector_t3)0.0;
        if (_sample.NdotL > numeric_limits<Scalar>::min && interaction.NdotV > numeric_limits<Scalar>::min)
        {
            const vector_t3 reflectance = fresnelConductor<Scalar>(ior[0], ior[1], cache.VdotH);
            Scalar G2_over_G1 = smith::beckmann_smith_G2_over_G1<Scalar>(onePlusLambda_V, _sample.NdotL2, a2);
            quo = reflectance * G2_over_G1;
        }
        
        return quotient_and_pdf<SpectralBins, Scalar>::create(SpectralBins(quo), pdf);
    }

    template<typename SpectralBins, class LightSample, class Aniso, class Cache NBL_FUNC_REQUIRES(spectral_of<SpectralBins,Scalar> && Sample<LightSample> && surface_interactions::Anisotropic<Aniso> && AnisotropicMicrofacetCache<Cache>)
    quotient_and_pdf<SpectralBins, Scalar> quotient_and_pdf(LightSample _sample, Aniso interaction, Cache cache)
    {
        params_t params = params_t::template create<LightSample, Aniso, Cache>(_sample, interaction, cache);
        const Scalar ax2 = A.x*A.x;
        const Scalar ay2 = A.y*A.y;

        const Scalar ndf = ndf::beckmann<Scalar>(A.x, A.y, ax2, ay2, params.TdotH2, params.BdotH2, params.NdotH2);
        Scalar onePlusLambda_V;
        const Scalar c2 = smith::beckmann_C2<Scalar>(params.TdotV2, params.BdotV2, params.NdotV2, A.x, A.y);
        Scalar lambda = smith::beckmann_Lambda<Scalar>(c2);
        Scalar pdf = smith::VNDF_pdf_wo_clamps<Scalar>(ndf, lambda, interaction.NdotV, onePlusLambda_V);
        vector_t3 quo = (vector_t3)0.0;
        if (_sample.NdotL > numeric_limits<Scalar>::min && interaction.NdotV > numeric_limits<Scalar>::min)
        {        
            const vector_t3 reflectance = fresnel_conductor<Scalar>(ior[0], ior[1], cache.VdotH);
            Scalar G2_over_G1 = smith::beckmann_smith_G2_over_G1<Scalar>(onePlusLambda_V, params.TdotL2, params.BdotL2, params.NdotL2, A.x, A.y);
            quo = reflectance * G2_over_G1;
        }
        
        return quotient_and_pdf<SpectralBins, Scalar>::create(SpectralBins(quo), pdf);
    }

    vector_t2 A;
    matrix<Scalar,3,2> ior;
};

template<typename Scalar NBL_PRIMARY_REQUIRES(is_scalar_v<Scalar>)
struct SGGXBxDF
{
    using this_t = SBeckmannBxDF<Scalar>;
    using vector_t2 = vector<Scalar,2>;
    using vector_t3 = vector<Scalar,3>;
    using params_t = SBxDFParams<Scalar>;

    // iso
    static this_t create(Scalar A,matrix<Scalar,3,2> ior)
    {
        this_t retval;
        retval.A = vector_t2(A,A);
        retval.ior = ior;
        return retval;
    }

    // aniso
    static this_t create(Scalar ax,Scalar ay,matrix<Scalar,3,2> ior)
    {
        this_t retval;
        retval.A = vector_t2(ax,ay);
        retval.ior = ior;
        return retval;
    }

    template<bool aniso>    // this or specialize?
    Scalar __eval_DG_wo_clamps(params_t params)
    {
        if (aniso)
        {
            const Scalar ax2 = A.x*A.x;
            const Scalar ay2 = A.y*A.y;
            Scalar NG = ndf::ggx_aniso<Scalar>(params.TdotH2, params.BdotH2, params.NdotH2, A.x, A.y, ax2, ay2);
            if (any(A > numeric_limits<Scalar>::min))
                NG *= smith::ggx_correlated_wo_numerator<Scalar>(params.NdotV, params.TdotV2, params.BdotV2, params.NdotV2, params.NdotL, params.TdotL2, params.BdotL2, params.NdotL2, ax2, ay2);
            return NG;
        }
        else
        {
            Scalar a2 = A.x*A.x;
            Scalar NG = ndf::ggx_trowbridge_reitz<Scalar>(a2, params.NdotH2);
            if (a2 > numeric_limits<Scalar>::min)
                NG *= smith::ggx_correlated_wo_numerator<Scalar>(max(params.NdotV,0.0), params.NdotV2, max(params.NdotL,0.0), params.NdotL2, a2);
            return NG;
        }
    }

    template<bool aniso>
    vector_t3 __eval_wo_clamps(params_t params)
    {
        Scalar scalar_part = __eval_DG_wo_clamps<aniso>(params);
        return fresnelConductor<Scalar>(ior[0], ior[1], params.VdotH) * microfacet_to_light_measure_transform<Scalar,true>(scalar_part, params.NdotL);        
    }

    template<class LightSample, class Iso, class Cache NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso> && IsotropicMicrofacetCache<Cache>)    // maybe put template in struct vs function?
    vector_t3 eval(LightSample _sample, Iso interaction, Cache cache)
    {
        if (_sample.NdotL > numeric_limits<Scalar>::min && interaction.NdotV > numeric_limits<Scalar>::min)
        {
            params_t params = params_t::template create<LightSample, Iso, Cache>(_sample, interaction, cache);
            return __eval_wo_clamps<false>(params);
        }
        else
            return (vector_t3)0.0;
    }

    template<class LightSample, class Aniso, class Cache NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Anisotropic<Iso> && AnisotropicMicrofacetCache<Cache>)    // maybe put template in struct vs function?
    vector_t3 eval(LightSample _sample, Aniso interaction, Cache cache)
    {
        if (_sample.NdotL > numeric_limits<Scalar>::min && interaction.NdotV > numeric_limits<Scalar>::min)
        {
            params_t params = params_t::template create<LightSample, Aniso, Cache>(_sample, interaction, cache);
            return __eval_wo_clamps<true>(params);
        }
        else
            return (vector_t3)0.0;
    }

    vector_t3 __generate(vector_t3 localV, vector_t2 u)
    {
        vector_t3 V = normalize(vector_t3(A.x*localV.x, A.y*localV.y, localV.z));//stretch view vector so that we're sampling as if roughness=1.0

        Scalar lensq = V.x*V.x + V.y*V.y;
        vector_t3 T1 = lensq > 0.0 ? vector_t3(-V.y, V.x, 0.0) * rsqrt(lensq) : vector_t3(1.0,0.0,0.0);
        vector_t3 T2 = cross(V,T1);

        Scalar r = sqrt(u.x);
        Scalar phi = 2.0 * nbl_glsl_PI * u.y;
        Scalar t1 = r * cos(phi);
        Scalar t2 = r * sin(phi);
        Scalar s = 0.5 * (1.0 + V.z);
        t2 = (1.0 - s)*sqrt(1.0 - t1*t1) + s*t2;
        
        //reprojection onto hemisphere
        //TODO try it wothout the max(), not sure if -t1*t1-t2*t2>-1.0
        vector_t3 H = t1*T1 + t2*T2 + sqrt(max(0.0, 1.0-t1*t1-t2*t2))*V;
        //unstretch
        return normalize(vector_t3(A.x*H.x, A.y*H.y, H.z));
    }

    template<class LightSample, class Aniso, class Cache NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Anisotropic<Aniso> && AnisotropicMicrofacetCache<Cache>)
    LightSample generate(Aniso interaction, vector<Scalar, 2> u, out Cache cache)
    {
        const vector_t3 localV = interaction.getTangentSpaceV();
        const vector_t3 H = __generate(localV, u);
        
        cache = Aniso<Scalar>::create(localV, H);
        vector_t3 localL = math::reflect<Scalar>(localV, H, cache.VdotH);

        return LightSample::createTangentSpace(localV, localL, interaction.getTangentFrame());
    }

    template<class LightSample, class Iso, class Cache NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso> && IsotropicMicrofacetCache<Cache>)
    Scalar pdf(LightSample _sample, Iso interaction, Cache cache)
    {
        const Scalar a2 = A.x*A.x;
        Scalar ndf = ndf::ggx_trowbridge_reitz<Scalar>(a2, cache.NdotH2);

        const Scalar devsh_v = smith::ggx_devsh_part<Scalar>(interaction.NdotV2, a2, 1.0-a2);
        const Scalar G1_over_2NdotV = smith::ggx_G1_wo_numerator<Scalar>(interaction.NdotV, devsh_v);
        return smith::VNDF_pdf_wo_clamps<Scalar>(ndf, G1_over_2NdotV);
    }

    template<class LightSample, class Aniso, class Cache NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Anisotropic<Aniso> && AnisotropicMicrofacetCache<Cache>)
    Scalar pdf(LightSample _sample, Aniso interaction, Cache cache)
    {
        const Scalar ax2 = A.x*A.x;
        const Scalar ay2 = A.y*A.y;
        Scalar ndf = ndf::ggx_aniso<Scalar>(cache.TdotH * cache.TdotH, cache.BdotH * cache.BdotH, cache.NdotH2, A.x, A.y, ax2, ay2);

        const Scalar devsh_v = smith::ggx_devsh_part<Scalar>(interaction.TdotV * interaction.TdotV, interaction.BdotV * interaction.BdotV, interaction.NdotV2, ax2, ay2);
        const Scalar G1_over_2NdotV = smith::ggx_G1_wo_numerator<Scalar>(interaction.NdotV, devsh_v);
        return smith::VNDF_pdf_wo_clamps<Scalar>(ndf, G1_over_2NdotV);
    }

    template<typename SpectralBins, class LightSample, class Iso, class Cache NBL_FUNC_REQUIRES(spectral_of<SpectralBins,Scalar> && Sample<LightSample> && surface_interactions::Isotropic<Iso> && IsotropicMicrofacetCache<Cache>)
    quotient_and_pdf<SpectralBins, Scalar> quotient_and_pdf_wo_clamps(LightSample _sample, Iso interaction, Cache cache, vector_t3 reflectance)
    {
        const Scalar a2 = A.x*A.x;
        const Scalar one_minus_a2 = 1.0 - a2;

        const Scalar ndf = ndf::ggx_trowbridge_reitz<Scalar>(a2, cache.NdotH2);
        const Scalar devsh_v = smith::ggx_devsh_part<Scalar>(interaction.NdotV2, a2, one_minus_a2);
        Scalar pdf = pdf<LightSample, Iso, Cache>(_sample, interaction, cache);

        Scalar G2_over_G1 = smith::ggx_G2_over_G1_devsh<Scalar>(_sample.NdotL, _sample.NdotL2, interaction.NdotV, devsh_v, a2, one_minus_a2);
        vector_t3 quo = reflectance * G2_over_G1;
        
        return quotient_and_pdf<SpectralBins, Scalar>::create(SpectralBins(quo), pdf);
    }

    template<typename SpectralBins, class LightSample, class Iso, class Cache NBL_FUNC_REQUIRES(spectral_of<SpectralBins,Scalar> && Sample<LightSample> && surface_interactions::Isotropic<Iso> && IsotropicMicrofacetCache<Cache>)
    quotient_and_pdf<SpectralBins, Scalar> quotient_and_pdf(LightSample _sample, Iso interaction, Cache cache)
    {
        const Scalar a2 = A.x*A.x;
        const Scalar one_minus_a2 = 1.0 - a2;

        const Scalar ndf = ndf::ggx_trowbridge_reitz<Scalar>(a2, cache.NdotH2);
        const Scalar devsh_v = smith::ggx_devsh_part<Scalar>(interaction.NdotV2, a2, one_minus_a2);
        Scalar pdf = pdf<LightSample, Iso, Cache>(_sample, interaction, cache);

        vector_t3 quo = (vector_t3)0.0;
        if (_sample.NdotL > numeric_limits<Scalar>::min && interaction.NdotV > numeric_limits<Scalar>::min)
        {
            const vector_t3 reflectance = fresnelConductor<Scalar>(ior[0], ior[1], cache.VdotH);
            Scalar G2_over_G1 = smith::ggx_G2_over_G1_devsh<Scalar>(_sample.NdotL, _sample.NdotL2, interaction.NdotV, devsh_v, a2, one_minus_a2);
            quo = reflectance * G2_over_G1;
        }
        
        return quotient_and_pdf<SpectralBins, Scalar>::create(SpectralBins(quo), pdf);
    }

    template<typename SpectralBins, class LightSample, class Aniso, class Cache NBL_FUNC_REQUIRES(spectral_of<SpectralBins,Scalar> && Sample<LightSample> && surface_interactions::Anisotropic<Aniso> && AnisotropicMicrofacetCache<Cache>)
    quotient_and_pdf<SpectralBins, Scalar> quotient_and_pdf_wo_clamps(LightSample _sample, Aniso interaction, Cache cache, vector_t3 reflectance)
    {
        params_t params = params_t::template create<LightSample, Aniso, Cache>(_sample, interaction, cache);
        const Scalar ax2 = A.x*A.x;
        const Scalar ay2 = A.y*A.y;

        const Scalar ndf = ndf::ggx_aniso<Scalar>(params.TdotH2, params.BdotH2, params.NdotH2, A.x, A.y, ax2, ay2);
        const Scalar devsh_v = smith::ggx_devsh_part<Scalar>(interaction.NdotV2, a2, one_minus_a2);
        Scalar pdf = pdf<LightSample, Aniso, Cache>(_sample, interaction, cache);

        Scalar G2_over_G1 = smith::ggx_G2_over_G1_devsh<Scalar>(params.NdotL, params.TdotL2, params.BdotL2, params.NdotL2, params.NdotV, devsh_v, ax2, ay2);
        vector_t3 quo = reflectance * G2_over_G1;
        
        return quotient_and_pdf<SpectralBins, Scalar>::create(SpectralBins(quo), pdf);
    }

    template<typename SpectralBins, class LightSample, class Aniso, class Cache NBL_FUNC_REQUIRES(spectral_of<SpectralBins,Scalar> && Sample<LightSample> && surface_interactions::Anisotropic<Aniso> && AnisotropicMicrofacetCache<Cache>)
    quotient_and_pdf<SpectralBins, Scalar> quotient_and_pdf(LightSample _sample, Aniso interaction, Cache cache)
    {
        params_t params = params_t::template create<LightSample, Aniso, Cache>(_sample, interaction, cache);
        const Scalar ax2 = A.x*A.x;
        const Scalar ay2 = A.y*A.y;

        const Scalar ndf = ndf::ggx_aniso<Scalar>(params.TdotH2, params.BdotH2, params.NdotH2, A.x, A.y, ax2, ay2);
        const Scalar devsh_v = smith::ggx_devsh_part<Scalar>(interaction.NdotV2, a2, one_minus_a2);
        Scalar pdf = pdf<LightSample, Aniso, Cache>(_sample, interaction, cache);

        vector_t3 quo = (vector_t3)0.0;
        if (_sample.NdotL > numeric_limits<Scalar>::min && interaction.NdotV > numeric_limits<Scalar>::min)
        {
            const vector_t3 reflectance = fresnel_conductor<Scalar>(ior[0], ior[1], cache.VdotH);
            Scalar G2_over_G1 = smith::ggx_G2_over_G1_devsh<Scalar>(params.NdotL, params.TdotL2, params.BdotL2, params.NdotL2, params.NdotV, devsh_v, ax2, ay2);
            quo = reflectance * G2_over_G1;
        }
        
        return quotient_and_pdf<SpectralBins, Scalar>::create(SpectralBins(quo), pdf);
    }

    vector_t2 A;
    matrix<Scalar,3,2> ior;
};

}
}
}
}

#endif
