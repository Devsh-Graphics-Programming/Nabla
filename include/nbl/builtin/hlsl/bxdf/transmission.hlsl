// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/sampling/cos_weighted.hlsl"
#include "nbl/builtin/hlsl/bxdf/reflection.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace transmission
{

template<class LightSample, class Iso, class Aniso, class RayDirInfo, typename Scalar 
        NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso> && surface_interactions::Anisotropic<Aniso> && ray_dir_info::Basic<RayDirInfo> && is_scalar_v<Scalar>)
LightSample cos_generate(NBL_CONST_REF_ARG(Iso) interaction)
{
    return LightSample(interaction.V.transmit(),-1.f,interaction.N);
}
template<class LightSample, class Iso, class Aniso, class RayDirInfo, typename Scalar 
    NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso> && surface_interactions::Anisotropic<Aniso> && ray_dir_info::Basic<RayDirInfo> && is_scalar_v<Scalar>)
LightSample cos_generate(NBL_CONST_REF_ARG(Aniso) interaction)
{
    return LightSample(interaction.V.transmit(),-1.f,interaction.T,interaction.B,interaction.N);
}

// Why don't we check that the incoming and outgoing directions equal each other
// (or similar for other delta distributions such as reflect, or smooth [thin] dielectrics):
// - The `quotient_and_pdf` functions are meant to be used with MIS and RIS
// - Our own generator can never pick an improbable path, so no checking necessary
// - For other generators the estimator will be `f_BSDF*f_Light*f_Visibility*clampedCos(theta)/(1+(p_BSDF^alpha+p_otherNonChosenGenerator^alpha+...)/p_ChosenGenerator^alpha)`
//	 therefore when `p_BSDF` equals `nbl_glsl_FLT_INF` it will drive the overall MIS estimator for the other generators to 0 so no checking necessary
template<typename SpectralBins, typename Pdf NBL_FUNC_REQUIRES(spectral_of<SpectralBins,Pdf> && is_floating_point_v<Pdf>)
quotient_and_pdf<SpectralBins, Pdf> cos_quotient_and_pdf()
{
    return quotient_and_pdf<SpectralBins, Pdf>::create(SpectralBins(1.f),nbl::hlsl::numeric_limits<Scalar>::inf());
}

// basic bxdf
template<typename Scalar NBL_PRIMARY_REQUIRES(is_scalar_v<Scalar>)
struct SLambertianBxDF
{
    static SLambertianBxDF<Scalar> create()
    {
        SLambertianBxDF<Scalar> retval;
        // nothing here, just keeping convention with others
        return retval;
    }

    Scalar __eval_pi_factored_out(Scalar absNdotL)
    {
        return absNdotL;
    }

    template<class LightSample, class Iso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso>)    // maybe put template struct vs function?
    Scalar __eval_wo_clamps(LightSample _sample, Iso interaction)
    {
        // probably doesn't need to use the param struct
        return __eval_pi_factored_out(_sample.NdotL) * numbers::inv_pi<Scalar> * 0.5;
    }

    template<class LightSample, class Iso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso>)    // maybe put template struct vs function?
    Scalar eval(LightSample _sample, Iso interaction)
    {
        // probably doesn't need to use the param struct
        return __eval_pi_factored_out(abs(_sample.NdotL)) * numbers::inv_pi<Scalar> * 0.5;
    }

    template<class LightSample, class Aniso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Anisotropic<Aniso>)
    LightSample generate_wo_clamps(Aniso interaction, vector<Scalar, 2> u)
    {
        vector<Scalar, 3> L = projected_sphere_generate<Scalar>(u);
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
        return projected_sphere_pdf<Scalar>(_sample.NdotL, 0.0);
    }

    template<class LightSample, class Iso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso>)
    Scalar pdf(LightSample _sample, Iso interaction)
    {
        return projected_sphere_pdf<Scalar>(abs(_sample.NdotL));
    }

    template<typename SpectralBins, class LightSample, class Iso NBL_FUNC_REQUIRES(spectral_of<SpectralBins,Pdf> && Sample<LightSample> && surface_interactions::Anisotropic<Aniso>)
    quotient_and_pdf<SpectralBins, Scalar> quotient_and_pdf_wo_clamps(LightSample _sample, Iso interaction)
    {
        Scalar pdf;
        Scalar q = projected_sphere_quotient_and_pdf<Scalar>(pdf, _sample.NdotL);
        return quotient_and_pdf<SpectralBins, Scalar>::create(SpectralBins(q), pdf);
    }

    template<typename SpectralBins, class LightSample, class Iso NBL_FUNC_REQUIRES(spectral_of<SpectralBins,Pdf> && Sample<LightSample> && surface_interactions::Anisotropic<Aniso>)
    quotient_and_pdf<SpectralBins, Scalar> quotient_and_pdf(LightSample _sample, Iso interaction)
    {
        Scalar pdf;
        Scalar q = projected_sphere_quotient_and_pdf<Scalar>(pdf, abs(_sample.NdotL));
        return quotient_and_pdf<SpectralBins, Scalar>::create(SpectralBins(q), pdf);
    }
};

// no oren nayar

// microfacet bxdfs

// the dielectric ones don't fit the concept at all :(
template<typename Scalar, bool thin = false NBL_PRIMARY_REQUIRES(is_scalar_v<Scalar>)
struct SSmoothDielectricBxDF
{
    using vector_t3 = vector<Scalar,3>;

    static SSmoothDielectricBxDF<Scalar,true> create(vector_t3 eta)
    {
        SSmoothDielectricBxDF<Scalar,true> retval;
        retval.eta = eta;
        return retval;
    }

    // where eval?

    template<class LightSample NBL_FUNC_REQUIRES(Sample<LightSample>)
    LightSample __generate_wo_clamps(vector_t3 V, vector_t3 T, vector_t3 B, vector_t3 N, bool backside, Scalar NdotV, Scalar absNdotV, Scalar NdotV2, inout vector_t3 u, Scalar rcpOrientedEta, Scalar orientedEta2, Scalar rcpOrientedEta2, out bool transmitted)
    {
        const vector_t3 reflectance = fresnelDielectric_common<vector_t3>(orientedEta2, absNdotV);

        Scalar rcpChoiceProb;
        transmitted = math::partitionRandVariable(reflectance, u.z, rcpChoiceProb);
        
        const vector_t3 L = math::reflectRefract(transmitted, V, N, backside, NdotV, NdotV2, rcpOrientedEta, rcpOrientedEta2);
        return LightSample::create(L, dot(V, L), T, B, N);
    }

    template<class LightSample, class Aniso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Anisotropic<Aniso>)
    LightSample generate_wo_clamps(Aniso interaction, inout vector<Scalar, 3> u)    // TODO: check vec3?
    {
        Scalar orientedEta, rcpOrientedEta;
        const bool backside = math::getOrientedEtas<Scalar>(orientedEta, rcpOrientedEta, interaction.NdotV, eta);
        bool dummy;
        return __generate_wo_clamps<LightSample>(interaction.V.direction, interaction.T, interaction.B, interaction.N, backside, interaction.NdotV, 
            interaction.NdotV, interaction.NdotV*interaction.NdotV, u, rcpOrientedEta, orientedEta*orientedEta, rcpOrientedEta*rcpOrientedEta, dummy);
    }

    template<class LightSample, class Aniso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Anisotropic<Aniso>)
    LightSample generate(Aniso interaction, inout vector<Scalar, 3> u)
    {
        Scalar orientedEta, rcpOrientedEta;
        const bool backside = math::getOrientedEtas<Scalar>(orientedEta, rcpOrientedEta, interaction.NdotV, eta);
        bool dummy;
        return __generate_wo_clamps<LightSample>(interaction.V.direction, interaction.T, interaction.B, interaction.N, backside, interaction.NdotV, 
            abs(interaction.NdotV), interaction.NdotV*interaction.NdotV, u, rcpOrientedEta, orientedEta*orientedEta, rcpOrientedEta*rcpOrientedEta, dummy);
    }

    // where pdf?

    template<typename SpectralBins, class LightSample, class Iso NBL_FUNC_REQUIRES(spectral_of<SpectralBins,Pdf> && Sample<LightSample> && surface_interactions::Isotropic<Iso>)
    quotient_and_pdf<SpectralBins, Scalar> quotient_and_pdf(LightSample _sample, Iso interaction)
    {
        const bool transmitted = isTransmissionPath(interaction.NdotV, _sample.NdotL);
        
        Scalar dummy, rcpOrientedEta;
        const bool backside = math::getOrientedEtas<Scalar>(dummy, rcpOrientedEta, interaction.NdotV, eta);

        const Scalar pdf = 1.0 / 0.0;
        Scalar quo = transmitted ? rcpOrientedEta2 : 1.0;
        return quotient_and_pdf<SpectralBins, Scalar>::create(SpectralBins(quo), pdf);
    }

    vector_t3 eta;
};

template<typename Scalar NBL_PRIMARY_REQUIRES(is_scalar_v<Scalar>)
struct SSmoothDielectricBxDF<Scalar,true>
{
    using vector_t3 = vector<Scalar,3>;

    static SSmoothDielectricBxDF<Scalar,true> create(vector_t3 eta2, vector_t3 luminosityContributionHint)
    {
        SSmoothDielectricBxDF<Scalar,true> retval;
        retval.eta2 = eta2;
        retval.luminosityContributionHint = luminosityContributionHint;
        return retval;
    }

    // where eval?

    // usually `luminosityContributionHint` would be the Rec.709 luma coefficients (the Y row of the RGB to CIE XYZ matrix)
    // its basically a set of weights that determine 
    // assert(1.0==luminosityContributionHint.r+luminosityContributionHint.g+luminosityContributionHint.b);
    // `remainderMetadata` is a variable which the generator function returns byproducts of sample generation that would otherwise have to be redundantly calculated `remainder_and_pdf`
    template<class LightSample NBL_FUNC_REQUIRES(Sample<LightSample>)
    LightSample __generate_wo_clamps(vector_t3 V, vector_t3 T, vector_t3 B, vector_t3 N, Scalar NdotV, Scalar absNdotV, inout vector_t3 u, vector_t3 eta2, vector_t3 luminosityContributionHint, out vector_t3 remainderMetadata)
    {
        // we will only ever intersect from the outside
        const vector_t3 reflectance = thindielectricInfiniteScatter<vector_t3>(fresnelDielectric_common<vector_t3>(eta2,absNdotV));

        // we are only allowed one choice for the entire ray, so make the probability a weighted sum
        const Scalar reflectionProb = dot(reflectance, luminosityContributionHint);

        Scalar rcpChoiceProb;
        const bool transmitted = math::partitionRandVariable(reflectionProb, u.z, rcpChoiceProb);
        remainderMetadata = (transmitted ? ((vector_t3)(1.0) - reflectance) : reflectance) * rcpChoiceProb;
        
        const vector_t3 L = (transmitted ? (vector_t3)(0.0) : N * 2.0 * NdotV) - V;
        return LightSample::create(L, dot(V, L), T, B, N);
    }

    template<class LightSample, class Aniso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Anisotropic<Aniso>)
    LightSample generate_wo_clamps(Aniso interaction, inout vector<Scalar, 3> u)    // TODO: check vec3?
    {
        return __generate_wo_clamps<LightSample>(interaction.V.direction, interaction.T, interaction.B, interaction.N, interaction.NdotV, interaction.NdotV, u, eta2, luminosityContributionHint);
    }

    template<class LightSample, class Aniso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Anisotropic<Aniso>)
    LightSample generate(Aniso interaction, inout vector<Scalar, 3> u)
    {
        return __generate_wo_clamps<LightSample>(interaction.V.direction, interaction.T, interaction.B, interaction.N, interaction.NdotV, abs(interaction.NdotV), u, eta2, luminosityContributionHint);
    }

    // where pdf?

    template<typename SpectralBins, class LightSample, class Iso NBL_FUNC_REQUIRES(spectral_of<SpectralBins,Pdf> && Sample<LightSample> && surface_interactions::Isotropic<Iso>)
    quotient_and_pdf<SpectralBins, Scalar> quotient_and_pdf_wo_clamps(LightSample _sample, Iso interaction)
    {
        const bool transmitted = isTransmissionPath(interaction.NdotV, _sample.NdotL);
        const vec3 reflectance = thindielectricInfiniteScatter<vector_t3>(fresnelDielectric_common<vector_t3>(eta2, interaction.NdotV));
        const vec3 sampleValue = transmitted ? ((vector_t3)(1.0) - reflectance) : reflectance;

        const Scalar sampleProb = dot(sampleValue,luminosityContributionHint);

        const Scalar pdf = 1.0 / 0.0;
        return quotient_and_pdf<SpectralBins, Scalar>::create(SpectralBins(sampleValue / sampleProb), pdf);
    }

    template<typename SpectralBins, class LightSample, class Iso NBL_FUNC_REQUIRES(spectral_of<SpectralBins,Pdf> && Sample<LightSample> && surface_interactions::Isotropic<Iso>)
    quotient_and_pdf<SpectralBins, Scalar> quotient_and_pdf(LightSample _sample, Iso interaction)
    {
        const bool transmitted = isTransmissionPath(interaction.NdotV, _sample.NdotL);
        const vec3 reflectance = thindielectricInfiniteScatter<vector_t3>(fresnelDielectric_common<vector_t3>(eta2, abs(interaction.NdotV)));
        const vec3 sampleValue = transmitted ? ((vector_t3)(1.0) - reflectance) : reflectance;

        const Scalar sampleProb = dot(sampleValue,luminosityContributionHint);

        const Scalar pdf = 1.0 / 0.0;
        return quotient_and_pdf<SpectralBins, Scalar>::create(SpectralBins(sampleValue / sampleProb), pdf);
    }

    vector_t3 eta2;
    vector_t3 luminosityContributionHint;
};

template<typename Scalar NBL_PRIMARY_REQUIRES(is_scalar_v<Scalar>)
struct SBeckmannDielectricBxDF
{
    using vector_t3 = vector<Scalar,3>;
    using vector_t2 = vector<Scalar,2>;
    using matrix_t3x3 = matrix<Scalar,3,3>;
    using params_t = SBxDFParams<Scalar>;

    static SBeckmannDielectricBxDF<Scalar> create(vector_t3 eta, Scalar A)
    {
        SBeckmannDielectricBxDF<Scalar> retval;
        retval.eta = eta;
        retval.A = vector_t2(A, A);
        return retval;
    }

    static SBeckmannDielectricBxDF<Scalar> create(vector_t3 eta, Scalar ax, Scalar ay)
    {
        SBeckmannDielectricBxDF<Scalar> retval;
        retval.eta = eta;
        retval.A = vector_t2(ax, ay);
        return retval;
    }

    template<class LightSample, class Iso, class Cache NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso> && IsotropicMicrofacetCache<Cache>)
    vector_t3 eval(LightSample _sample, Iso interaction, Cache cache)
    {
        float orientedEta, dummy;
        const bool backside = math::getOrientedEtas<Scalar>(orientedEta, dummy, cache.VdotH, eta);
        const float orientedEta2 = orientedEta * orientedEta;
        
        const float VdotHLdotH = cache.VdotH * cache.LdotH;
        const bool transmitted = VdotHLdotH < 0.0;

        matrix<Scalar,3,2> dummyior;
        params_t params = params_t::template create<LightSample, Iso, Cache>(_sample, interaction, cache);
        reflection::SBeckmannBxDF<Scalar> beckmann = reflection::SBeckmannBxDF<Scalar>::create(A.x, dummyior);
        const float scalar_part = beckmann::template __eval_DG_wo_clamps<false>(params);

        return fresnelDielectric_common<Scalar>(orientedEta2, abs(cache.VdotH)) * microfacet_to_light_measure_transform<Scalar,false>(scalar_part,abs(interaction.NdotV),transmitted,cache.VdotH,cache.LdotH,VdotHLdotH,orientedEta);
    }

    template<class LightSample, class Aniso, class Cache NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Anisotropic<Iso> && AnisotropicMicrofacetCache<Cache>)
    vector_t3 eval(LightSample _sample, Aniso interaction, Cache cache)
    {
        float orientedEta, dummy;
        const bool backside = math::getOrientedEtas<Scalar>(orientedEta, dummy, cache.VdotH, eta);
        const float orientedEta2 = orientedEta * orientedEta;
        
        const float VdotHLdotH = cache.VdotH * cache.LdotH;
        const bool transmitted = VdotHLdotH < 0.0;

        matrix<Scalar,3,2> dummyior;
        params_t params = params_t::template create<LightSample, Aniso, Cache>(_sample, interaction, cache);
        reflection::SBeckmannBxDF<Scalar> beckmann = reflection::SBeckmannBxDF<Scalar>::create(A.x, A.y, dummyior);
        const float scalar_part = beckmann::template __eval_DG_wo_clamps<true>(params);

        return fresnelDielectric_common<Scalar>(orientedEta2, abs(cache.VdotH)) * microfacet_to_light_measure_transform<Scalar,true>(scalar_part,abs(interaction.NdotV),transmitted,cache.VdotH,cache.LdotH,VdotHLdotH,orientedEta);
    }

    template<class LightSample, class Cache NBL_FUNC_REQUIRES(Sample<LightSample> && AnisotropicMicrofacetCache<Cache>)
    LightSample __generate_wo_clamps(vector_t3 localV, bool backside, vector_t3 H, matrix_t3x3 m, inout vector_t3 u, Scalar rcpOrientedEta, Scalar orientedEta2, Scalar rcpOrientedEta2, out Cache cache)
    {
        const Scalar VdotH = dot(localV,H);
        const Scalar reflectance = fresnelDielectric_common<vector_t3>(orientedEta2,abs(VdotH));
        
        Scalar rcpChoiceProb;
        bool transmitted = math::partitionRandVariable(reflectance, u.z, rcpChoiceProb);
        
        vec3 localL;
        cache = Aniso<Scalar>::create(localV, H);

        const Scalar VdotH = cache.VdotH;
        cache.LdotH = transmitted ? nbl_glsl_refract_compute_NdotT(VdotH<0.0,VdotH*VdotH,rcpOrientedEta2):VdotH;
        tangentSpaceL = math::reflectRefract_impl(transmitted, tangentSpaceV, tangentSpaceH, VdotH, cache.LdotH, rcpOrientedEta);

        return LightSample::createTangentSpace(localV, localL, m);
    }

    template<class LightSample, class Aniso, class Cache NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Anisotropic<Aniso> && AnisotropicMicrofacetCache<Cache>)
    LightSample generate(Aniso interaction, inout vector<Scalar, 3> u, out Cache cache)
    {
        const vector_t3 localV = interaction.getTangentSpaceV();

        Scalar orientedEta, rcpOrientedEta;
        const bool backside = math::getOrientedEtas<Scalar>(orientedEta, rcpOrientedEta, interaction.NdotV, eta);

        const vector_t3 upperHemisphereV = backside ? -localV : localV;

        matrix<Scalar,3,2> dummyior;
        reflection::SBeckmannBxDF<Scalar> beckmann = reflection::SBeckmannBxDF<Scalar>::create(A.x, A.y, dummyior);
        const vector_t3 H = beckmann.__generate(upperHemisphereV, u.xy);

        return __generate_wo_clamps<LightSample, Cache>(localV, backside, H, interaction.getTangentFrame(), rcpOrientedEta, orientedEta*orientedEta, rcpOrientedEta*rcpOrientedEta, cache);
    }

    template<class LightSample, class Aniso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Anisotropic<Aniso>)
    LightSample generate(Aniso interaction, inout vector<Scalar, 3> u)
    {
        SAnisotropicMicrofacetCache<Scalar> dummycache;
        return generate<LightSample, Aniso, SAnisotropicMicrofacetCache<Scalar> >(interaction, u, dummycache);
    }

    Scalar pdf_wo_clamps(bool transmitted, Scalar reflectance, Scalar ndf, Scalar absNdotV, Scalar NdotV2, Scalar VdotH, Scalar LdotH, Scalar VdotHLdotH, Scalar orientedEta, out Scalar onePlusLambda_V)
    {
        const Scalar lambda = smith::beckmann_Lambda<Scalar>(NdotV2, A.x*A.x);
        return smith::VNDF_pdf_wo_clamps<Scalar>(ndf,lambda,absNdotV,transmitted,VdotH,LdotH,VdotHLdotH,orientedEta,reflectance,onePlusLambda_V);
    }

    Scalar pdf_wo_clamps(bool transmitted, Scalar reflectance, Scalar ndf, Scalar absNdotV, Scalar TdotV2, Scalar BdotV2, Scalar NdotV2, Scalar VdotH, Scalar LdotH, Scalar VdotHLdotH, Scalar ax2, Scalar ay2, Scalar orientedEta, out Scalar onePlusLambda_V)
    {
        Scalar c2 = smith::beckmann_C2<Scalar>(TdotV2, BdotV2, NdotV2, ax2, ay2);
        Scalar lambda = smith::beckmann_Lambda<Scalar>(c2);
        return smith::VNDF_pdf_wo_clamps<Scalar>(ndf,lambda,absNdotV,transmitted,VdotH,LdotH,VdotHLdotH,orientedEta,reflectance,onePlusLambda_V);
    }

    template<class LightSample, class Iso, class Cache NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso> && IsotropicMicrofacetCache<Cache>)
    Scalar pdf(LightSample _sample, Iso interaction, Cache cache)
    {
        Scalar ndf = ndf::beckmann<Scalar>(A.x*A.x, cache.NdotH2);

        Scalar orientedEta, dummy;
        const bool backside = math::getOrientedEtas<Scalar>(orientedEta, dummy, cache.VdotH, eta);
        const Scalar orientedEta2 = orientedEta * orientedEta;

        const float VdotHLdotH = cache.VdotH * cache.LdotH;
        const bool transmitted = VdotHLdotH < 0.0;

        const Scalar reflectance = fresnelDielectric_common<Scalar>(orientedEta2, abs(cache.VdotH));

        const Scalar absNdotV = abs(interaction.NdotV);
        return pdf_wo_clamps(transmitted, reflectance, ndf, absNdotV, interaction.NdotV2, cache.VdotH, cache.LdotH, VdotHLdotH, orientedEta, dummy);
    }

    template<class LightSample, class Aniso, class Cache NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Anisotropic<Aniso> && AnisotropicMicrofacetCache<Cache>)
    Scalar pdf(LightSample _sample, Aniso interaction, Cache cache)
    {
        const Scalar ax2 = A.x*A.x;
        const Scalar ay2 = A.y*A.y;
        params_t params = params_t::template create<LightSample, Aniso, Cache>(_sample, interaction, cache);

        Scalar ndf = ndf::beckmann<Scalar>(A.x, A.y, ax2, ay2, params.TdotH2, params.BdotH2, params.NdotH2);

        Scalar orientedEta, dummy;
        const bool backside = math::getOrientedEtas<Scalar>(orientedEta, dummy, cache.VdotH, eta);
        const Scalar orientedEta2 = orientedEta * orientedEta;

        const float VdotHLdotH = cache.VdotH * cache.LdotH;
        const bool transmitted = VdotHLdotH < 0.0;

        const Scalar reflectance = fresnelDielectric_common<Scalar>(orientedEta2, abs(cache.VdotH));
        
        const Scalar absNdotV = abs(interaction.NdotV);
        return pdf_wo_clamps(transmitted, reflectance, ndf, absNdotV, params.TdotV2, params.BdotV2, params.NdotV2, params.VdotH, cache.LdotH, VdotHLdotH, ax2, ay2, orientedEta, dummy);
    }

    template<typename SpectralBins, class LightSample, class Iso, class Cache NBL_FUNC_REQUIRES(spectral_of<SpectralBins,Pdf> && Sample<LightSample> && surface_interactions::Isotropic<Iso> && IsotropicMicrofacetCache<Cache>)
    quotient_and_pdf<SpectralBins, Scalar> quotient_and_pdf(LightSample _sample, Iso interaction, Cache cache)
    {
        Scalar ndf = ndf::beckmann<Scalar>(A.x*A.x, cache.NdotH2);

        Scalar orientedEta, dummy;
        const bool backside = math::getOrientedEtas<Scalar>(orientedEta, dummy, cache.VdotH, eta);
        const Scalar orientedEta2 = orientedEta * orientedEta;

        const float VdotHLdotH = cache.VdotH * cache.LdotH;
        const bool transmitted = VdotHLdotH < 0.0;

        const Scalar reflectance = fresnelDielectric_common<Scalar>(orientedEta2, abs(cache.VdotH));
        const Scalar absNdotV = abs(interaction.NdotV);

        float onePlusLambda_V;
        pdf = pdf_wo_clamps(transmitted, reflectance, ndf, absNdotV, interaction.NdotV2, cache.VdotH, cache.LdotH, VdotHLdotH, orientedEta, dummy);
        Scalar quo = smith::beckmann_smith_G2_over_G1<Scalar>(onePlusLambda_V, NdotL2, a2);

        return quotient_and_pdf<SpectralBins, Scalar>::create(SpectralBins(quo), pdf);
    }

    template<typename SpectralBins, class LightSample, class Aniso, class Cache NBL_FUNC_REQUIRES(spectral_of<SpectralBins,Pdf> && Sample<LightSample> && surface_interactions::Anisotropic<Aniso> && AnisotropicMicrofacetCache<Cache>)
    quotient_and_pdf<SpectralBins, Scalar> quotient_and_pdf(LightSample _sample, Aniso interaction, Cache cache)
    {
        const Scalar ax2 = A.x*A.x;
        const Scalar ay2 = A.y*A.y;
        params_t params = params_t::template create<LightSample, Aniso, Cache>(_sample, interaction, cache);

        Scalar ndf = ndf::beckmann<Scalar>(A.x, A.y, ax2, ay2, params.TdotH2, params.BdotH2, params.NdotH2);

        Scalar orientedEta, dummy;
        const bool backside = math::getOrientedEtas<Scalar>(orientedEta, dummy, cache.VdotH, eta);
        const Scalar orientedEta2 = orientedEta * orientedEta;

        const float VdotHLdotH = cache.VdotH * cache.LdotH;
        const bool transmitted = VdotHLdotH < 0.0;

        const Scalar reflectance = fresnelDielectric_common<Scalar>(orientedEta2, abs(cache.VdotH));       
        const Scalar absNdotV = abs(interaction.NdotV);
        
        float onePlusLambda_V;
        pdf = pdf_wo_clamps(transmitted, reflectance, ndf, absNdotV, params.TdotV2, params.BdotV2, params.NdotV2, params.VdotH, cache.LdotH, VdotHLdotH, ax2, ay2, orientedEta, dummy);
        Scalar quo = smith::beckmann_smith_G2_over_G1<Scalar>(onePlusLambda_V, TdotL2, BdotL2, NdotL2, ax2, ay2);

        return quotient_and_pdf<SpectralBins, Scalar>::create(SpectralBins(quo), pdf);
    }

    vector_t2 A;
    vector_t3 eta;
};

}
}
}
}

#endif
