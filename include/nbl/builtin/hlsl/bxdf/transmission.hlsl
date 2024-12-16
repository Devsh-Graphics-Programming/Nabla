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
    return quotient_and_pdf<SpectralBins, Pdf>::create(SpectralBins(1.f),nbl::hlsl::numeric_limits<Pdf>::inf());
}

// basic bxdf
template<class LightSample, class Iso, class Aniso NBL_FUNC_REQUIRES(Sample<LightSample> && surface_interactions::Isotropic<Iso> && surface_interactions::Anisotropic<Aniso>)
struct SLambertianBxDF
{
    using this_t = SLambertianBxDF<LightSample, Iso, Aniso>;
    using scalar_type = typename LightSample::scalar_type;
    using isotropic_type = Iso;
    using anisotropic_type = Aniso;
    using sample_type = LightSample;
    using spectral_type = vector<scalar_type, 3>;   // TODO: most likely change this
    using quotient_pdf_type = quotient_and_pdf<spectral_type, scalar_type>;

    static this_t create()
    {
        this_t retval;
        // nothing here, just keeping convention with others
        return retval;
    }

    scalar_type __eval_pi_factored_out(scalar_type absNdotL)
    {
        return absNdotL;
    }

    scalar_type __eval_wo_clamps(sample_type _sample, isotropic_type interaction)
    {
        // probably doesn't need to use the param struct
        return __eval_pi_factored_out(_sample.NdotL) * numbers::inv_pi<scalar_type> * 0.5;
    }

    scalar_type eval(sample_type _sample, isotropic_type interaction)
    {
        // probably doesn't need to use the param struct
        return __eval_pi_factored_out(abs<scalar_type>(_sample.NdotL)) * numbers::inv_pi<scalar_type> * 0.5;
    }

    sample_type generate_wo_clamps(anisotropic_type interaction, vector<scalar_type, 2> u)
    {
        vector<scalar_type, 3> L = projected_sphere_generate<scalar_type>(u);
        return sample_type::createTangentSpace(interaction.getTangentSpaceV(), L, interaction.getTangentFrame());
    }

    sample_type generate(anisotropic_type interaction, vector<scalar_type, 2> u)
    {
        return generate_wo_clamps(interaction, u);
    }

    scalar_type pdf_wo_clamps(sample_type _sample, isotropic_type interaction)
    {
        return projected_sphere_pdf<scalar_type>(_sample.NdotL, 0.0);
    }

    scalar_type pdf(sample_type _sample, isotropic_type interaction)
    {
        return projected_sphere_pdf<scalar_type>(abs<scalar_type>(_sample.NdotL));
    }

    quotient_pdf_type quotient_and_pdf_wo_clamps(sample_type _sample, isotropic_type interaction)
    {
        scalar_type pdf;
        scalar_type q = projected_sphere_quotient_and_pdf<scalar_type>(pdf, _sample.NdotL);
        return quotient_pdf_type::create(spectral_type(q), pdf);
    }

    quotient_pdf_type quotient_and_pdf(sample_type _sample, isotropic_type interaction)
    {
        scalar_type pdf;
        scalar_type q = projected_sphere_quotient_and_pdf<scalar_type>(pdf, abs<scalar_type>(_sample.NdotL));
        return quotient_pdf_type::create(spectral_type(q), pdf);
    }
};

// no oren nayar

// microfacet bxdfs

// the dielectric ones don't fit the concept at all :(
template<class LightSample, class IsoCache, class AnisoCache, bool thin = false NBL_FUNC_REQUIRES(Sample<LightSample> && IsotropicMicrofacetCache<IsoCache> && AnisotropicMicrofacetCache<AnisoCache>)
struct SSmoothDielectricBxDF
{
    using this_t = SSmoothDielectricBxDF<LightSample, IsoCache, AnisoCache, false>;
    using scalar_type = typename LightSample::scalar_type
    using vector3_type = vector<scalar_type, 3>;
    using params_t = SBxDFParams<scalar_type>;

    using isotropic_type = typename IsoCache::isotropic_type;
    using anisotropic_type = typename AnisoCache::anisotropic_type
    using sample_type = LightSample;
    using spectral_type = vector<scalar_type, 3>;   // TODO: most likely change this
    using quotient_pdf_type = quotient_and_pdf<spectral_type, scalar_type>;
    using isocache_type = IsoCache;
    using anisocache_type = AnisoCache;

    static this_t create(vector3_type eta)
    {
        this_t retval;
        retval.eta = eta;
        return retval;
    }

    // where eval?

    sample_type __generate_wo_clamps(vector3_type V, vector3_type T, vector3_type B, vector3_type N, bool backside, scalar_type NdotV, scalar_type absNdotV, scalar_type NdotV2, inout vector3_type u, scalar_type rcpOrientedEta, scalar_type orientedEta2, scalar_type rcpOrientedEta2, out bool transmitted)
    {
        const vector3_type reflectance = fresnelDielectric_common<vector3_type>(orientedEta2, absNdotV);

        scalar_type rcpChoiceProb;
        transmitted = math::partitionRandVariable(reflectance, u.z, rcpChoiceProb);
        
        const vector3_type L = math::reflectRefract(transmitted, V, N, backside, NdotV, NdotV2, rcpOrientedEta, rcpOrientedEta2);
        return sample_type::create(L, dot<scalar_type>(V, L), T, B, N);
    }

    sample_type generate_wo_clamps(anisotropic_type interaction, inout vector<scalar_type, 3> u)    // TODO: check vec3?
    {
        scalar_type orientedEta, rcpOrientedEta;
        const bool backside = math::getOrientedEtas<scalar_type>(orientedEta, rcpOrientedEta, interaction.NdotV, eta);
        bool dummy;
        return __generate_wo_clamps(interaction.V.direction, interaction.T, interaction.B, interaction.N, backside, interaction.NdotV, 
            interaction.NdotV, interaction.NdotV*interaction.NdotV, u, rcpOrientedEta, orientedEta*orientedEta, rcpOrientedEta*rcpOrientedEta, dummy);
    }

    sample_type generate(anisotropic_type interaction, inout vector<scalar_type, 3> u)
    {
        scalar_type orientedEta, rcpOrientedEta;
        const bool backside = math::getOrientedEtas<scalar_type>(orientedEta, rcpOrientedEta, interaction.NdotV, eta);
        bool dummy;
        return __generate_wo_clamps(interaction.V.direction, interaction.T, interaction.B, interaction.N, backside, interaction.NdotV, 
            abs<scalar_type>(interaction.NdotV), interaction.NdotV*interaction.NdotV, u, rcpOrientedEta, orientedEta*orientedEta, rcpOrientedEta*rcpOrientedEta, dummy);
    }

    // where pdf?

    quotient_pdf_type quotient_and_pdf(sample_type _sample, isotropic_type interaction)
    {
        const bool transmitted = isTransmissionPath(interaction.NdotV, _sample.NdotL);
        
        scalar_type dummy, rcpOrientedEta;
        const bool backside = math::getOrientedEtas<scalar_type>(dummy, rcpOrientedEta, interaction.NdotV, eta);

        const scalar_type pdf = 1.0 / 0.0;
        scalar_type quo = transmitted ? rcpOrientedEta2 : 1.0;
        return quotient_pdf_type::create(spectral_type(quo), pdf);
    }

    vector3_type eta;
};

template<class LightSample, class IsoCache, class AnisoCache NBL_FUNC_REQUIRES(Sample<LightSample> && IsotropicMicrofacetCache<IsoCache> && AnisotropicMicrofacetCache<AnisoCache>)
struct SSmoothDielectricBxDF<LightSample, IsoCache, AnisoCache, true>
{
    using this_t = SSmoothDielectricBxDF<LightSample, IsoCache, AnisoCache, true>;
    using scalar_type = typename LightSample::scalar_type
    using vector3_type = vector<scalar_type, 3>;
    using params_t = SBxDFParams<scalar_type>;

    using isotropic_type = typename IsoCache::isotropic_type;
    using anisotropic_type = typename AnisoCache::anisotropic_type
    using sample_type = LightSample;
    using spectral_type = vector<scalar_type, 3>;   // TODO: most likely change this
    using quotient_pdf_type = quotient_and_pdf<spectral_type, scalar_type>;
    using isocache_type = IsoCache;
    using anisocache_type = AnisoCache;

    static this_t create(vector3_type eta2, vector3_type luminosityContributionHint)
    {
        this_t retval;
        retval.eta2 = eta2;
        retval.luminosityContributionHint = luminosityContributionHint;
        return retval;
    }

    // where eval?

    // usually `luminosityContributionHint` would be the Rec.709 luma coefficients (the Y row of the RGB to CIE XYZ matrix)
    // its basically a set of weights that determine 
    // assert(1.0==luminosityContributionHint.r+luminosityContributionHint.g+luminosityContributionHint.b);
    // `remainderMetadata` is a variable which the generator function returns byproducts of sample generation that would otherwise have to be redundantly calculated `remainder_and_pdf`
    sample_type __generate_wo_clamps(vector3_type V, vector3_type T, vector3_type B, vector3_type N, scalar_type NdotV, scalar_type absNdotV, inout vector3_type u, vector3_type eta2, vector3_type luminosityContributionHint, out vector3_type remainderMetadata)
    {
        // we will only ever intersect from the outside
        const vector3_type reflectance = thindielectricInfiniteScatter<vector3_type>(fresnelDielectric_common<vector3_type>(eta2,absNdotV));

        // we are only allowed one choice for the entire ray, so make the probability a weighted sum
        const scalar_type reflectionProb = dot<scalar_type>(reflectance, luminosityContributionHint);

        scalar_type rcpChoiceProb;
        const bool transmitted = math::partitionRandVariable(reflectionProb, u.z, rcpChoiceProb);
        remainderMetadata = (transmitted ? ((vector3_type)(1.0) - reflectance) : reflectance) * rcpChoiceProb;
        
        const vector3_type L = (transmitted ? (vector3_type)(0.0) : N * 2.0 * NdotV) - V;
        return sample_type::create(L, dot<scalar_type>(V, L), T, B, N);
    }

    sample_type generate_wo_clamps(anisotropic_type interaction, inout vector<scalar_type, 3> u)    // TODO: check vec3?
    {
        return __generate_wo_clamps(interaction.V.direction, interaction.T, interaction.B, interaction.N, interaction.NdotV, interaction.NdotV, u, eta2, luminosityContributionHint);
    }

    sample_type generate(anisotropic_type interaction, inout vector<scalar_type, 3> u)
    {
        return __generate_wo_clamps(interaction.V.direction, interaction.T, interaction.B, interaction.N, interaction.NdotV, abs<scalar_type>(interaction.NdotV), u, eta2, luminosityContributionHint);
    }

    // where pdf?

    quotient_pdf_type quotient_and_pdf_wo_clamps(sample_type _sample, isotropic_type interaction)
    {
        const bool transmitted = isTransmissionPath(interaction.NdotV, _sample.NdotL);
        const vec3 reflectance = thindielectricInfiniteScatter<vector3_type>(fresnelDielectric_common<vector3_type>(eta2, interaction.NdotV));
        const vec3 sampleValue = transmitted ? ((vector3_type)(1.0) - reflectance) : reflectance;

        const scalar_type sampleProb = dot<scalar_type>(sampleValue,luminosityContributionHint);

        const scalar_type pdf = 1.0 / 0.0;
        return quotient_pdf_type::create(spectral_type(sampleValue / sampleProb), pdf);
    }

    quotient_pdf_type quotient_and_pdf(sample_type _sample, isotropic_type interaction)
    {
        const bool transmitted = isTransmissionPath(interaction.NdotV, _sample.NdotL);
        const vec3 reflectance = thindielectricInfiniteScatter<vector3_type>(fresnelDielectric_common<vector3_type>(eta2, abs<scalar_type>(interaction.NdotV)));
        const vec3 sampleValue = transmitted ? ((vector3_type)(1.0) - reflectance) : reflectance;

        const scalar_type sampleProb = dot<scalar_type>(sampleValue,luminosityContributionHint);

        const scalar_type pdf = 1.0 / 0.0;
        return quotient_pdf_type::create(spectral_type(sampleValue / sampleProb), pdf);
    }

    vector3_type eta2;
    vector3_type luminosityContributionHint;
};

template<class LightSample, class IsoCache, class AnisoCache NBL_FUNC_REQUIRES(Sample<LightSample> && IsotropicMicrofacetCache<IsoCache> && AnisotropicMicrofacetCache<AnisoCache>)
struct SBeckmannDielectricBxDF
{
    using this_t = SBeckmannDielectricBxDF<LightSample, IsoCache, AnisoCache>;
    using scalar_type = typename LightSample::scalar_type
    using vector2_type = vector<scalar_type, 2>;
    using vector3_type = vector<scalar_type, 3>;
    using matrix3x3_type = matrix<scalar_type,3,3>;
    using params_t = SBxDFParams<scalar_type>;

    using isotropic_type = typename IsoCache::isotropic_type;
    using anisotropic_type = typename AnisoCache::anisotropic_type
    using sample_type = LightSample;
    using spectral_type = vector<scalar_type, 3>;   // TODO: most likely change this
    using quotient_pdf_type = quotient_and_pdf<spectral_type, scalar_type>;
    using isocache_type = IsoCache;
    using anisocache_type = AnisoCache;

    static this_t create(vector3_type eta, scalar_type A)
    {
        this_t retval;
        retval.eta = eta;
        retval.A = vector2_type(A, A);
        return retval;
    }

    static this_t create(vector3_type eta, scalar_type ax, scalar_type ay)
    {
        this_t retval;
        retval.eta = eta;
        retval.A = vector2_type(ax, ay);
        return retval;
    }

    vector3_type eval(sample_type _sample, isotropic_type interaction, isocache_type cache)
    {
        scalar_type orientedEta, dummy;
        const bool backside = math::getOrientedEtas<scalar_type>(orientedEta, dummy, cache.VdotH, eta);
        const scalar_type orientedEta2 = orientedEta * orientedEta;
        
        const scalar_type VdotHLdotH = cache.VdotH * cache.LdotH;
        const bool transmitted = VdotHLdotH < 0.0;

        matrix<scalar_type,3,2> dummyior;
        params_t params = params_t::template create<sample_type, isotropic_type, isocache_type>(_sample, interaction, cache);
        reflection::SBeckmannBxDF<sample_type, isocache_type, anisocache_type> beckmann = reflection::SBeckmannBxDF<sample_type, isocache_type, anisocache_type>::create(A.x, dummyior);
        const scalar_type scalar_part = beckmann.template __eval_DG_wo_clamps<false>(params);

        ndf::microfacet_to_light_measure_transform<ndf::Beckmann<scalar_type>,ndf::REFLECT_REFRACT_BIT> microfacet_transform =
            ndf::microfacet_to_light_measure_transform<ndf::Beckmann<scalar_type>,ndf::REFLECT_REFRACT_BIT>::create(scalar_part,abs<scalar_type>(interaction.NdotV),transmitted,cache.VdotH,cache.LdotH,VdotHLdotH,orientedEta);
        return fresnelDielectric_common<scalar_type>(orientedEta2, abs<scalar_type>(cache.VdotH)) * microfacet_transform();
    }

    vector3_type eval(sample_type _sample, anisotropic_type interaction, anisocache_type cache)
    {
        scalar_type orientedEta, dummy;
        const bool backside = math::getOrientedEtas<scalar_type>(orientedEta, dummy, cache.VdotH, eta);
        const scalar_type orientedEta2 = orientedEta * orientedEta;
        
        const scalar_type VdotHLdotH = cache.VdotH * cache.LdotH;
        const bool transmitted = VdotHLdotH < 0.0;

        matrix<scalar_type,3,2> dummyior;
        params_t params = params_t::template create<sample_type, anisotropic_type, anisocache_type>(_sample, interaction, cache);
        reflection::SBeckmannBxDF<sample_type, isocache_type, anisocache_type> beckmann = reflection::SBeckmannBxDF<sample_type, isocache_type, anisocache_type>::create(A.x, A.y, dummyior);
        const scalar_type scalar_part = beckmann.template __eval_DG_wo_clamps<true>(params);

        ndf::microfacet_to_light_measure_transform<ndf::Beckmann<scalar_type>,ndf::REFLECT_REFRACT_BIT> microfacet_transform =
            ndf::microfacet_to_light_measure_transform<ndf::Beckmann<scalar_type>,ndf::REFLECT_REFRACT_BIT>::create(scalar_part,abs<scalar_type>(interaction.NdotV),transmitted,cache.VdotH,cache.LdotH,VdotHLdotH,orientedEta);
        return fresnelDielectric_common<scalar_type>(orientedEta2, abs<scalar_type>(cache.VdotH)) * microfacet_transform();
    }

    sample_type __generate_wo_clamps(vector3_type localV, bool backside, vector3_type H, matrix_t3x3 m, inout vector3_type u, scalar_type rcpOrientedEta, scalar_type orientedEta2, scalar_type rcpOrientedEta2, out anisocache_type cache)
    {
        const scalar_type VdotH = dot<scalar_type>(localV,H);
        const scalar_type reflectance = fresnelDielectric_common<vector3_type>(orientedEta2,abs<scalar_type>(VdotH));
        
        scalar_type rcpChoiceProb;
        bool transmitted = math::partitionRandVariable(reflectance, u.z, rcpChoiceProb);
        
        cache = anisocache_type::create(localV, H);

        const scalar_type VdotH = cache.VdotH;
        cache.LdotH = transmitted ? reflectRefract_computeNdotT<scalar_type>(VdotH < 0.0, VdotH * VdotH, rcpOrientedEta2) : VdotH;
        vector3_type localL = math::reflectRefract_impl(transmitted, tangentSpaceV, tangentSpaceH, VdotH, cache.LdotH, rcpOrientedEta);

        return sample_type::createTangentSpace(localV, localL, m);
    }

    sample_type generate(anisotropic_type interaction, inout vector3_type u, out anisocache_type cache)
    {
        const vector3_type localV = interaction.getTangentSpaceV();

        scalar_type orientedEta, rcpOrientedEta;
        const bool backside = math::getOrientedEtas<scalar_type>(orientedEta, rcpOrientedEta, interaction.NdotV, eta);

        const vector3_type upperHemisphereV = backside ? -localV : localV;

        matrix<scalar_type,3,2> dummyior;
        reflection::SBeckmannBxDF<sample_type, isocache_type, anisocache_type> beckmann = reflection::SBeckmannBxDF<sample_type, isocache_type, anisocache_type>::create(A.x, A.y, dummyior);
        const vector3_type H = beckmann.__generate(upperHemisphereV, u.xy);

        return __generate_wo_clamps(localV, backside, H, interaction.getTangentFrame(), rcpOrientedEta, orientedEta*orientedEta, rcpOrientedEta*rcpOrientedEta, cache);
    }

    sample_type generate(anisotropic_type interaction, inout vector3_type u)
    {
        anisocache_type dummycache;
        return generate(interaction, u, dummycache);
    }

    scalar_type pdf_wo_clamps(bool transmitted, scalar_type reflectance, scalar_type ndf, scalar_type absNdotV, scalar_type NdotV2, scalar_type VdotH, scalar_type LdotH, scalar_type VdotHLdotH, scalar_type orientedEta, out scalar_type onePlusLambda_V)
    {
        smith::Beckmann<scalar_type> beckmann_smith;
        const scalar_type lambda = beckmann_smith.Lambda(NdotV2, A.x*A.x);
        return smith::VNDF_pdf_wo_clamps<smith::Beckmann<scalar_type> >(ndf,lambda,absNdotV,transmitted,VdotH,LdotH,VdotHLdotH,orientedEta,reflectance,onePlusLambda_V);
    }

    scalar_type pdf_wo_clamps(bool transmitted, scalar_type reflectance, scalar_type ndf, scalar_type absNdotV, scalar_type TdotV2, scalar_type BdotV2, scalar_type NdotV2, scalar_type VdotH, scalar_type LdotH, scalar_type VdotHLdotH, scalar_type ax2, scalar_type ay2, scalar_type orientedEta, out scalar_type onePlusLambda_V)
    {
        smith::Beckmann<scalar_type> beckmann_smith;
        scalar_type c2 = beckmann_smith.C2(TdotV2, BdotV2, NdotV2, ax2, ay2);
        scalar_type lambda = beckmann_smith.Lambda(c2);
        return smith::VNDF_pdf_wo_clamps<smith::Beckmann<scalar_type> >(ndf,lambda,absNdotV,transmitted,VdotH,LdotH,VdotHLdotH,orientedEta,reflectance,onePlusLambda_V);
    }

    scalar_type pdf(sample_type _sample, isotropic_type interaction, isocache_type cache)
    {
        ndf::SIsotropicParams<scalar_type> ndfparams = ndf::SIsotropicParams<scalar_type>::create(A.x*A.x, cache.NdotH2);
        ndf::Beckmann<scalar_type> beckmann_ndf;
        scalar_type ndf = beckmann_ndf(ndfparams);

        scalar_type orientedEta, dummy;
        const bool backside = math::getOrientedEtas<scalar_type>(orientedEta, dummy, cache.VdotH, eta);
        const scalar_type orientedEta2 = orientedEta * orientedEta;

        const scalar_type VdotHLdotH = cache.VdotH * cache.LdotH;
        const bool transmitted = VdotHLdotH < 0.0;

        const scalar_type reflectance = fresnelDielectric_common<scalar_type>(orientedEta2, abs<scalar_type>(cache.VdotH));

        const scalar_type absNdotV = abs<scalar_type>(interaction.NdotV);
        return pdf_wo_clamps(transmitted, reflectance, ndf, absNdotV, interaction.NdotV2, cache.VdotH, cache.LdotH, VdotHLdotH, orientedEta, dummy);
    }

    scalar_type pdf(sample_type _sample, anisotropic_type interaction, anisocache_type cache)
    {
        const scalar_type ax2 = A.x*A.x;
        const scalar_type ay2 = A.y*A.y;
        params_t params = params_t::template create<sample_type, anisotropic_type, anisocache_type>(_sample, interaction, cache);

        ndf::SAnisotropicParams<scalar_type> ndfparams = ndf::SAnisotropicParams<scalar_type>::create(A.x, A.y, ax2, ay2, params.TdotH2, params.BdotH2, params.NdotH2);
        ndf::Beckmann<scalar_type> beckmann_ndf;
        scalar_type ndf = backmann_ndf(ndfparams);

        scalar_type orientedEta, dummy;
        const bool backside = math::getOrientedEtas<scalar_type>(orientedEta, dummy, cache.VdotH, eta);
        const scalar_type orientedEta2 = orientedEta * orientedEta;

        const scalar_type VdotHLdotH = cache.VdotH * cache.LdotH;
        const bool transmitted = VdotHLdotH < 0.0;

        const scalar_type reflectance = fresnelDielectric_common<scalar_type>(orientedEta2, abs<scalar_type>(cache.VdotH));
        
        const scalar_type absNdotV = abs<scalar_type>(interaction.NdotV);
        return pdf_wo_clamps(transmitted, reflectance, ndf, absNdotV, params.TdotV2, params.BdotV2, params.NdotV2, params.VdotH, cache.LdotH, VdotHLdotH, ax2, ay2, orientedEta, dummy);
    }

    quotient_pdf_type quotient_and_pdf(sample_type _sample, isotropic_type interaction, isocache_type cache)
    {
        const scalar_type a2 = A.x*A.x;
        ndf::SIsotropicParams<scalar_type> ndfparams = ndf::SIsotropicParams<scalar_type>::create(a2, cache.NdotH2);
        ndf::Beckmann<scalar_type> beckmann_ndf;
        scalar_type ndf = beckmann_ndf(ndfparams);

        scalar_type orientedEta, dummy;
        const bool backside = math::getOrientedEtas<scalar_type>(orientedEta, dummy, cache.VdotH, eta);
        const scalar_type orientedEta2 = orientedEta * orientedEta;

        const scalar_type VdotHLdotH = cache.VdotH * cache.LdotH;
        const bool transmitted = VdotHLdotH < 0.0;

        const scalar_type reflectance = fresnelDielectric_common<scalar_type>(orientedEta2, abs<scalar_type>(cache.VdotH));
        const scalar_type absNdotV = abs<scalar_type>(interaction.NdotV);

        scalar_type onePlusLambda_V;
        scalar_type pdf = pdf_wo_clamps(transmitted, reflectance, ndf, absNdotV, interaction.NdotV2, cache.VdotH, cache.LdotH, VdotHLdotH, orientedEta, dummy);

        smith::SIsotropicParams<scalar_type> smithparams = smith::SIsotropicParams<scalar_type>::create(a2, 0, _sample.NdotL2, onePlusLambda_V);
        smith::Beckmann<scalar_type> beckmann_smith;
        scalar_type quo = beckmann_smith.G2_over_G1(smithparams);

        return quotient_pdf_type::create(spectral_type(quo), pdf);
    }

    quotient_pdf_type quotient_and_pdf(sample_type _sample, anisotropic_type interaction, anisocache_type cache)
    {
        const scalar_type ax2 = A.x*A.x;
        const scalar_type ay2 = A.y*A.y;
        params_t params = params_t::template create<sample_type, anisotropic_type, anisocache_type>(_sample, interaction, cache);

        ndf::SAnisotropicParams<scalar_type> ndfparams = ndf::SAnisotropicParams<scalar_type>::create(A.x, A.y, ax2, ay2, params.TdotH2, params.BdotH2, params.NdotH2);
        ndf::Beckmann<scalar_type> beckmann_ndf;
        scalar_type ndf = backmann_ndf(ndfparams);

        scalar_type orientedEta, dummy;
        const bool backside = math::getOrientedEtas<scalar_type>(orientedEta, dummy, cache.VdotH, eta);
        const scalar_type orientedEta2 = orientedEta * orientedEta;

        const scalar_type VdotHLdotH = cache.VdotH * cache.LdotH;
        const bool transmitted = VdotHLdotH < 0.0;

        const scalar_type reflectance = fresnelDielectric_common<scalar_type>(orientedEta2, abs<scalar_type>(cache.VdotH));       
        const scalar_type absNdotV = abs<scalar_type>(interaction.NdotV);
        
        scalar_type onePlusLambda_V;
        scalar_type pdf = pdf_wo_clamps(transmitted, reflectance, ndf, absNdotV, params.TdotV2, params.BdotV2, params.NdotV2, params.VdotH, cache.LdotH, VdotHLdotH, ax2, ay2, orientedEta, dummy);

        smith::SAnisotropicParams<scalar_type> smithparams = smith::SAnisotropicParams<scalar_type>::create(ax2, ay2, params.TdotV2, params.BdotV2, params.NdotV2, params.TdotL2, params.BdotL2, params.NdotL2, onePlusLambda_V);
        smith::Beckmann<scalar_type> beckmann_smith;
        scalar_type quo = beckmann_smith.G2_over_G1(smithparams);

        return quotient_pdf_type::create(spectral_type(quo), pdf);
    }

    vector2_type A;
    vector3_type eta;
};

template<class LightSample, class IsoCache, class AnisoCache NBL_FUNC_REQUIRES(Sample<LightSample> && IsotropicMicrofacetCache<IsoCache> && AnisotropicMicrofacetCache<AnisoCache>)
struct SGGXDielectricBxDF
{
    using this_t = SGGXDielectricBxDF<LightSample, IsoCache, AnisoCache>;
    using scalar_type = typename LightSample::scalar_type
    using vector2_type = vector<scalar_type, 2>;
    using vector3_type = vector<scalar_type, 3>;
    using matrix3x3_type = matrix<scalar_type,3,3>;
    using params_t = SBxDFParams<scalar_type>;

    using isotropic_type = typename IsoCache::isotropic_type;
    using anisotropic_type = typename AnisoCache::anisotropic_type
    using sample_type = LightSample;
    using spectral_type = vector<scalar_type, 3>;   // TODO: most likely change this
    using quotient_pdf_type = quotient_and_pdf<spectral_type, scalar_type>;
    using isocache_type = IsoCache;
    using anisocache_type = AnisoCache;

    static this_t create(vector3_type eta, scalar_type A)
    {
        this_t retval;
        retval.eta = eta;
        retval.A = vector2_type(A, A);
        return retval;
    }

    static this_t create(vector3_type eta, scalar_type ax, scalar_type ay)
    {
        this_t retval;
        retval.eta = eta;
        retval.A = vector2_type(ax, ay);
        return retval;
    }

    vector3_type eval(sample_type _sample, isotropic_type interaction, isocache_type cache)
    {
        scalar_type orientedEta, dummy;
        const bool backside = math::getOrientedEtas<scalar_type>(orientedEta, dummy, cache.VdotH, eta);
        const scalar_type orientedEta2 = orientedEta * orientedEta;
        
        const scalar_type VdotHLdotH = cache.VdotH * cache.LdotH;
        const bool transmitted = VdotHLdotH < 0.0;

        matrix<scalar_type,3,2> dummyior;
        params_t params = params_t::template create<sample_type, isotropic_type, isocache_type>(_sample, interaction, cache);
        reflection::GGX<sample_type, isocache_type, anisocache_type> ggx = reflection::GGX<sample_type, isocache_type, anisocache_type>::create(A.x, dummyior);
        const scalar_type NG_already_in_reflective_dL_measure = ggx.template __eval_DG_wo_clamps<false>(params);

        ndf::microfacet_to_light_measure_transform<ndf::Beckmann<scalar_type>,ndf::REFLECT_REFRACT_BIT> microfacet_transform =
            ndf::microfacet_to_light_measure_transform<ndf::Beckmann<scalar_type>,ndf::REFLECT_REFRACT_BIT>::create(NG_already_in_reflective_dL_measure,abs(_sample.NdotL),transmitted,cache.VdotH,cache.LdotH,VdotHLdotH,orientedEta);
        return fresnelDielectric_common<scalar_type>(orientedEta2, abs<scalar_type>(cache.VdotH)) * microfacet_transform();
    }

    vector3_type eval(sample_type _sample, anisotropic_type interaction, anisocache_type cache)
    {
        scalar_type orientedEta, dummy;
        const bool backside = math::getOrientedEtas<scalar_type>(orientedEta, dummy, cache.VdotH, eta);
        const scalar_type orientedEta2 = orientedEta * orientedEta;
        
        const scalar_type VdotHLdotH = cache.VdotH * cache.LdotH;
        const bool transmitted = VdotHLdotH < 0.0;

        matrix<scalar_type,3,2> dummyior;
        params_t params = params_t::template create<sample_type, isotropic_type, isocache_type>(_sample, interaction, cache);
        reflection::GGX<sample_type, isocache_type, anisocache_type> ggx = reflection::GGX<sample_type, isocache_type, anisocache_type>::create(A.x, A.y dummyior);
        const scalar_type NG_already_in_reflective_dL_measure = ggx.template __eval_DG_wo_clamps<true>(params);

        ndf::microfacet_to_light_measure_transform<ndf::Beckmann<scalar_type>,ndf::REFLECT_REFRACT_BIT> microfacet_transform =
            ndf::microfacet_to_light_measure_transform<ndf::Beckmann<scalar_type>,ndf::REFLECT_REFRACT_BIT>::create(NG_already_in_reflective_dL_measure,abs(_sample.NdotL),transmitted,cache.VdotH,cache.LdotH,VdotHLdotH,orientedEta);
        return fresnelDielectric_common<scalar_type>(orientedEta2, abs<scalar_type>(cache.VdotH)) * microfacet_transform();
    }

    sample_type __generate_wo_clamps(vector3_type localV, bool backside, vector3_type H, matrix_t3x3 m, inout vector3_type u, scalar_type rcpOrientedEta, scalar_type orientedEta2, scalar_type rcpOrientedEta2, out anisocache_type cache)
    {
        const scalar_type VdotH = dot<scalar_type>(localV,H);
        const scalar_type reflectance = fresnelDielectric_common<vector3_type>(orientedEta2,abs<scalar_type>(VdotH));
        
        scalar_type rcpChoiceProb;
        bool transmitted = math::partitionRandVariable(reflectance, u.z, rcpChoiceProb);
        
        cache = anisocache_type::create(localV, H);

        const scalar_type VdotH = cache.VdotH;
        cache.LdotH = transmitted ? reflectRefract_computeNdotT<scalar_type>(VdotH < 0.0, VdotH * VdotH, rcpOrientedEta2) : VdotH;
        vector3_type localL = math::reflectRefract_impl(transmitted, tangentSpaceV, tangentSpaceH, VdotH, cache.LdotH, rcpOrientedEta);

        return sample_type::createTangentSpace(localV, localL, m);
    }

    sample_type generate(anisotropic_type interaction, inout vector3_type u, out anisocache_type cache)
    {
        const vector3_type localV = interaction.getTangentSpaceV();

        scalar_type orientedEta, rcpOrientedEta;
        const bool backside = math::getOrientedEtas<scalar_type>(orientedEta, rcpOrientedEta, interaction.NdotV, eta);

        const vector3_type upperHemisphereV = backside ? -localV : localV;

        matrix<scalar_type,3,2> dummyior;
        reflection::SGGXBxDF<sample_type, isocache_type, anisocache_type> ggx = reflection::SBeckmannBxDF<sample_type, isocache_type, anisocache_type>::create(A.x, A.y, dummyior);
        const vector3_type H = ggx.__generate(upperHemisphereV, u.xy);

        return __generate_wo_clamps(localV, backside, H, interaction.getTangentFrame(), rcpOrientedEta, orientedEta*orientedEta, rcpOrientedEta*rcpOrientedEta, cache);
    }

    sample_type generate(anisotropic_type interaction, inout vector3_type u)
    {
        anisocache_type dummycache;
        return generate(interaction, u, dummycache);
    }

    scalar_type pdf_wo_clamps(bool transmitted, scalar_type reflectance, scalar_type ndf, scalar_type devsh_v, scalar_type absNdotV, scalar_type VdotH, scalar_type LdotH, scalar_type VdotHLdotH, scalar_type orientedEta)
    {
        smith::GGX<scalar_type> ggx_smith;
        const scalar_type lambda = ggx_smith.G1_wo_numerator(absNdotV, A.x*A.x);
        return smith::VNDF_pdf_wo_clamps<scalar_type>(ndf,lambda,absNdotV,transmitted,VdotH,LdotH,VdotHLdotH,orientedEta,reflectance);
    }

    scalar_type pdf(sample_type _sample, isotropic_type interaction, isocache_type cache)
    {
        const scalar_type a2 = A.x*A.x;
        params_t params = params_t::template create<sample_type, anisotropic_type, anisocache_type>(_sample, interaction, cache);

        ndf::SIsotropicParams<scalar_type> ndfparams = ndf::SIsotropicParams<scalar_type>::create(a2, params.NdotV, params.NdotV2, params.NdotL, params.NdotL2);
        ndf::GGX<scalar_type> ggx_ndf;
        scalar_type ndf = ggx_ndf(ndfparams);

        smith::GGX<scalar_type> ggx_smith;
        const scalar_type devsh_v = ggx_smith.devsh_part(params.NdotV2, a2, 1.0-a2);

        scalar_type orientedEta, dummy;
        const bool backside = math::getOrientedEtas<scalar_type>(orientedEta, dummy, cache.VdotH, eta);
        const scalar_type orientedEta2 = orientedEta * orientedEta;

        const scalar_type VdotHLdotH = cache.VdotH * cache.LdotH;
        const bool transmitted = VdotHLdotH < 0.0;

        const scalar_type reflectance = fresnelDielectric_common<scalar_type>(orientedEta2, abs<scalar_type>(cache.VdotH));

        const scalar_type absNdotV = abs<scalar_type>(interaction.NdotV);
        return pdf_wo_clamps(transmitted, reflectance, ndf, devsh_v, absNdotV, cache.VdotH, cache.LdotH, VdotHLdotH, orientedEta);
    }

    scalar_type pdf(sample_type _sample, anisotropic_type interaction, anisocache_type cache)
    {
        const scalar_type ax2 = A.x*A.x;
        const scalar_type ay2 = A.y*A.y;
        params_t params = params_t::template create<sample_type, anisotropic_type, anisocache_type>(_sample, interaction, cache);

        ndf::SAnisotropicParams<scalar_type> ndfparams = ndf::SAnisotropicParams<scalar_type>::create(A.x, A.y, ax2, ay2, params.TdotH2, params.BdotH2, params.NdotH2);
        ndf::GGX<scalar_type> ggx_ndf;
        scalar_type ndf = ggx_ndf(ndfparams);

        smith::GGX<scalar_type> ggx_smith;
        const scalar_type devsh_v = ggx_smith.devsh_part(params.TdotV2, params.BdotV2, params.NdotV2, ax2, ay2);

        scalar_type orientedEta, dummy;
        const bool backside = math::getOrientedEtas<scalar_type>(orientedEta, dummy, cache.VdotH, eta);
        const scalar_type orientedEta2 = orientedEta * orientedEta;

        const scalar_type VdotHLdotH = cache.VdotH * cache.LdotH;
        const bool transmitted = VdotHLdotH < 0.0;

        const scalar_type reflectance = fresnelDielectric_common<scalar_type>(orientedEta2, abs<scalar_type>(cache.VdotH));
        
        const scalar_type absNdotV = abs<scalar_type>(interaction.NdotV);
        return pdf_wo_clamps(transmitted, reflectance, ndf, absNdotV, params.TdotV2, params.BdotV2, params.NdotV2, params.VdotH, cache.LdotH, VdotHLdotH, ax2, ay2, orientedEta, dummy);
    }

    quotient_pdf_type quotient_and_pdf(sample_type _sample, isotropic_type interaction, isocache_type cache)
    {
        const scalar_type a2 = A.x*A.x;
        params_t params = params_t::template create<sample_type, anisotropic_type, anisocache_type>(_sample, interaction, cache);

        ndf::SIsotropicParams<scalar_type> ndfparams = ndf::SIsotropicParams<scalar_type>::create(a2, params.NdotV, params.NdotV2, params.NdotL, params.NdotL2);
        ndf::GGX<scalar_type> ggx_ndf;
        scalar_type ndf = ggx_ndf(ndfparams);

        smith::GGX<scalar_type> ggx_smith;
        const scalar_type devsh_v = ggx_smith.devsh_part(params.NdotV2, a2, 1.0-a2);

        scalar_type orientedEta, dummy;
        const bool backside = math::getOrientedEtas<scalar_type>(orientedEta, dummy, cache.VdotH, eta);
        const scalar_type orientedEta2 = orientedEta * orientedEta;

        const scalar_type VdotHLdotH = cache.VdotH * cache.LdotH;
        const bool transmitted = VdotHLdotH < 0.0;

        const scalar_type reflectance = fresnelDielectric_common<scalar_type>(orientedEta2, abs<scalar_type>(cache.VdotH));

        const scalar_type absNdotV = abs<scalar_type>(interaction.NdotV);
        scalar_type pdf = pdf_wo_clamps(transmitted, reflectance, ndf, devsh_v, absNdotV, cache.VdotH, cache.LdotH, VdotHLdotH, orientedEta);

        smith::SIsotropicParams<scalar_type> smithparams = smith::SIsotropicParams<scalar_type>::create(a2, params.NdotV, params.NdotV2, params.NdotL, params.NdotL2);
        scalar_type quo = ggx_smith.G2_over_G1(smithparams);

        return quotient_pdf_type::create(spectral_type(quo), pdf);
    }

    quotient_pdf_type quotient_and_pdf(sample_type _sample, anisotropic_type interaction, anisocache_type cache)
    {
        const scalar_type ax2 = A.x*A.x;
        const scalar_type ay2 = A.y*A.y;
        params_t params = params_t::template create<sample_type, anisotropic_type, anisocache_type>(_sample, interaction, cache);

        ndf::SAnisotropicParams<scalar_type> ndfparams = ndf::SAnisotropicParams<scalar_type>::create(A.x, A.y, ax2, ay2, params.TdotH2, params.BdotH2, params.NdotH2);
        ndf::GGX<scalar_type> ggx_ndf;
        scalar_type ndf = ggx_ndf(ndfparams);

        smith::GGX<scalar_type> ggx_smith;
        const scalar_type devsh_v = ggx_smith.devsh_part(params.TdotV2, params.BdotV2, params.NdotV2, ax2, ay2);

        scalar_type orientedEta, dummy;
        const bool backside = math::getOrientedEtas<scalar_type>(orientedEta, dummy, cache.VdotH, eta);
        const scalar_type orientedEta2 = orientedEta * orientedEta;

        const scalar_type VdotHLdotH = cache.VdotH * cache.LdotH;
        const bool transmitted = VdotHLdotH < 0.0;

        const scalar_type reflectance = fresnelDielectric_common<scalar_type>(orientedEta2, abs<scalar_type>(cache.VdotH));       
        const scalar_type absNdotV = abs<scalar_type>(interaction.NdotV);
        
        scalar_type pdf = pdf_wo_clamps(transmitted, reflectance, ndf, devsh_v, absNdotV, cache.VdotH, cache.LdotH, VdotHLdotH, orientedEta);

        smith::SAnisotropicParams<scalar_type> smithparams = smith::SAnisotropicParams<scalar_type>::create(ax2, ay2, absNdotV, params.TdotV2, params.BdotV2, params.NdotV2, abs(params.NdotL), params.TdotL2, params.BdotL2, params.NdotL2);
        scalar_type quo = ggx_smith.G2_over_G1(smithparams);

        return quotient_pdf_type::create(spectral_type(quo), pdf);
    }

    vector2_type A;
    vector3_type eta;
};

}
}
}
}

#endif
