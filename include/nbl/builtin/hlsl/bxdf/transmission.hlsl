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
    return quotient_and_pdf<SpectralBins, Pdf>::create(SpectralBins(1.f), numeric_limits<Pdf>::infinity);
}

// basic bxdf
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
        // nothing here, just keeping convention with others
        return retval;
    }

    static this_t create(NBL_CONST_REF_ARG(SBxDFCreationParams<scalar_type, spectral_type>) params)
    {
        return create();
    }

    void init(NBL_CONST_REF_ARG(SBxDFCreationParams<scalar_type, spectral_type>) params)
    {
        // do nothing
    }

    scalar_type __eval_pi_factored_out(scalar_type absNdotL)
    {
        return absNdotL;
    }

    scalar_type eval(NBL_CONST_REF_ARG(params_t) params)
    {
        return __eval_pi_factored_out(params.NdotL) * numbers::inv_pi<scalar_type> * 0.5;
    }

    sample_type generate_wo_clamps(NBL_CONST_REF_ARG(anisotropic_type) interaction, NBL_CONST_REF_ARG(vector<scalar_type, 3>) u)
    {
        ray_dir_info_type L;
        L.direction = sampling::ProjectedSphere<scalar_type>::generate(u);
        return sample_type::createFromTangentSpace(interaction.getTangentSpaceV(), L, interaction.getFromTangentSpace());
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_type) interaction, NBL_CONST_REF_ARG(vector<scalar_type, 3>) u)
    {
        return generate_wo_clamps(interaction, u);
    }

    scalar_type pdf(NBL_CONST_REF_ARG(params_t) params)
    {
        return sampling::ProjectedSphere<scalar_type>::pdf(params.NdotL);
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(params_t) params)
    {
        return sampling::ProjectedSphere<scalar_type>::template quotient_and_pdf<spectral_type>(params.NdotL);
    }
};


// microfacet bxdfs
template<class LightSample, class IsoCache, class AnisoCache, class Spectrum, bool thin> // NBL_FUNC_REQUIRES(Sample<LightSample> && IsotropicMicrofacetCache<IsoCache> && AnisotropicMicrofacetCache<AnisoCache>) // dxc won't let me put this in
struct SSmoothDielectricBxDF;

template<class LightSample, class IsoCache, class AnisoCache, class Spectrum>
struct SSmoothDielectricBxDF<LightSample, IsoCache, AnisoCache, Spectrum, false>
{
    using this_t = SSmoothDielectricBxDF<LightSample, IsoCache, AnisoCache, Spectrum, false>;
    using scalar_type = typename LightSample::scalar_type;
    using ray_dir_info_type = typename LightSample::ray_dir_info_type;
    using vector3_type = vector<scalar_type, 3>;
    using params_t = SBxDFParams<scalar_type>;

    using isotropic_type = typename IsoCache::isotropic_type;
    using anisotropic_type = typename AnisoCache::anisotropic_type;
    using sample_type = LightSample;
    using spectral_type = Spectrum;
    using quotient_pdf_type = quotient_and_pdf<spectral_type, scalar_type>;
    using isocache_type = IsoCache;
    using anisocache_type = AnisoCache;

    static this_t create(scalar_type eta)
    {
        this_t retval;
        retval.eta = eta;
        return retval;
    }

    static this_t create(NBL_CONST_REF_ARG(SBxDFCreationParams<scalar_type, spectral_type>) params)
    {
        return create(params.eta);
    }

    void init(NBL_CONST_REF_ARG(SBxDFCreationParams<scalar_type, spectral_type>) params)
    {
        eta = params.eta;
    }

    spectral_type eval(NBL_CONST_REF_ARG(params_t) params)
    {
        return (spectral_type)0;
    }

    sample_type __generate_wo_clamps(NBL_CONST_REF_ARG(vector3_type) V, NBL_CONST_REF_ARG(vector3_type) T, NBL_CONST_REF_ARG(vector3_type) B, NBL_CONST_REF_ARG(vector3_type) N, bool backside, scalar_type NdotV, scalar_type absNdotV, scalar_type NdotV2, NBL_REF_ARG(vector3_type) u, scalar_type rcpOrientedEta, scalar_type orientedEta2, scalar_type rcpOrientedEta2, NBL_REF_ARG(bool) transmitted)
    {
        const scalar_type reflectance = fresnel<scalar_type>::dielectric_common(orientedEta2, absNdotV);

        scalar_type rcpChoiceProb;
        transmitted = math::partitionRandVariable(reflectance, u.z, rcpChoiceProb);

        ray_dir_info_type L;
        refract<vector3_type> r = refract<vector3_type>::create(V, N, backside, NdotV, NdotV2, rcpOrientedEta, rcpOrientedEta2);
        L.direction = r.doReflectRefract(transmitted);
        return sample_type::create(L, nbl::hlsl::dot<vector3_type>(V, L.direction), T, B, N);
    }

    sample_type generate_wo_clamps(NBL_CONST_REF_ARG(anisotropic_type) interaction, NBL_REF_ARG(vector<scalar_type, 3>) u)
    {
        scalar_type orientedEta, rcpOrientedEta;
        const bool backside = bxdf::getOrientedEtas<scalar_type>(orientedEta, rcpOrientedEta, interaction.isotropic.NdotV, eta);
        bool dummy;
        return __generate_wo_clamps(interaction.isotropic.V.direction, interaction.T, interaction.B, interaction.isotropic.N, backside, interaction.isotropic.NdotV, 
            interaction.isotropic.NdotV, interaction.isotropic.NdotV*interaction.isotropic.NdotV, u, rcpOrientedEta, orientedEta*orientedEta, rcpOrientedEta*rcpOrientedEta, dummy);
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_type) interaction, NBL_REF_ARG(vector<scalar_type, 3>) u)
    {
        scalar_type orientedEta, rcpOrientedEta;
        const bool backside = bxdf::getOrientedEtas<scalar_type>(orientedEta, rcpOrientedEta, interaction.isotropic.NdotV, eta);
        bool dummy;
        return __generate_wo_clamps(interaction.isotropic.V.direction, interaction.T, interaction.B, interaction.isotropic.N, backside, interaction.isotropic.NdotV, 
            nbl::hlsl::abs<scalar_type>(interaction.isotropic.NdotV), interaction.isotropic.NdotV*interaction.isotropic.NdotV, u, rcpOrientedEta, orientedEta*orientedEta, rcpOrientedEta*rcpOrientedEta, dummy);
    }

    // eval and pdf return 0 because smooth dielectric/conductor BxDFs are dirac delta distributions, model perfectly specular objects that scatter light to only one outgoing direction
    scalar_type pdf(NBL_CONST_REF_ARG(params_t) params)
    {
        return 0;
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(params_t) params)
    {
        const bool transmitted = isTransmissionPath(params.uNdotV, params.uNdotL);

        scalar_type dummy, rcpOrientedEta;
        const bool backside = bxdf::getOrientedEtas<scalar_type>(dummy, rcpOrientedEta, params.NdotV, eta);

        const scalar_type _pdf = bit_cast<scalar_type, uint32_t>(numeric_limits<scalar_type>::infinity);
        scalar_type quo = transmitted ? rcpOrientedEta : 1.0;
        return quotient_pdf_type::create((spectral_type)(quo), _pdf);
    }

    scalar_type eta;
};

template<class LightSample, class IsoCache, class AnisoCache, class Spectrum>
struct SSmoothDielectricBxDF<LightSample, IsoCache, AnisoCache, Spectrum, true>
{
    using this_t = SSmoothDielectricBxDF<LightSample, IsoCache, AnisoCache, Spectrum, true>;
    using scalar_type = typename LightSample::scalar_type;
    using ray_dir_info_type = typename LightSample::ray_dir_info_type;
    using vector3_type = vector<scalar_type, 3>;
    using params_t = SBxDFParams<scalar_type>;

    using isotropic_type = typename IsoCache::isotropic_type;
    using anisotropic_type = typename AnisoCache::anisotropic_type;
    using sample_type = LightSample;
    using spectral_type = Spectrum;
    using quotient_pdf_type = quotient_and_pdf<spectral_type, scalar_type>;
    using isocache_type = IsoCache;
    using anisocache_type = AnisoCache;

    static this_t create(NBL_CONST_REF_ARG(spectral_type) eta2, NBL_CONST_REF_ARG(spectral_type) luminosityContributionHint)
    {
        this_t retval;
        retval.eta2 = eta2;
        retval.luminosityContributionHint = luminosityContributionHint;
        return retval;
    }

    static this_t create(NBL_CONST_REF_ARG(SBxDFCreationParams<scalar_type, spectral_type>) params)
    {
        return create(params.eta2, params.luminosityContributionHint);
    }

    void init(NBL_CONST_REF_ARG(SBxDFCreationParams<scalar_type, spectral_type>) params)
    {
        eta2 = params.eta2;
        luminosityContributionHint = params.luminosityContributionHint;
    }

    spectral_type eval(NBL_CONST_REF_ARG(params_t) params)
    {
        return (spectral_type)0;
    }

    // usually `luminosityContributionHint` would be the Rec.709 luma coefficients (the Y row of the RGB to CIE XYZ matrix)
    // its basically a set of weights that determine
    // assert(1.0==luminosityContributionHint.r+luminosityContributionHint.g+luminosityContributionHint.b);
    // `remainderMetadata` is a variable which the generator function returns byproducts of sample generation that would otherwise have to be redundantly calculated `quotient_and_pdf`
    sample_type __generate_wo_clamps(NBL_CONST_REF_ARG(vector3_type) V, NBL_CONST_REF_ARG(vector3_type) T, NBL_CONST_REF_ARG(vector3_type) B, NBL_CONST_REF_ARG(vector3_type) N, scalar_type NdotV, scalar_type absNdotV, NBL_REF_ARG(vector3_type) u, NBL_CONST_REF_ARG(spectral_type) eta2, NBL_CONST_REF_ARG(spectral_type) luminosityContributionHint, NBL_REF_ARG(spectral_type) remainderMetadata)
    {
        // we will only ever intersect from the outside
        const spectral_type reflectance = thindielectricInfiniteScatter<spectral_type>(fresnel<spectral_type>::dielectric_common(eta2,absNdotV));

        // we are only allowed one choice for the entire ray, so make the probability a weighted sum
        const scalar_type reflectionProb = nbl::hlsl::dot<spectral_type>(reflectance, luminosityContributionHint);

        scalar_type rcpChoiceProb;
        const bool transmitted = math::partitionRandVariable(reflectionProb, u.z, rcpChoiceProb);
        remainderMetadata = (transmitted ? ((spectral_type)(1.0) - reflectance) : reflectance) * rcpChoiceProb;

        ray_dir_info_type L;
        L.direction = (transmitted ? (vector3_type)(0.0) : N * 2.0f * NdotV) - V;
        return sample_type::create(L, nbl::hlsl::dot<vector3_type>(V, L.direction), T, B, N);
    }

    sample_type generate_wo_clamps(NBL_CONST_REF_ARG(anisotropic_type) interaction, NBL_REF_ARG(vector<scalar_type, 3>) u)
    {
        vector3_type dummy;
        return __generate_wo_clamps(interaction.isotropic.V.direction, interaction.T, interaction.B, interaction.isotropic.N, interaction.isotropic.NdotV, interaction.isotropic.NdotV, u, eta2, luminosityContributionHint, dummy);
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_type) interaction, NBL_REF_ARG(vector<scalar_type, 3>) u)
    {
        vector3_type dummy;
        return __generate_wo_clamps(interaction.isotropic.V.direction, interaction.T, interaction.B, interaction.isotropic.N, interaction.isotropic.NdotV, nbl::hlsl::abs<scalar_type>(interaction.isotropic.NdotV), u, eta2, luminosityContributionHint, dummy);
    }

    scalar_type pdf(NBL_CONST_REF_ARG(params_t) params)
    {
        return 0;
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(params_t) params)   // isotropic
    {
        const bool transmitted = isTransmissionPath(params.uNdotV, params.uNdotL);
        const spectral_type reflectance = thindielectricInfiniteScatter<spectral_type>(fresnel<spectral_type>::dielectric_common(eta2, params.NdotV));
        const spectral_type sampleValue = transmitted ? ((spectral_type)(1.0) - reflectance) : reflectance;

        const scalar_type sampleProb = nbl::hlsl::dot<spectral_type>(sampleValue,luminosityContributionHint);

        const scalar_type _pdf = bit_cast<scalar_type, uint32_t>(numeric_limits<scalar_type>::infinity);
        return quotient_pdf_type::create((spectral_type)(sampleValue / sampleProb), _pdf);
    }

    spectral_type eta2;
    spectral_type luminosityContributionHint;
};

template<class LightSample, class IsoCache, class AnisoCache, class Spectrum NBL_FUNC_REQUIRES(Sample<LightSample> && IsotropicMicrofacetCache<IsoCache> && AnisotropicMicrofacetCache<AnisoCache>)
struct SBeckmannDielectricBxDF
{
    using this_t = SBeckmannDielectricBxDF<LightSample, IsoCache, AnisoCache, Spectrum>;
    using scalar_type = typename LightSample::scalar_type;
    using ray_dir_info_type = typename LightSample::ray_dir_info_type;
    using vector2_type = vector<scalar_type, 2>;
    using vector3_type = vector<scalar_type, 3>;
    using matrix3x3_type = matrix<scalar_type,3,3>;
    using params_t = SBxDFParams<scalar_type>;

    using isotropic_type = typename IsoCache::isotropic_type;
    using anisotropic_type = typename AnisoCache::anisotropic_type;
    using sample_type = LightSample;
    using spectral_type = Spectrum;
    using quotient_pdf_type = quotient_and_pdf<spectral_type, scalar_type>;
    using isocache_type = IsoCache;
    using anisocache_type = AnisoCache;

    static this_t create(scalar_type eta, scalar_type A)
    {
        this_t retval;
        retval.eta = eta;
        retval.A = vector2_type(A, A);
        return retval;
    }

    static this_t create(scalar_type eta, scalar_type ax, scalar_type ay)
    {
        this_t retval;
        retval.eta = eta;
        retval.A = vector2_type(ax, ay);
        return retval;
    }

    static this_t create(NBL_CONST_REF_ARG(SBxDFCreationParams<scalar_type, spectral_type>) params)
    {
        if (params.is_aniso)
            return create(params.eta, params.A.x, params.A.y);
        else
            return create(params.eta, params.A.x);
    }

    void init(NBL_CONST_REF_ARG(SBxDFCreationParams<scalar_type, spectral_type>) params)
    {
        A = params.A;
        eta = params.eta;
    }

    spectral_type eval(NBL_CONST_REF_ARG(params_t) params)
    {
        scalar_type orientedEta, dummy;
        const bool backside = bxdf::getOrientedEtas<scalar_type>(orientedEta, dummy, params.VdotH, eta);
        const scalar_type orientedEta2 = orientedEta * orientedEta;

        const scalar_type VdotHLdotH = params.VdotH * params.LdotH;
        const bool transmitted = VdotHLdotH < 0.0;

        spectral_type dummyior;
        reflection::SBeckmannBxDF<sample_type, isocache_type, anisocache_type, spectral_type> beckmann;
        if (params.is_aniso)
            beckmann = reflection::SBeckmannBxDF<sample_type, isocache_type, anisocache_type, spectral_type>::create(A.x, A.y, dummyior, dummyior);
        else
            beckmann = reflection::SBeckmannBxDF<sample_type, isocache_type, anisocache_type, spectral_type>::create(A.x, dummyior, dummyior);
        const scalar_type scalar_part = beckmann.__eval_DG_wo_clamps(params);

        ndf::microfacet_to_light_measure_transform<ndf::Beckmann<scalar_type>,ndf::REFLECT_REFRACT_BIT> microfacet_transform =
            ndf::microfacet_to_light_measure_transform<ndf::Beckmann<scalar_type>,ndf::REFLECT_REFRACT_BIT>::create(scalar_part,params.NdotV,transmitted,params.VdotH,params.LdotH,VdotHLdotH,orientedEta);
        return (spectral_type)fresnel<scalar_type>::dielectric_common(orientedEta2, nbl::hlsl::abs<scalar_type>(params.VdotH)) * microfacet_transform();
    }

    sample_type __generate_wo_clamps(NBL_CONST_REF_ARG(vector3_type) localV, bool backside, NBL_CONST_REF_ARG(vector3_type) H, NBL_CONST_REF_ARG(matrix3x3_type) m, NBL_REF_ARG(vector3_type) u, scalar_type rcpOrientedEta, scalar_type orientedEta2, scalar_type rcpOrientedEta2, NBL_REF_ARG(anisocache_type) cache)
    {
        const scalar_type localVdotH = nbl::hlsl::dot<vector3_type>(localV,H);
        const scalar_type reflectance = fresnel<scalar_type>::dielectric_common(orientedEta2,nbl::hlsl::abs<scalar_type>(localVdotH));
        
        scalar_type rcpChoiceProb;
        bool transmitted = math::partitionRandVariable(reflectance, u.z, rcpChoiceProb);

        cache = anisocache_type::create(localV, H);

        const scalar_type VdotH = cache.iso_cache.VdotH;
        cache.iso_cache.LdotH = transmitted ? refract<vector3_type>::computeNdotT(VdotH < 0.0, VdotH * VdotH, rcpOrientedEta2) : VdotH;
        ray_dir_info_type localL;
        localL.direction = refract<vector3_type>::doReflectRefract(transmitted, localV, H, VdotH, cache.iso_cache.LdotH, rcpOrientedEta);

        return sample_type::createFromTangentSpace(localV, localL, m);
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_type) interaction, NBL_REF_ARG(vector3_type) u, NBL_REF_ARG(anisocache_type) cache)
    {
        const vector3_type localV = interaction.getTangentSpaceV();

        scalar_type orientedEta, rcpOrientedEta;
        const bool backside = bxdf::getOrientedEtas<scalar_type>(orientedEta, rcpOrientedEta, interaction.isotropic.NdotV, eta);

        const vector3_type upperHemisphereV = backside ? -localV : localV;

        spectral_type dummyior;
        reflection::SBeckmannBxDF<sample_type, isocache_type, anisocache_type, spectral_type> beckmann = reflection::SBeckmannBxDF<sample_type, isocache_type, anisocache_type, spectral_type>::create(A.x, A.y, dummyior, dummyior);
        const vector3_type H = beckmann.__generate(upperHemisphereV, u.xy);

        return __generate_wo_clamps(localV, backside, H, interaction.getFromTangentSpace(), u, rcpOrientedEta, orientedEta*orientedEta, rcpOrientedEta*rcpOrientedEta, cache);
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_type) interaction, NBL_REF_ARG(vector3_type) u)
    {
        anisocache_type dummycache;
        return generate(interaction, u, dummycache);
    }

    scalar_type pdf(NBL_CONST_REF_ARG(params_t) params, NBL_REF_ARG(scalar_type) onePlusLambda_V)
    {
        scalar_type orientedEta, dummy;
        const bool backside = bxdf::getOrientedEtas<scalar_type>(orientedEta, dummy, params.VdotH, eta);
        const scalar_type orientedEta2 = orientedEta * orientedEta;

        const scalar_type VdotHLdotH = params.VdotH * params.LdotH;
        const bool transmitted = VdotHLdotH < 0.0;

        const scalar_type reflectance = fresnel<scalar_type>::dielectric_common(orientedEta2, nbl::hlsl::abs<scalar_type>(params.VdotH));
        
        scalar_type ndf, lambda;
        if (params.is_aniso)
        {
            const scalar_type ax2 = A.x*A.x;
            const scalar_type ay2 = A.y*A.y;
            ndf::SAnisotropicParams<scalar_type> ndfparams = ndf::SAnisotropicParams<scalar_type>::create(A.x, A.y, ax2, ay2, params.TdotH2, params.BdotH2, params.NdotH2);
            ndf::Beckmann<scalar_type> beckmann_ndf;
            ndf = beckmann_ndf(ndfparams);

            smith::Beckmann<scalar_type> beckmann_smith;
            scalar_type c2 = beckmann_smith.C2(params.TdotV2, params.BdotV2, params.NdotV2, ax2, ay2);
            lambda = beckmann_smith.Lambda(c2);
        }
        else
        {
            const scalar_type a2 = A.x*A.x;
            ndf::SIsotropicParams<scalar_type> ndfparams = ndf::SIsotropicParams<scalar_type>::create(a2, params.NdotH, params.NdotH2);
            ndf::Beckmann<scalar_type> beckmann_ndf;
            ndf = beckmann_ndf(ndfparams);

            smith::Beckmann<scalar_type> beckmann_smith;
            lambda = beckmann_smith.Lambda(params.NdotV2, a2);
        }

        return smith::VNDF_pdf_wo_clamps<smith::Beckmann<scalar_type> >(ndf,lambda,params.NdotV,transmitted,params.VdotH,params.LdotH,VdotHLdotH,orientedEta,reflectance,onePlusLambda_V);
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

        scalar_type quo;
        if (params.is_aniso)
        {
            smith::SAnisotropicParams<scalar_type> smithparams = smith::SAnisotropicParams<scalar_type>::create(A.x*A.x, A.y*A.y, params.TdotV2, params.BdotV2, params.NdotV2, params.TdotL2, params.BdotL2, params.NdotL2, onePlusLambda_V);
            smith::Beckmann<scalar_type> beckmann_smith;
            quo = beckmann_smith.G2_over_G1(smithparams);
        }
        else
        {
            smith::SIsotropicParams<scalar_type> smithparams = smith::SIsotropicParams<scalar_type>::create(A.x*A.x, params.NdotV2, params.NdotL2, onePlusLambda_V);
            smith::Beckmann<scalar_type> beckmann_smith;
            quo = beckmann_smith.G2_over_G1(smithparams);
        }

        return quotient_pdf_type::create((spectral_type)(quo), _pdf);
    }

    vector2_type A;
    scalar_type eta;
};

template<class LightSample, class IsoCache, class AnisoCache, class Spectrum NBL_FUNC_REQUIRES(Sample<LightSample> && IsotropicMicrofacetCache<IsoCache> && AnisotropicMicrofacetCache<AnisoCache>)
struct SGGXDielectricBxDF
{
    using this_t = SGGXDielectricBxDF<LightSample, IsoCache, AnisoCache, Spectrum>;
    using scalar_type = typename LightSample::scalar_type;
    using ray_dir_info_type = typename LightSample::ray_dir_info_type;
    using vector2_type = vector<scalar_type, 2>;
    using vector3_type = vector<scalar_type, 3>;
    using matrix3x3_type = matrix<scalar_type,3,3>;
    using params_t = SBxDFParams<scalar_type>;

    using isotropic_type = typename IsoCache::isotropic_type;
    using anisotropic_type = typename AnisoCache::anisotropic_type;
    using sample_type = LightSample;
    using spectral_type = Spectrum;
    using quotient_pdf_type = quotient_and_pdf<spectral_type, scalar_type>;
    using isocache_type = IsoCache;
    using anisocache_type = AnisoCache;

    static this_t create(scalar_type eta, scalar_type A)
    {
        this_t retval;
        retval.eta = eta;
        retval.A = vector2_type(A, A);
        return retval;
    }

    static this_t create(scalar_type eta, scalar_type ax, scalar_type ay)
    {
        this_t retval;
        retval.eta = eta;
        retval.A = vector2_type(ax, ay);
        return retval;
    }

    static this_t create(NBL_CONST_REF_ARG(SBxDFCreationParams<scalar_type, spectral_type>) params)
    {
        if (params.is_aniso)
            return create(params.eta, params.A.x, params.A.y);
        else
            return create(params.eta, params.A.x);
    }

    void init(NBL_CONST_REF_ARG(SBxDFCreationParams<scalar_type, spectral_type>) params)
    {
        A = params.A;
        eta = params.eta;
    }

    spectral_type eval(NBL_CONST_REF_ARG(params_t) params)
    {
        scalar_type orientedEta, dummy;
        const bool backside = bxdf::getOrientedEtas<scalar_type>(orientedEta, dummy, params.VdotH, eta);
        const scalar_type orientedEta2 = orientedEta * orientedEta;

        const scalar_type VdotHLdotH = params.VdotH * params.LdotH;
        const bool transmitted = VdotHLdotH < 0.0;

        scalar_type NG_already_in_reflective_dL_measure;
        if (params.is_aniso)
        {
            spectral_type dummyior;
            reflection::SGGXBxDF<sample_type, isocache_type, anisocache_type, spectral_type> ggx = reflection::SGGXBxDF<sample_type, isocache_type, anisocache_type, spectral_type>::create(A.x, A.y, dummyior, dummyior);
            NG_already_in_reflective_dL_measure = ggx.__eval_DG_wo_clamps(params);
        }
        else
        {
            spectral_type dummyior;
            reflection::SGGXBxDF<sample_type, isocache_type, anisocache_type, spectral_type> ggx = reflection::SGGXBxDF<sample_type, isocache_type, anisocache_type, spectral_type>::create(A.x, dummyior, dummyior);
            NG_already_in_reflective_dL_measure = ggx.__eval_DG_wo_clamps(params);
        }

        ndf::microfacet_to_light_measure_transform<ndf::GGX<scalar_type>,ndf::REFLECT_REFRACT_BIT> microfacet_transform =
            ndf::microfacet_to_light_measure_transform<ndf::GGX<scalar_type>,ndf::REFLECT_REFRACT_BIT>::create(NG_already_in_reflective_dL_measure,params.NdotL,transmitted,params.VdotH,params.LdotH,VdotHLdotH,orientedEta);
        return (spectral_type)fresnel<scalar_type>::dielectric_common(orientedEta2, nbl::hlsl::abs<scalar_type>(params.VdotH)) * microfacet_transform();
    }

    sample_type __generate_wo_clamps(NBL_CONST_REF_ARG(vector3_type) localV, bool backside, NBL_CONST_REF_ARG(vector3_type) H, NBL_CONST_REF_ARG(matrix3x3_type) m, NBL_REF_ARG(vector3_type) u, scalar_type rcpOrientedEta, scalar_type orientedEta2, scalar_type rcpOrientedEta2, NBL_REF_ARG(anisocache_type) cache)
    {
        const scalar_type localVdotH = nbl::hlsl::dot<vector3_type>(localV,H);
        const scalar_type reflectance = fresnel<scalar_type>::dielectric_common(orientedEta2,nbl::hlsl::abs<scalar_type>(localVdotH));
        
        scalar_type rcpChoiceProb;
        bool transmitted = math::partitionRandVariable(reflectance, u.z, rcpChoiceProb);

        cache = anisocache_type::create(localV, H);

        const scalar_type VdotH = cache.iso_cache.VdotH;
        cache.iso_cache.LdotH = transmitted ? refract<vector3_type>::computeNdotT(VdotH < 0.0, VdotH * VdotH, rcpOrientedEta2) : VdotH;
        ray_dir_info_type localL;
        localL.direction = refract<vector3_type>::doReflectRefract(transmitted, localV, H, VdotH, cache.iso_cache.LdotH, rcpOrientedEta);

        return sample_type::createFromTangentSpace(localV, localL, m);
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_type) interaction, NBL_REF_ARG(vector3_type) u, NBL_REF_ARG(anisocache_type) cache)
    {
        const vector3_type localV = interaction.getTangentSpaceV();

        scalar_type orientedEta, rcpOrientedEta;
        const bool backside = bxdf::getOrientedEtas<scalar_type>(orientedEta, rcpOrientedEta, interaction.isotropic.NdotV, eta);

        const vector3_type upperHemisphereV = backside ? -localV : localV;

        spectral_type dummyior;
        reflection::SGGXBxDF<sample_type, isocache_type, anisocache_type, spectral_type> ggx = reflection::SGGXBxDF<sample_type, isocache_type, anisocache_type, spectral_type>::create(A.x, A.y, dummyior, dummyior);
        const vector3_type H = ggx.__generate(upperHemisphereV, u.xy);

        return __generate_wo_clamps(localV, backside, H, interaction.getFromTangentSpace(), u, rcpOrientedEta, orientedEta*orientedEta, rcpOrientedEta*rcpOrientedEta, cache);
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_type) interaction, NBL_REF_ARG(vector3_type) u)
    {
        anisocache_type dummycache;
        return generate(interaction, u, dummycache);
    }

    scalar_type pdf(NBL_CONST_REF_ARG(params_t) params)
    {
        scalar_type orientedEta, dummy;
        const bool backside = bxdf::getOrientedEtas<scalar_type>(orientedEta, dummy, params.VdotH, eta);
        const scalar_type orientedEta2 = orientedEta * orientedEta;

        const scalar_type VdotHLdotH = params.VdotH * params.LdotH;
        const bool transmitted = VdotHLdotH < 0.0;

        const scalar_type reflectance = fresnel<scalar_type>::dielectric_common(orientedEta2, nbl::hlsl::abs<scalar_type>(params.VdotH));

        scalar_type ndf, devsh_v;
        if (params.is_aniso)
        {
            const scalar_type ax2 = A.x*A.x;
            const scalar_type ay2 = A.y*A.y;

            ndf::SAnisotropicParams<scalar_type> ndfparams = ndf::SAnisotropicParams<scalar_type>::create(A.x, A.y, ax2, ay2, params.TdotH2, params.BdotH2, params.NdotH2);
            ndf::GGX<scalar_type> ggx_ndf;
            ndf = ggx_ndf(ndfparams);

            smith::GGX<scalar_type> ggx_smith;
            devsh_v = ggx_smith.devsh_part(params.TdotV2, params.BdotV2, params.NdotV2, ax2, ay2);
        }
        else
        {
            const scalar_type a2 = A.x*A.x;
            ndf::SIsotropicParams<scalar_type> ndfparams = ndf::SIsotropicParams<scalar_type>::create(a2, params.NdotH, params.NdotH2);
            ndf::GGX<scalar_type> ggx_ndf;
            ndf = ggx_ndf(ndfparams);

            smith::GGX<scalar_type> ggx_smith;
            devsh_v = ggx_smith.devsh_part(params.NdotV2, a2, 1.0-a2);
        }

        smith::GGX<scalar_type> ggx_smith;
        const scalar_type lambda = ggx_smith.G1_wo_numerator(params.NdotV, devsh_v);
        return smith::VNDF_pdf_wo_clamps<scalar_type>(ndf, lambda, params.NdotV, transmitted, params.VdotH, params.LdotH, VdotHLdotH, orientedEta, reflectance);
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(params_t) params)
    {
        const scalar_type ax2 = A.x*A.x;
        const scalar_type ay2 = A.y*A.y;

        scalar_type _pdf = pdf(params);

        smith::GGX<scalar_type> ggx_smith;
        scalar_type quo;
        if (params.is_aniso)
        {
            smith::SAnisotropicParams<scalar_type> smithparams = smith::SAnisotropicParams<scalar_type>::create(ax2, ay2, params.NdotV, params.TdotV2, params.BdotV2, params.NdotV2, params.NdotL, params.TdotL2, params.BdotL2, params.NdotL2);
            quo = ggx_smith.G2_over_G1(smithparams);
        }
        else
        {
            smith::SIsotropicParams<scalar_type> smithparams = smith::SIsotropicParams<scalar_type>::create(ax2, params.NdotV, params.NdotV2, params.NdotL, params.NdotL2);
            quo = ggx_smith.G2_over_G1(smithparams);
        }

        return quotient_pdf_type::create((spectral_type)(quo), _pdf);
    }

    vector2_type A;
    scalar_type eta;
};

}
}
}
}

#endif
