// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_GGX_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_GGX_INCLUDED_

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

template<class LS, class IsoCache, class AnisoCache, class Spectrum NBL_FUNC_REQUIRES(LightSample<LS> && CreatableIsotropicMicrofacetCache<IsoCache> && AnisotropicMicrofacetCache<AnisoCache>)
struct SGGXDielectricBxDF
{
    using this_t = SGGXDielectricBxDF<LS, IsoCache, AnisoCache, Spectrum>;
    using scalar_type = typename LS::scalar_type;
    using ray_dir_info_type = typename LS::ray_dir_info_type;
    using vector2_type = vector<scalar_type, 2>;
    using vector3_type = vector<scalar_type, 3>;
    using matrix3x3_type = matrix<scalar_type,3,3>;
    using params_t = SBxDFParams<scalar_type>;

    using isotropic_type = typename IsoCache::isotropic_type;
    using anisotropic_type = typename AnisoCache::anisotropic_type;
    using sample_type = LS;
    using spectral_type = Spectrum;
    using quotient_pdf_type = sampling::quotient_and_pdf<spectral_type, scalar_type>;
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
        fresnel::OrientedEtas<scalar_type> orientedEta = fresnel::OrientedEtas<scalar_type>::create(params.VdotH, eta);
        const scalar_type orientedEta2 = orientedEta.value * orientedEta.value;


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
            ndf::microfacet_to_light_measure_transform<ndf::GGX<scalar_type>,ndf::REFLECT_REFRACT_BIT>::create(NG_already_in_reflective_dL_measure,params.NdotL,transmitted,params.VdotH,params.LdotH,VdotHLdotH,orientedEta.value);
        scalar_type f = fresnel::Dielectric<scalar_type>::__call(orientedEta2, nbl::hlsl::abs<scalar_type>(params.VdotH));
        return hlsl::promote<spectral_type>(f) * microfacet_transform();
    }

    sample_type __generate_wo_clamps(NBL_CONST_REF_ARG(vector3_type) localV, NBL_CONST_REF_ARG(vector3_type) H, NBL_CONST_REF_ARG(matrix3x3_type) m, NBL_REF_ARG(vector3_type) u, NBL_CONST_REF_ARG(fresnel::OrientedEtas<scalar_type>) orientedEta, NBL_CONST_REF_ARG(fresnel::OrientedEtaRcps<scalar_type>) rcpEta, NBL_REF_ARG(anisocache_type) cache)
    {
        const scalar_type localVdotH = nbl::hlsl::dot<vector3_type>(localV,H);
        const scalar_type reflectance = fresnel::Dielectric<scalar_type>::__call(orientedEta.value * orientedEta.value,nbl::hlsl::abs<scalar_type>(localVdotH));
        
        scalar_type rcpChoiceProb;
        bool transmitted = math::partitionRandVariable(reflectance, u.z, rcpChoiceProb);

        cache = anisocache_type::create(localV, H);

        const scalar_type VdotH = cache.iso_cache.getVdotH();
        Refract<scalar_type> r;
        r.recomputeNdotT(VdotH < 0.0, VdotH * VdotH, rcpEta.value2);
        cache.iso_cache.LdotH = hlsl::mix(VdotH, r.NdotT, transmitted);
        ray_dir_info_type localL;
        bxdf::ReflectRefract<scalar_type> rr = bxdf::ReflectRefract<scalar_type>::create(transmitted, localV, H, VdotH, cache.iso_cache.getLdotH(), rcpEta.value);
        localL.direction = rr(transmitted);

        return sample_type::createFromTangentSpace(localV, localL, m);
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_type) interaction, NBL_REF_ARG(vector3_type) u, NBL_REF_ARG(anisocache_type) cache)
    {
        const vector3_type localV = interaction.getTangentSpaceV();

        fresnel::OrientedEtas<scalar_type> orientedEta = fresnel::OrientedEtas<scalar_type>::create(interaction.isotropic.getNdotV(), eta);
        fresnel::OrientedEtaRcps<scalar_type> rcpEta = fresnel::OrientedEtaRcps<scalar_type>::create(interaction.isotropic.getNdotV(), eta);

        const vector3_type upperHemisphereV = orientedEta.backside ? -localV : localV;

        spectral_type dummyior;
        reflection::SGGXBxDF<sample_type, isocache_type, anisocache_type, spectral_type> ggx = reflection::SGGXBxDF<sample_type, isocache_type, anisocache_type, spectral_type>::create(A.x, A.y, dummyior, dummyior);
        const vector3_type H = ggx.__generate(upperHemisphereV, u.xy);

        return __generate_wo_clamps(localV, H, interaction.getFromTangentSpace(), u, orientedEta, rcpEta, cache);
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_type) interaction, NBL_REF_ARG(vector3_type) u)
    {
        anisocache_type dummycache;
        return generate(interaction, u, dummycache);
    }

    sample_type generate(NBL_CONST_REF_ARG(isotropic_type) interaction, NBL_REF_ARG(vector3_type) u, NBL_REF_ARG(isocache_type) cache)
    {
        anisocache_type anisocache;
        sample_type s = generate(anisotropic_type::create(interaction), u, anisocache);
        cache = anisocache.iso_cache;
        return s;
    }

    scalar_type pdf(NBL_CONST_REF_ARG(params_t) params)
    {
        fresnel::OrientedEtas<scalar_type> orientedEta = fresnel::OrientedEtas<scalar_type>::create(params.VdotH, eta);
        const scalar_type orientedEta2 = orientedEta.value * orientedEta.value;

        const scalar_type VdotHLdotH = params.VdotH * params.LdotH;
        const bool transmitted = VdotHLdotH < 0.0;

        const scalar_type reflectance = fresnel::Dielectric<scalar_type>::__call(orientedEta2, nbl::hlsl::abs<scalar_type>(params.VdotH));

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

        smith::bsdf::VNDF_pdf<ndf::GGX<scalar_type> > vndf = smith::bsdf::VNDF_pdf<ndf::GGX<scalar_type> >::create(ndf, params.NdotV);
        return vndf(lambda, transmitted, params.VdotH, params.LdotH, VdotHLdotH, orientedEta.value, reflectance);
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
