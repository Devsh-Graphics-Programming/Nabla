// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_COOK_TORRANCE_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_COOK_TORRANCE_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/config.hlsl"
#include "nbl/builtin/hlsl/bxdf/ndf.hlsl"
#include "nbl/builtin/hlsl/bxdf/fresnel.hlsl"
#include "nbl/builtin/hlsl/sampling/basic.hlsl"
#include "nbl/builtin/hlsl/bxdf/ndf/microfacet_to_light_transform.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{

namespace impl
{
template<class N, class F, bool IsBSDF>
struct quant_query_helper;

template<class N, class F>
struct quant_query_helper<N, F, true>
{
    using quant_query_type = typename N::quant_query_type;

    template<class I, class C>
    static quant_query_type __call(NBL_CONST_REF_ARG(N) ndf, NBL_CONST_REF_ARG(F) fresnel, NBL_CONST_REF_ARG(I) interaction, NBL_CONST_REF_ARG(C) cache)
    {
        return ndf.template createQuantQuery<I,C>(interaction, cache, fresnel.getRefractionOrientedEta());
    }
};

template<class N, class F>
struct quant_query_helper<N, F, false>
{
    using quant_query_type = typename N::quant_query_type;

    template<class I, class C>
    static quant_query_type __call(NBL_CONST_REF_ARG(N) ndf, NBL_CONST_REF_ARG(F) fresnel, NBL_CONST_REF_ARG(I) interaction, NBL_CONST_REF_ARG(C) cache)
    {
        typename N::scalar_type dummy;
        return ndf.template createQuantQuery<I,C>(interaction, cache, dummy);
    }
};
}

// N (NDF), F (fresnel)
template<class Config, class N, class F NBL_PRIMARY_REQUIRES(config_concepts::MicrofacetConfiguration<Config> && ndf::NDF<N> && fresnel::Fresnel<F>)
struct SCookTorrance
{
    MICROFACET_BXDF_CONFIG_TYPE_ALIASES(Config);

    using this_t = SCookTorrance<Config, N, F>;
    using quant_type = typename N::quant_type;
    using ndf_type = N;
    using fresnel_type = F;

    NBL_CONSTEXPR_STATIC_INLINE bool IsAnisotropic = ndf_type::IsAnisotropic;
    NBL_CONSTEXPR_STATIC_INLINE bool IsBSDF = ndf_type::SupportedPaths != ndf::MTT_REFLECT;
    using random_type = conditional_t<IsBSDF, vector3_type, vector2_type>;
    NBL_HLSL_BXDF_ANISOTROPIC_COND_DECLS(IsAnisotropic);
    using cache_type = conditional_t<IsAnisotropic,anisocache_type,isocache_type>;

    struct PdfQuery
    {
        scalar_type pdf;
        fresnel_type orientedFresnel;
        typename ndf_type::quant_query_type quantQuery;
        
        spectral_type reflectance;
        scalar_type scaled_reflectance;
    };

    // utility functions
    template<class Interaction=conditional_t<IsAnisotropic,anisotropic_interaction_type,isotropic_interaction_type>, 
            class MicrofacetCache=conditional_t<IsAnisotropic,anisocache_type,isocache_type>
            NBL_FUNC_REQUIRES(!ndf::NDF_CanOverwriteDG<ndf_type> && RequiredInteraction<Interaction> && RequiredMicrofacetCache<MicrofacetCache>)
    static void __overwriteDG(NBL_REF_ARG(scalar_type) DG, ndf_type ndf, NBL_CONST_REF_ARG(typename ndf_type::g2g1_query_type) query, NBL_CONST_REF_ARG(typename ndf_type::quant_query_type) quant_query,
                    NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache, NBL_REF_ARG(bool) isInfinity)
    {
    }
    template<class Interaction=conditional_t<IsAnisotropic,anisotropic_interaction_type,isotropic_interaction_type>, 
            class MicrofacetCache=conditional_t<IsAnisotropic,anisocache_type,isocache_type>
            NBL_FUNC_REQUIRES(ndf::NDF_CanOverwriteDG<ndf_type> && RequiredInteraction<Interaction> && RequiredMicrofacetCache<MicrofacetCache>)
    static void __overwriteDG(NBL_REF_ARG(scalar_type) DG, ndf_type ndf, NBL_CONST_REF_ARG(typename ndf_type::g2g1_query_type) query, NBL_CONST_REF_ARG(typename ndf_type::quant_query_type) quant_query,
                    NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache, NBL_REF_ARG(bool) isInfinity)
    {
        quant_type dg = ndf.template Dcorrelated<sample_type, Interaction, MicrofacetCache>(query, quant_query, _sample, interaction, cache, isInfinity);
        DG = dg.projectedLightMeasure;
    }

    template<typename PH=fresnel_type NBL_FUNC_REQUIRES(!fresnel::TwoSidedFresnel<PH>)
    static fresnel_type __getOrientedFresnel(NBL_CONST_REF_ARG(fresnel_type) fresnel, scalar_type NdotV)
    {
        // expect conductor fresnel
        return fresnel;
    }
    template<typename PH=fresnel_type NBL_FUNC_REQUIRES(fresnel::TwoSidedFresnel<PH>)
    static fresnel_type __getOrientedFresnel(NBL_CONST_REF_ARG(fresnel_type) fresnel, scalar_type NdotV)
    {
        return fresnel.getReorientedFresnel(NdotV);
    }

    template<class Interaction=conditional_t<IsAnisotropic,anisotropic_interaction_type,isotropic_interaction_type>,
            class MicrofacetCache=conditional_t<IsAnisotropic,anisocache_type,isocache_type>, typename C=bool_constant<!IsBSDF> NBL_FUNC_REQUIRES(C::value && !IsBSDF)
    static bool __checkValid(NBL_CONST_REF_ARG(fresnel_type) orientedFresnel, NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        return _sample.getNdotL() > numeric_limits<scalar_type>::min && interaction.getNdotV() > numeric_limits<scalar_type>::min;
    }
    template<class Interaction=conditional_t<IsAnisotropic,anisotropic_interaction_type,isotropic_interaction_type>,
            class MicrofacetCache=conditional_t<IsAnisotropic,anisocache_type,isocache_type>, typename C=bool_constant<IsBSDF> NBL_FUNC_REQUIRES(C::value && IsBSDF)
    static bool __checkValid(NBL_CONST_REF_ARG(fresnel_type) orientedFresnel, NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        fresnel::OrientedEtas<monochrome_type> orientedEta = fresnel::OrientedEtas<monochrome_type>::create(scalar_type(1.0), hlsl::promote<monochrome_type>(orientedFresnel.getRefractionOrientedEta()));
        return cache.isValid(orientedEta);
    }

    template<class Interaction=conditional_t<IsAnisotropic,anisotropic_interaction_type,isotropic_interaction_type>,
            typename C=bool_constant<!fresnel_type::ReturnsMonochrome> NBL_FUNC_REQUIRES(C::value && !fresnel_type::ReturnsMonochrome)
    static scalar_type __getScaledReflectance(NBL_CONST_REF_ARG(fresnel_type) orientedFresnel, NBL_CONST_REF_ARG(Interaction) interaction, scalar_type clampedVdotH, bool transmitted, NBL_REF_ARG(spectral_type) outFresnelVal)
    {
        spectral_type throughputWeights = interaction.getLuminosityContributionHint();
        spectral_type reflectance = orientedFresnel(clampedVdotH);
        outFresnelVal = hlsl::mix(reflectance, hlsl::promote<spectral_type>(1.0)-reflectance, transmitted);
        return hlsl::dot<spectral_type>(outFresnelVal, throughputWeights);
    }
    template<class Interaction=conditional_t<IsAnisotropic,anisotropic_interaction_type,isotropic_interaction_type>,
            typename C=bool_constant<fresnel_type::ReturnsMonochrome> NBL_FUNC_REQUIRES(C::value && fresnel_type::ReturnsMonochrome)
    static scalar_type __getScaledReflectance(NBL_CONST_REF_ARG(fresnel_type) orientedFresnel, NBL_CONST_REF_ARG(Interaction) interaction, scalar_type clampedVdotH, bool transmitted, NBL_REF_ARG(spectral_type) outFresnelVal)
    {
        scalar_type reflectance = orientedFresnel(clampedVdotH)[0];
        reflectance = hlsl::mix(reflectance, scalar_type(1.0)-reflectance, transmitted);
        outFresnelVal = hlsl::promote<spectral_type>(reflectance);
        return reflectance;
    }

    bool __dotIsValue(const vector3_type a, const vector3_type b, const scalar_type value) NBL_CONST_MEMBER_FUNC
    {
        const scalar_type ab = hlsl::dot(a, b);
        return hlsl::max(ab, value / ab) <= scalar_type(value + 1e-3);
    }

    template<class Interaction=conditional_t<IsAnisotropic,anisotropic_interaction_type,isotropic_interaction_type>,
            typename C=bool_constant<!fresnel_type::ReturnsMonochrome> NBL_FUNC_REQUIRES(C::value && !fresnel_type::ReturnsMonochrome)
    static scalar_type __getMonochromeEta(NBL_CONST_REF_ARG(fresnel_type) fresnel, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        spectral_type throughputWeights = interaction.getLuminosityContributionHint();
        return hlsl::dot<spectral_type>(fresnel.eta, throughputWeights) / (throughputWeights.r + throughputWeights.g + throughputWeights.b);
    }
    template<class Interaction=conditional_t<IsAnisotropic,anisotropic_interaction_type,isotropic_interaction_type>,
            typename C=bool_constant<fresnel_type::ReturnsMonochrome> NBL_FUNC_REQUIRES(C::value && fresnel_type::ReturnsMonochrome)
    static scalar_type __getMonochromeEta(NBL_CONST_REF_ARG(fresnel_type) fresnel, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        return fresnel.getRefractionOrientedEta();
    }

    // bxdf stuff
    template<class Interaction=conditional_t<IsAnisotropic,anisotropic_interaction_type,isotropic_interaction_type>, 
            class MicrofacetCache=conditional_t<IsAnisotropic,anisocache_type,isocache_type>
            NBL_FUNC_REQUIRES(RequiredInteraction<Interaction> && RequiredMicrofacetCache<MicrofacetCache>)
    value_weight_type evalAndWeight(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache) NBL_CONST_MEMBER_FUNC
    {
        PdfQuery pdfQuery = __forwardPdf<Interaction, MicrofacetCache, false>(_sample, interaction, cache);
        scalar_type _pdf = pdfQuery.pdf;
        if (_pdf == scalar_type(0.0) || hlsl::isinf(_pdf))
            return value_weight_type::create(scalar_type(0.0), scalar_type(0.0));

        fresnel_type _f = pdfQuery.orientedFresnel;

        using quant_query_type = typename ndf_type::quant_query_type;
        quant_query_type qq = pdfQuery.quantQuery;

        using g2g1_query_type = typename ndf_type::g2g1_query_type;
        g2g1_query_type gq = ndf.template createG2G1Query<sample_type, Interaction>(_sample, interaction);

        bool isInfinity;
        quant_type D = ndf.template D<sample_type, Interaction, MicrofacetCache>(qq, _sample, interaction, cache, isInfinity);
        scalar_type DG = D.projectedLightMeasure;
        if (!isInfinity)
            DG *= ndf.template correlated<sample_type, Interaction, MicrofacetCache>(gq, _sample, interaction, cache);

        __overwriteDG<Interaction, MicrofacetCache>(DG, ndf, gq, qq, _sample, interaction, cache, isInfinity);

        // immediately return only after all calls setting DG
        // allows compiler to throw away calls to ndf.D if using __overwriteDG, before that we only avoid computation for G2(correlated)
        if (isInfinity)
            return value_weight_type::create(scalar_type(0.0), scalar_type(0.0));
        
        spectral_type eval;
        NBL_IF_CONSTEXPR(IsBSDF)
            eval = pdfQuery.reflectance;
        else
            eval = _f(cache.getVdotH());
        eval *= DG;

        return value_weight_type::create(eval, _pdf);
    }

    template<class Interaction=conditional_t<IsAnisotropic,anisotropic_interaction_type,isotropic_interaction_type>, 
            class MicrofacetCache=conditional_t<IsAnisotropic,anisocache_type,isocache_type>
            NBL_FUNC_REQUIRES(RequiredInteraction<Interaction> && RequiredMicrofacetCache<MicrofacetCache>)
    value_weight_type evalAndWeight(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(Interaction) interaction) NBL_CONST_MEMBER_FUNC
    {
        const scalar_type eta = __getMonochromeEta(fresnel, interaction);
        fresnel::OrientedEtas<monochrome_type> orientedEta = fresnel::OrientedEtas<monochrome_type>::create(interaction.getNdotV(), hlsl::promote<monochrome_type>(eta));
        MicrofacetCache cache = MicrofacetCache::template create<anisotropic_interaction_type, sample_type>(interaction, _sample, orientedEta);
        return evalAndWeight(_sample, interaction, cache);
    }

    sample_type __generate_common(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, const vector3_type localH,
                                const scalar_type NdotV, const scalar_type VdotH, const scalar_type LdotH, bool transmitted,
                                NBL_CONST_REF_ARG(fresnel::OrientedEtaRcps<monochrome_type>) rcpEta, 
                                NBL_REF_ARG(bool) valid) NBL_CONST_MEMBER_FUNC
    {
        // fail if samples have invalid paths
        const scalar_type NdotL = hlsl::mix(scalar_type(2.0) * VdotH * localH.z - NdotV,
                                    localH.z * (VdotH * rcpEta.value[0] + LdotH) - NdotV * rcpEta.value[0], transmitted);
        assert(!hlsl::isnan(NdotL));
        // VNDF sampling guarantees that `VdotH` has same sign as `NdotV`
        // and `transmitted` controls the sign of `LdotH` relative to `VdotH` by construction (reflect -> same sign, or refract -> opposite sign)
        if (ComputeMicrofacetNormal<scalar_type>::isTransmissionPath(NdotV, NdotL) != transmitted)
        {
            valid = false;
            return sample_type::createInvalid();    // should check if sample direction is invalid
        }

        ray_dir_info_type V = interaction.getV();
        const matrix3x3_type fromTangent = interaction.getFromTangentSpace();
        // tangent frame orthonormality
        assert(__dotIsValue(fromTangent[0],fromTangent[1],0.0));
        assert(__dotIsValue(fromTangent[1],fromTangent[2],0.0));
        assert(__dotIsValue(fromTangent[2],fromTangent[0],0.0));
        // NDF sampling produced a unit length direction
        assert(__dotIsValue(localH,localH,1.0));
        const vector3_type H = hlsl::mul(interaction.getFromTangentSpace(), localH);
        Refract<scalar_type> r = Refract<scalar_type>::create(V.getDirection(), H);

        struct reflect_refract_wrapper  // so we don't recalculate LdotH
        {
            vector3_type operator()(const bool doRefract, const scalar_type rcpOrientedEta) NBL_CONST_MEMBER_FUNC
            {
                return rr(NdotTorR, rcpOrientedEta);
            }
            bxdf::ReflectRefract<scalar_type> rr;
            scalar_type NdotTorR;
        };
        bxdf::ReflectRefract<scalar_type> rr;   // rr.getNdotTorR() and calls to mix as well as a good part of the computations should CSE with our computation of NdotL above
        rr.refract = r;
        reflect_refract_wrapper rrw;
        rrw.rr = rr;
        rrw.NdotTorR = LdotH;
        ray_dir_info_type L = V.template reflectRefract<reflect_refract_wrapper>(rrw, transmitted, rcpEta.value[0]);

        const vector3_type T = interaction.getT();
        const vector3_type B = interaction.getB();

        valid = true;
        return sample_type::create(L, T, B, NdotL);
    }
    template<typename C=bool_constant<!IsBSDF> NBL_FUNC_REQUIRES(C::value && !IsBSDF)
    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, const random_type u, NBL_REF_ARG(anisocache_type) cache) NBL_CONST_MEMBER_FUNC
    {
        const scalar_type NdotV = interaction.getNdotV();
        if (NdotV < numeric_limits<scalar_type>::min)
            return sample_type::createInvalid();
        assert(!hlsl::isnan(NdotV));

        const vector3_type localV = interaction.getTangentSpaceV();
        const vector3_type localH = ndf.generateH(localV, u);
        const scalar_type VdotH = hlsl::dot(localV, localH);
        NBL_IF_CONSTEXPR(!ndf_type::GuaranteedVNDF)   // VNDF sampling guarantees VdotH has same sign as NdotV (should be positive for BRDF)
        {
            // allow for rejection sampling, theoretically NdotV=0 or VdotH=0 is valid, but leads to 0 value contribution anyway
            if (VdotH <= scalar_type(0.0))
                return sample_type::createInvalid();
            assert(!hlsl::isnan(NdotV*VdotH));
        }
        else
        {
            assert(VdotH >= scalar_type(0.0));
        }

        fresnel::OrientedEtaRcps<monochrome_type> dummy;
        bool valid;
        sample_type s = __generate_common(interaction, localH, NdotV, VdotH, VdotH, false, dummy, valid);
        if (valid)
            cache = anisocache_type::createForReflection(localV, localH);
        return s;
    }
    template<typename C=bool_constant<IsBSDF> NBL_FUNC_REQUIRES(C::value && IsBSDF)
    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, const random_type u, NBL_REF_ARG(anisocache_type) cache) NBL_CONST_MEMBER_FUNC
    {
        const vector3_type localV = interaction.getTangentSpaceV();
        const scalar_type NdotV = localV.z;

        fresnel_type _f = __getOrientedFresnel(fresnel, NdotV);
        fresnel::OrientedEtaRcps<monochrome_type> rcpEta = _f.getRefractionOrientedEtaRcps();

        const vector3_type upperHemisphereV = ieee754::flipSignIfRHSNegative<vector3_type>(localV, hlsl::promote<vector3_type>(NdotV));
        const vector3_type localH = ndf.generateH(upperHemisphereV, u.xy);
        const scalar_type VdotH = hlsl::dot(localV, localH);

        NBL_IF_CONSTEXPR(!ndf_type::GuaranteedVNDF)
        {
            // allow for rejection sampling, theoretically NdotV=0 or VdotH=0 is valid, but leads to 0 value contribution anyway
            if (NdotV*VdotH <= scalar_type(0.0))
                return sample_type::createInvalid();
            assert(!hlsl::isnan(NdotV*VdotH));
        }
        else
        {
            assert(NdotV*VdotH >= scalar_type(0.0));
        }

        spectral_type dummy;
        const scalar_type reflectance = __getScaledReflectance(_f, interaction, hlsl::abs(VdotH), false, dummy);

        scalar_type rcpChoiceProb;
        scalar_type z = u.z;
        sampling::PartitionRandVariable<scalar_type> partitionRandVariable;
        partitionRandVariable.leftProb = reflectance;
        bool transmitted = partitionRandVariable(z, rcpChoiceProb);

        scalar_type LdotH;
        if (transmitted)
        {
            scalar_type det = rcpEta.value2[0]*VdotH*VdotH + scalar_type(1.0) - rcpEta.value2[0];
            if (det < scalar_type(0.0))
                return sample_type::createInvalid();
            LdotH = ieee754::copySign(hlsl::sqrt(det), -VdotH);
        }
        else
            LdotH = VdotH;
        bool valid;
        sample_type s = __generate_common(interaction, localH, NdotV, VdotH, LdotH, transmitted, rcpEta, valid);
        if (valid)
        {
            cache = anisocache_type::createPartial(VdotH, LdotH, localH.z, transmitted, rcpEta);
            assert(cache.isValid(fresnel::OrientedEtas<monochrome_type>::create(scalar_type(1.0), hlsl::promote<monochrome_type>(_f.getRefractionOrientedEta()))));
            const vector3_type T = interaction.getT();
            const vector3_type B = interaction.getB();
            const vector3_type H = hlsl::mul(interaction.getFromTangentSpace(), localH);
            cache.fillTangents(T, B, H);
        }
        return s;
    }
    template<typename C=bool_constant<!IsAnisotropic> NBL_FUNC_REQUIRES(C::value && !IsAnisotropic)
    sample_type generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, const random_type u, NBL_REF_ARG(isocache_type) cache) NBL_CONST_MEMBER_FUNC
    {
        anisocache_type aniso_cache;
        sample_type s = generate(anisotropic_interaction_type::create(interaction), u, aniso_cache);
        cache = aniso_cache.iso_cache;
        return s;
    }

    template<class Interaction, class MicrofacetCache, bool FromGenerator>
    PdfQuery __forwardPdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache) NBL_CONST_MEMBER_FUNC
    {
        // MicrofacetCache cache = _cache;
        // cache.VdotH = 0.5;
        PdfQuery query;
        query.pdf = scalar_type(0.0);
        query.orientedFresnel = __getOrientedFresnel(fresnel, interaction.getNdotV());
        const bool cacheIsValid = __checkValid<Interaction, MicrofacetCache>(query.orientedFresnel, _sample, interaction, cache);
        assert(!FromGenerator || cacheIsValid);
        const bool isValid = FromGenerator ? _sample.isValid() : cacheIsValid; // when from generator, expect the generated sample to always be valid, different checks for brdf and btdf

        if (isValid)
        {
            using dg1_query_type = typename ndf_type::dg1_query_type;
            dg1_query_type dq = ndf.template createDG1Query<Interaction, MicrofacetCache>(interaction, cache);

            bool isInfinity;
            query.quantQuery = impl::quant_query_helper<ndf_type, fresnel_type, IsBSDF>::template __call<Interaction, MicrofacetCache>(ndf, query.orientedFresnel, interaction, cache);
            quant_type DG1 = ndf.template DG1<sample_type, Interaction>(dq, query.quantQuery, _sample, interaction, isInfinity);

            if (isInfinity)
            {
                query.pdf = bit_cast<scalar_type>(numeric_limits<scalar_type>::infinity);
                return query;
            }

            query.pdf = DG1.projectedLightMeasure;
            NBL_IF_CONSTEXPR(IsBSDF)
            {
                query.scaled_reflectance = __getScaledReflectance(query.orientedFresnel, interaction, hlsl::abs(cache.getVdotH()), cache.isTransmission(), query.reflectance);
                query.pdf *= query.scaled_reflectance;
            }
        }

        return query;
    }
    template<class Interaction=conditional_t<IsAnisotropic,anisotropic_interaction_type,isotropic_interaction_type>, 
            class MicrofacetCache=conditional_t<IsAnisotropic,anisocache_type,isocache_type>
            NBL_FUNC_REQUIRES(RequiredInteraction<Interaction> && RequiredMicrofacetCache<MicrofacetCache>)
    scalar_type forwardPdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache) NBL_CONST_MEMBER_FUNC
    {
        PdfQuery query = __forwardPdf<Interaction, MicrofacetCache, false>(_sample, interaction, cache);
        return query.pdf;
    }

    template<class Interaction=conditional_t<IsAnisotropic,anisotropic_interaction_type,isotropic_interaction_type>, 
            class MicrofacetCache=conditional_t<IsAnisotropic,anisocache_type,isocache_type>
            NBL_FUNC_REQUIRES(RequiredInteraction<Interaction> && RequiredMicrofacetCache<MicrofacetCache>)
    quotient_weight_type quotientAndWeight(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache) NBL_CONST_MEMBER_FUNC
    {
        PdfQuery pdfQuery = __forwardPdf<Interaction, MicrofacetCache, true>(_sample, interaction, cache);
        scalar_type _pdf = pdfQuery.pdf;
        if (_pdf == scalar_type(0.0))
            return quotient_weight_type::create(scalar_type(0.0), scalar_type(0.0));

        fresnel_type _f = pdfQuery.orientedFresnel;

        scalar_type G2_over_G1 = scalar_type(1.0);
        if (!hlsl::isinf(_pdf))
        {
            using g2g1_query_type = typename N::g2g1_query_type;
            g2g1_query_type gq = ndf.template createG2G1Query<sample_type, Interaction>(_sample, interaction);
            G2_over_G1 = ndf.template G2_over_G1<sample_type, Interaction, MicrofacetCache>(gq, _sample, interaction, cache);
        }

        spectral_type quo;
        NBL_IF_CONSTEXPR(IsBSDF)
        {
            NBL_IF_CONSTEXPR(fresnel_type::ReturnsMonochrome)
                quo = hlsl::promote<spectral_type>(G2_over_G1);
            else
                quo = pdfQuery.reflectance / pdfQuery.scaled_reflectance * G2_over_G1;
        }
        else
        {
            const scalar_type VdotH = cache.getVdotH();
            assert(VdotH > scalar_type(0.0));
            quo = _f(VdotH) * G2_over_G1;
        }

        return quotient_weight_type::create(quo, _pdf);
    }

    ndf_type ndf;
    fresnel_type fresnel;   // always front-facing
};


template<class Config, class N, class F>
struct traits<SCookTorrance<Config,N,F> >
{
   using __type = SCookTorrance<Config,N,F>;

    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = conditional_value<__type::IsBSDF, BxDFType, BxDFType::BT_BSDF, BxDFType::BT_BRDF>::value;
    NBL_CONSTEXPR_STATIC_INLINE bool IsMicrofacet = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = !__type::IsBSDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = !__type::IsBSDF;
    NBL_CONSTEXPR_STATIC_INLINE bool TractablePdf = true;
};

}
}
}

#endif
