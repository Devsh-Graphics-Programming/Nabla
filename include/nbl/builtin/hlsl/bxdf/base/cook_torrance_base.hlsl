// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_COOK_TORRANCE_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_COOK_TORRANCE_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/config.hlsl"
#include "nbl/builtin/hlsl/bxdf/ndf.hlsl"
#include "nbl/builtin/hlsl/bxdf/fresnel.hlsl"
#include "nbl/builtin/hlsl/bxdf/ndf/microfacet_to_light_transform.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{

namespace impl
{
template<typename T, typename U>
struct __implicit_promote;

template<typename T>
struct __implicit_promote<T,T>
{
    static T __call(const T v)
    {
        return v;
    }
};

template<typename T>
struct __implicit_promote<T,vector<typename vector_traits<T>::scalar_type, 1> >
{
    static T __call(const vector<typename vector_traits<T>::scalar_type, 1> v)
    {
        return hlsl::promote<T>(v[0]);
    }
};

template<class N, class F, bool IsBSDF>
struct quant_query_helper;

template<class N, class F>
struct quant_query_helper<N, F, true>
{
    using quant_query_type = typename N::quant_query_type;

    template<class C>
    static quant_query_type __call(NBL_REF_ARG(N) ndf, NBL_CONST_REF_ARG(F) fresnel, NBL_CONST_REF_ARG(C) cache)
    {
        return ndf.template createQuantQuery<C>(cache, fresnel.orientedEta.value[0]);
    }
};

template<class N, class F>
struct quant_query_helper<N, F, false>
{
    using quant_query_type = typename N::quant_query_type;

    template<class C>
    static quant_query_type __call(NBL_REF_ARG(N) ndf, NBL_CONST_REF_ARG(F) fresnel, NBL_CONST_REF_ARG(C) cache)
    {
        typename N::scalar_type dummy;
        return ndf.template createQuantQuery<C>(cache, dummy);
    }
};

template<class F, bool IsBSDF>
struct check_TIR_helper;

template<class F>
struct check_TIR_helper<F, false>
{
    template<class MicrofacetCache>
    static bool __call(NBL_CONST_REF_ARG(F) fresnel, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        return true;
    }
};

template<class F>
struct check_TIR_helper<F, true>
{
    using vector_type = typename F::vector_type;    // expect monochrome

    template<class MicrofacetCache>
    static bool __call(NBL_CONST_REF_ARG(F) fresnel, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        fresnel::OrientedEtas<vector_type> orientedEta = fresnel::OrientedEtas<vector_type>::create(typename F::scalar_type(1.0), hlsl::promote<vector_type>(fresnel.getRefractionOrientedEta()));
        return cache.isValid(orientedEta);
    }
};

template<class F, bool IsBSDF NBL_STRUCT_CONSTRAINABLE>
struct getOrientedFresnel;

template<class F>
struct getOrientedFresnel<F, false>
{
    static F __call(NBL_CONST_REF_ARG(F) fresnel, typename F::scalar_type NdotV)
    {
        // expect conductor fresnel
        return fresnel;
    }
};

template<class F>
NBL_PARTIAL_REQ_TOP(fresnel::TwoSidedFresnel<F>)
struct getOrientedFresnel<F, true NBL_PARTIAL_REQ_BOT(fresnel::TwoSidedFresnel<F>) >
{
    using scalar_type = typename F::scalar_type;

    static F __call(NBL_CONST_REF_ARG(F) fresnel, scalar_type NdotV)
    {
        return fresnel.getReorientedFresnel(NdotV);
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

    template<class Interaction, class MicrofacetCache>
    static bool __checkValid(NBL_CONST_REF_ARG(fresnel_type) f, NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        NBL_IF_CONSTEXPR(IsBSDF)
            return impl::check_TIR_helper<fresnel_type, IsBSDF>::template __call<MicrofacetCache>(f, cache);
        else
            return _sample.getNdotL() > numeric_limits<scalar_type>::min && interaction.getNdotV() > numeric_limits<scalar_type>::min;
    }

    template<class Interaction, class MicrofacetCache>
    spectral_type __eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        fresnel_type _f = fresnel;
        NBL_IF_CONSTEXPR(IsBSDF)
            _f = impl::getOrientedFresnel<fresnel_type, IsBSDF>::__call(fresnel, interaction.getNdotV());
        if (!__checkValid<Interaction, MicrofacetCache>(_f, _sample, interaction, cache))
            return hlsl::promote<spectral_type>(0.0);

        using quant_query_type = typename ndf_type::quant_query_type;
        scalar_type dummy;
        quant_query_type qq = ndf.template createQuantQuery<MicrofacetCache>(cache, dummy);

        quant_type D = ndf.template D<sample_type, Interaction, MicrofacetCache>(qq, _sample, interaction, cache);
        scalar_type DG = D.projectedLightMeasure;
        if (D.microfacetMeasure < bit_cast<scalar_type>(numeric_limits<scalar_type>::infinity))
        {
            using g2g1_query_type = typename ndf_type::g2g1_query_type;
            g2g1_query_type gq = ndf.template createG2G1Query<sample_type, Interaction>(_sample, interaction);
            DG *= ndf.template correlated<sample_type, Interaction>(gq, _sample, interaction);
        }
        scalar_type clampedVdotH = cache.getVdotH();
        NBL_IF_CONSTEXPR(IsBSDF)
            clampedVdotH = hlsl::abs(clampedVdotH);
        return impl::__implicit_promote<spectral_type, typename fresnel_type::vector_type>::__call(_f(clampedVdotH)) * DG;
    }
    template<typename C=bool_constant<!IsAnisotropic> >
    enable_if_t<C::value && !IsAnisotropic, spectral_type> eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        return __eval<isotropic_interaction_type, isocache_type>(_sample, interaction, cache);
    }
    spectral_type eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        return __eval<anisotropic_interaction_type, anisocache_type>(_sample, interaction, cache);
    }

    template<typename C=bool_constant<!IsBSDF> >
    enable_if_t<C::value && !IsBSDF, sample_type> generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, const vector2_type u, NBL_REF_ARG(anisocache_type) cache)
    {
        if (interaction.getNdotV() > numeric_limits<scalar_type>::min)
        {
            ray_dir_info_type invalidL;
            invalidL.makeInvalid();
            return sample_type::createFromTangentSpace(invalidL, interaction.getFromTangentSpace());
        }

        const vector3_type localV = interaction.getTangentSpaceV();
        const vector3_type localH = ndf.generateH(localV, u);
        const scalar_type VdotH = hlsl::dot(localV, localH);
        const vector3_type H = hlsl::mul(interaction.getFromTangentSpace(), localH);

        ray_dir_info_type L;
        if (scalar_type(2.0) * VdotH * localH.z > localV.z) // NdotL>0, compiler's Common Subexpression Elimination pass should re-use 2*VdotH later
        {
            assert(VdotH >= scalar_type(0.0));
            ray_dir_info_type V = interaction.getV();
            struct reflect_wrapper  // so we don't recalculate VdotH
            {
                vector3_type operator()() NBL_CONST_MEMBER_FUNC
                {
                    return r(VdotH);
                }
                bxdf::Reflect<scalar_type> r;
                scalar_type VdotH;
            };
            reflect_wrapper rw;
            rw.r = bxdf::Reflect<scalar_type>::create(V.getDirection(), H);
            rw.VdotH = VdotH;
            L = V.template reflect<reflect_wrapper>(rw);

            cache = anisocache_type::createForReflection(localV, localH);
        }
        else    // fail if samples have invalid paths
            L.makeInvalid();    // should check if sample direction is invalid

        const vector3_type T = interaction.getT();
        const vector3_type B = interaction.getB();
        const vector3_type _N = interaction.getN();

        return sample_type::create(L, T, B, _N);
    }
    template<typename C=bool_constant<IsBSDF> >
    enable_if_t<C::value && IsBSDF, sample_type> generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, const vector3_type u, NBL_REF_ARG(anisocache_type) cache)
    {
        const vector3_type localV = interaction.getTangentSpaceV();
        const scalar_type NdotV = localV.z;

        fresnel_type _f = impl::getOrientedFresnel<fresnel_type, IsBSDF>::__call(fresnel, NdotV);
        fresnel::OrientedEtaRcps<monochrome_type> rcpEta = _f.getOrientedEtaRcps();

        const vector3_type upperHemisphereV = ieee754::flipSignIfRHSNegative<vector3_type>(localV, hlsl::promote<vector3_type>(NdotV));
        const vector3_type localH = ndf.generateH(upperHemisphereV, u.xy);
        const scalar_type VdotH = hlsl::dot(localV, localH);
        const vector3_type H = hlsl::mul(interaction.getFromTangentSpace(), localH);

        assert(NdotV * VdotH > scalar_type(0.0));
        const scalar_type reflectance = _f(hlsl::abs(VdotH))[0];

        scalar_type rcpChoiceProb;
        scalar_type z = u.z;
        bool transmitted = math::partitionRandVariable(reflectance, z, rcpChoiceProb);

        ray_dir_info_type V = interaction.getV();
        const vector3_type _N = interaction.getN();
        Refract<scalar_type> r = Refract<scalar_type>::create(V.getDirection(), H);
        const scalar_type LdotH = hlsl::mix(VdotH, r.getNdotT(rcpEta.value2[0]), transmitted);
        cache = anisocache_type::createPartial(VdotH, LdotH, hlsl::dot(_N, H), transmitted, rcpEta);

        struct reflect_refract_wrapper  // so we don't recalculate LdotH
        {
            vector3_type operator()(const bool doRefract, const scalar_type rcpOrientedEta) NBL_CONST_MEMBER_FUNC
            {
                return rr(NdotTorR, rcpOrientedEta);
            }
            bxdf::ReflectRefract<scalar_type> rr;
            scalar_type NdotTorR;
        };
        bxdf::ReflectRefract<scalar_type> rr;
        rr.refract = r;
        reflect_refract_wrapper rrw;
        rrw.rr = rr;
        rrw.NdotTorR = LdotH;
        ray_dir_info_type L = V.template reflectRefract<reflect_refract_wrapper>(rrw, transmitted, rcpEta.value[0]);

        const vector3_type T = interaction.getT();
        const vector3_type B = interaction.getB();

        // fail if samples have invalid paths
        const scalar_type NdotL = scalar_type(2.0) * VdotH * localH.z - localV.z;
        if ((ComputeMicrofacetNormal<scalar_type>::isTransmissionPath(NdotV, NdotL) != transmitted))
            L.makeInvalid(); // should check if sample direction is invalid
        else
        {
            fresnel::OrientedEtas<monochrome_type> orientedEta = fresnel::OrientedEtas<monochrome_type>::create(scalar_type(1.0), hlsl::promote<monochrome_type>(fresnel.getRefractionOrientedEta()));
            assert(ComputeMicrofacetNormal::isValidMicrofacet(transmitted,cache.getVdotL(),localH.z,orientedEta));
            cache.fillTangents(T, B, H);
        }

        return sample_type::create(L, T, B, _N);
    }
    template<typename C=bool_constant<!IsAnisotropic>, typename T=conditional_t<IsBSDF, vector3_type, vector2_type> >
    enable_if_t<C::value && !IsAnisotropic, sample_type> generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, const T u, NBL_REF_ARG(isocache_type) cache)
    {
        anisocache_type aniso_cache;
        sample_type s = generate(anisotropic_interaction_type::create(interaction), u, aniso_cache);
        cache = aniso_cache.iso_cache;
        return s;
    }

    template<class Interaction, class MicrofacetCache>
    scalar_type __pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        using quant_query_type = typename ndf_type::quant_query_type;
        using dg1_query_type = typename ndf_type::dg1_query_type;

        dg1_query_type dq = ndf.template createDG1Query<Interaction, MicrofacetCache>(interaction, cache);

        fresnel_type _f = impl::getOrientedFresnel<fresnel_type, IsBSDF>::__call(fresnel, interaction.getNdotV());
        quant_query_type qq = impl::quant_query_helper<ndf_type, fresnel_type, IsBSDF>::template __call<MicrofacetCache>(ndf, _f, cache);
        quant_type DG1 = ndf.template DG1<sample_type, Interaction>(dq, qq, _sample, interaction);

        NBL_IF_CONSTEXPR(IsBSDF)
        {
            const scalar_type reflectance = _f(hlsl::abs(cache.getVdotH()))[0];
            return hlsl::mix(reflectance, scalar_type(1.0) - reflectance, cache.isTransmission()) * DG1.projectedLightMeasure;
        }
        else
        {
            return DG1.projectedLightMeasure;
        }
    }
    template<typename C=bool_constant<!IsAnisotropic> >
    enable_if_t<C::value && !IsAnisotropic, scalar_type> pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        fresnel_type _f = fresnel;
        NBL_IF_CONSTEXPR(IsBSDF)
            _f = impl::getOrientedFresnel<fresnel_type, IsBSDF>::__call(fresnel, interaction.getNdotV());
        if (!__checkValid<isotropic_interaction_type, isocache_type>(_f, _sample, interaction, cache))
            return scalar_type(0.0);

        scalar_type _pdf = __pdf<isotropic_interaction_type, isocache_type>(_sample, interaction, cache);
        return hlsl::mix(scalar_type(0.0), _pdf, _pdf < bit_cast<scalar_type>(numeric_limits<scalar_type>::infinity));
    }
    scalar_type pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        fresnel_type _f = fresnel;
        NBL_IF_CONSTEXPR(IsBSDF)
            _f = impl::getOrientedFresnel<fresnel_type, IsBSDF>::__call(fresnel, interaction.getNdotV());
        if (!__checkValid<anisotropic_interaction_type, anisocache_type>(_f, _sample, interaction, cache))
            return scalar_type(0.0);

        scalar_type _pdf = __pdf<anisotropic_interaction_type, anisocache_type>(_sample, interaction, cache);
        return hlsl::mix(scalar_type(0.0), _pdf, _pdf < bit_cast<scalar_type>(numeric_limits<scalar_type>::infinity));
    }

    template<class Interaction, class MicrofacetCache>
    quotient_pdf_type __quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        if (!_sample.isValid())
            return quotient_pdf_type::create(scalar_type(0.0), scalar_type(0.0));   // set pdf=0 when quo=0 because we don't want to give high weight to sampling strategy that yields 0 contribution

        scalar_type _pdf = __pdf<Interaction, MicrofacetCache>(_sample, interaction, cache);
        fresnel_type _f = impl::getOrientedFresnel<fresnel_type, IsBSDF>::__call(fresnel, interaction.getNdotV());

        const bool notTIR = impl::check_TIR_helper<fresnel_type, IsBSDF>::template __call<MicrofacetCache>(_f, cache);
        assert(notTIR);
        
        scalar_type G2_over_G1 = scalar_type(1.0);
        if (_pdf < bit_cast<scalar_type>(numeric_limits<scalar_type>::infinity))
        {
            using g2g1_query_type = typename N::g2g1_query_type;
            g2g1_query_type gq = ndf.template createG2G1Query<sample_type, Interaction>(_sample, interaction);
            G2_over_G1 = ndf.template G2_over_G1<sample_type, Interaction, MicrofacetCache>(gq, _sample, interaction, cache);
        }

        spectral_type quo;
        NBL_IF_CONSTEXPR(IsBSDF)
            quo = hlsl::promote<spectral_type>(G2_over_G1);
        else
        {
            const scalar_type VdotH = cache.getVdotH();
            assert(VdotH > scalar_type(0.0));
            quo = _f(VdotH) * G2_over_G1;
        }

        return quotient_pdf_type::create(quo, _pdf);
    }
    template<typename C=bool_constant<!IsAnisotropic> >
    enable_if_t<C::value && !IsAnisotropic, quotient_pdf_type> quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        return __quotient_and_pdf<isotropic_interaction_type, isocache_type>(_sample, interaction, cache);
    }
    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        return __quotient_and_pdf<anisotropic_interaction_type, anisocache_type>(_sample, interaction, cache);
    }

    ndf_type ndf;
    fresnel_type fresnel;   // always front-facing
};

}
}
}

#endif
