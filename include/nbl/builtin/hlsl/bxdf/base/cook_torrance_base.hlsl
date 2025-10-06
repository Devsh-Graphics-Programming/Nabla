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
    template<class MicrofacetCache>
    static bool __call(NBL_CONST_REF_ARG(F) fresnel, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        return ComputeMicrofacetNormal<typename F::scalar_type>::isValidMicrofacet(cache.isTransmission(), cache.getVdotL(), cache.getAbsNdotH(), fresnel.getRefractionOrientedEta());
    }
};

template<class F, bool IsBSDF>
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
struct getOrientedFresnel<F, true>
{
    using scalar_type = typename F::scalar_type;

    static F __call(NBL_CONST_REF_ARG(F) fresnel, scalar_type NdotV)
    {
        fresnel::OrientedEtas<typename F::vector_type> orientedEta = fresnel::OrientedEtas<typename F::vector_type>::create(NdotV, fresnel.orientedEta.value);
        return F::create(orientedEta);
    }
};

template<class F, typename Spectral, bool IsAnisotropic>
struct CookTorranceParams;

template<class F, typename Spectral>
struct CookTorranceParams<F,Spectral,false>
{
    using scalar_type = typename F::scalar_type;
    using spectral_type = Spectral;

    scalar_type A;
    F fresnel;
};

template<class F, typename Spectral>
struct CookTorranceParams<F,Spectral,true>
{
    using scalar_type = typename F::scalar_type;
    using spectral_type = Spectral;

    scalar_type ax;
    scalar_type ay;
    F fresnel;
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
    NBL_CONSTEXPR_STATIC_INLINE bool IsBSDF = ndf_type::IsBSDF;

    using creation_params_type = impl::CookTorranceParams<fresnel_type, spectral_type, IsAnisotropic>;

    template<typename C=bool_constant<!IsAnisotropic> >
    static enable_if_t<C::value && !IsAnisotropic, this_t> create(NBL_CONST_REF_ARG(creation_params_type) params)
    {
        this_t retval;
        retval.ndf = ndf_type::create(params.A);
        retval.fresnel = params.fresnel;
        return retval;
    }
    template<typename C=bool_constant<IsAnisotropic> >
    static enable_if_t<C::value && IsAnisotropic, this_t> create(NBL_CONST_REF_ARG(creation_params_type) params)
    {
        this_t retval;
        retval.ndf = ndf_type::create(params.ax, params.ay);
        retval.fresnel = params.fresnel;
        return retval;
    }

    template<class Interaction, class MicrofacetCache>
    spectral_type __eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        fresnel_type _f = impl::getOrientedFresnel<fresnel_type, IsBSDF>::__call(fresnel, interaction.getNdotV());
        const bool notTIR = impl::check_TIR_helper<fresnel_type, IsBSDF>::template __call<MicrofacetCache>(_f, cache);
        if ((IsBSDF && notTIR) || (!IsBSDF && _sample.getNdotL() > numeric_limits<scalar_type>::min && interaction.getNdotV() > numeric_limits<scalar_type>::min))
        {
            using quant_query_type = typename ndf_type::quant_query_type;
            using g2g1_query_type = typename ndf_type::g2g1_query_type;

            scalar_type dummy;
            quant_query_type qq = ndf.template createQuantQuery<MicrofacetCache>(cache, dummy);

            quant_type D = ndf.template D<sample_type, Interaction, MicrofacetCache>(qq, _sample, interaction, cache);
            scalar_type DG = D.projectedLightMeasure;
            if (D.microfacetMeasure < bit_cast<scalar_type>(numeric_limits<scalar_type>::infinity))
            {
                g2g1_query_type gq = ndf.template createG2G1Query<sample_type, Interaction>(_sample, interaction);
                DG *= ndf.template correlated<sample_type, Interaction>(gq, _sample, interaction);
            }
            NBL_IF_CONSTEXPR(IsBSDF)
                return impl::__implicit_promote<spectral_type, typename fresnel_type::vector_type>::__call(_f(hlsl::abs(cache.getVdotH()))) * DG;
            else
                return impl::__implicit_promote<spectral_type, typename fresnel_type::vector_type>::__call(_f(cache.getVdotH())) * DG;
        }
        else
            return hlsl::promote<spectral_type>(0.0);
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
        ray_dir_info_type localV_raydir = interaction.getV().transform(interaction.getToTangentSpace());
        const vector3_type localV = interaction.getTangentSpaceV();
        const vector3_type localH = ndf.generateH(localV, u);

        cache = anisocache_type::createForReflection(localV, localH);
        struct reflect_wrapper
        {
            vector3_type operator()() NBL_CONST_MEMBER_FUNC
            {
                return r(VdotH);
            }
            bxdf::Reflect<scalar_type> r;
            scalar_type VdotH;
        };
        reflect_wrapper r;
        r.r = bxdf::Reflect<scalar_type>::create(localV, localH);
        r.VdotH = cache.getVdotH();
        ray_dir_info_type localL = localV_raydir.template reflect<reflect_wrapper>(r);

        // fail if samples have invalid paths
        if (localL.getDirection().z < scalar_type(0.0))  // NdotL<0
            localL.makeInvalid(); // should check if sample direction is invalid

        return sample_type::createFromTangentSpace(localL, interaction.getFromTangentSpace());
    }
    template<typename C=bool_constant<IsBSDF> >
    enable_if_t<C::value && IsBSDF, sample_type> generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, const vector3_type u, NBL_REF_ARG(anisocache_type) cache)
    {
        fresnel_type _f = impl::getOrientedFresnel<fresnel_type, IsBSDF>::__call(fresnel, interaction.getNdotV());
        fresnel::OrientedEtaRcps<monochrome_type> rcpEta = _f.getOrientedEtaRcps();

        ray_dir_info_type V = interaction.getV();
        const vector3_type localV = interaction.getTangentSpaceV();
        const vector3_type upperHemisphereV = ieee754::flipSignIfRHSNegative<vector3_type>(localV, hlsl::promote<vector3_type>(interaction.getNdotV()));
        const vector3_type localH = ndf.generateH(upperHemisphereV, u.xy);
        const vector3_type H = hlsl::mul(interaction.getFromTangentSpace(), localH);

        const scalar_type VdotH = hlsl::dot(V.getDirection(), H);
        const scalar_type reflectance = _f(hlsl::abs(VdotH))[0];

        scalar_type rcpChoiceProb;
        scalar_type z = u.z;
        bool transmitted = math::partitionRandVariable(reflectance, z, rcpChoiceProb);

        Refract<scalar_type> r = Refract<scalar_type>::create(V.getDirection(), H);
        bxdf::ReflectRefract<scalar_type> rr;
        rr.refract = r;
        ray_dir_info_type L = V.reflectRefract(rr, transmitted, rcpEta.value[0]);

        const vector3_type T = interaction.getT();
        const vector3_type B = interaction.getB();
        const vector3_type _N = interaction.getN();

        // fail if samples have invalid paths
        const vector3_type Ldir = L.getDirection();
        const scalar_type LdotH = hlsl::dot(Ldir, H);
        if ((ComputeMicrofacetNormal<scalar_type>::isTransmissionPath(VdotH, LdotH) != transmitted) || (LdotH * hlsl::dot(_N, Ldir) < scalar_type(0.0)))
            L.makeInvalid(); // should check if sample direction is invalid
        else
            cache = anisocache_type::create(VdotH, Ldir, H, T, B, _N, transmitted);

        return sample_type::create(L, T, B, _N);
    }
    template<typename C=bool_constant<!IsAnisotropic>, typename D=bool_constant<!IsBSDF> >
    enable_if_t<C::value && !IsAnisotropic && D::value && !IsBSDF, sample_type> generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, const vector2_type u, NBL_REF_ARG(isocache_type) cache)
    {
        anisocache_type aniso_cache;
        sample_type s = generate(anisotropic_interaction_type::create(interaction), u, aniso_cache);
        cache = aniso_cache.iso_cache;
        return s;
    }
    template<typename C=bool_constant<!IsAnisotropic>, typename D=bool_constant<IsBSDF> >
    enable_if_t<C::value && !IsAnisotropic && D::value && IsBSDF, sample_type> generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, const vector3_type u, NBL_REF_ARG(isocache_type) cache)
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
        if (IsBSDF || (_sample.getNdotL() > numeric_limits<scalar_type>::min && interaction.getNdotV() > numeric_limits<scalar_type>::min))
        {
            scalar_type _pdf = __pdf<isotropic_interaction_type, isocache_type>(_sample, interaction, cache);
            return hlsl::mix(scalar_type(0.0), _pdf, _pdf < bit_cast<scalar_type>(numeric_limits<scalar_type>::infinity));
        }
        else
            return scalar_type(0.0);
    }
    scalar_type pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        if (IsBSDF || (_sample.getNdotL() > numeric_limits<scalar_type>::min && interaction.getNdotV() > numeric_limits<scalar_type>::min))
        {
            scalar_type _pdf = __pdf<anisotropic_interaction_type, anisocache_type>(_sample, interaction, cache);
            return hlsl::mix(scalar_type(0.0), _pdf, _pdf < bit_cast<scalar_type>(numeric_limits<scalar_type>::infinity));
        }
        else
            return scalar_type(0.0);
    }

    template<class Interaction, class MicrofacetCache>
    quotient_pdf_type __quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        scalar_type _pdf = __pdf<Interaction, MicrofacetCache>(_sample, interaction, cache);
        fresnel_type _f = impl::getOrientedFresnel<fresnel_type, IsBSDF>::__call(fresnel, interaction.getNdotV());

        spectral_type quo = hlsl::promote<spectral_type>(0.0);
        const bool notTIR = impl::check_TIR_helper<fresnel_type, IsBSDF>::template __call<MicrofacetCache>(_f, cache);
        if (notTIR)
        {
            using g2g1_query_type = typename N::g2g1_query_type;
            g2g1_query_type gq = ndf.template createG2G1Query<sample_type, Interaction>(_sample, interaction);
            scalar_type G2_over_G1 = ndf.template G2_over_G1<sample_type, Interaction, MicrofacetCache>(gq, _sample, interaction, cache);
            NBL_IF_CONSTEXPR(IsBSDF)
                quo = hlsl::promote<spectral_type>(G2_over_G1);
            else
                quo = _f(cache.getVdotH()) * G2_over_G1;
        }

        // set pdf=0 when quo=0 because we don't want to give high weight to sampling strategy that yields 0 contribution
        _pdf = hlsl::mix(_pdf, scalar_type(0.0), hlsl::all(quo < hlsl::promote<spectral_type>(numeric_limits<scalar_type>::min)));
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
