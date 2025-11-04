// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_NDF_BECKMANN_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_NDF_BECKMANN_INCLUDED_

#include "nbl/builtin/hlsl/limits.hlsl"
#include "nbl/builtin/hlsl/bxdf/ndf/microfacet_to_light_transform.hlsl"
#include "nbl/builtin/hlsl/bxdf/ndf.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace ndf
{

namespace beckmann_concepts
{
#define NBL_CONCEPT_NAME DG1Query
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (query, T)
NBL_CONCEPT_BEGIN(1)
#define query NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((query.getNdf()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((query.getLambdaV()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
);
#undef query
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

#define NBL_CONCEPT_NAME G2overG1Query
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (query, T)
NBL_CONCEPT_BEGIN(1)
#define query NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((query.getLambdaL()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((query.getLambdaV()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
);
#undef query
#include <nbl/builtin/hlsl/concepts/__end.hlsl>
}

namespace impl
{

template<typename T>
struct SBeckmannDG1Query
{
    using scalar_type = T;

    scalar_type getNdf() NBL_CONST_MEMBER_FUNC { return ndf; }
    scalar_type getLambdaV() NBL_CONST_MEMBER_FUNC { return lambda_V; }

    scalar_type ndf;
    scalar_type lambda_V;
};

template<typename T>
struct SBeckmannG2overG1Query
{
    using scalar_type = T;

    scalar_type getLambdaL() NBL_CONST_MEMBER_FUNC { return lambda_L; }
    scalar_type getLambdaV() NBL_CONST_MEMBER_FUNC { return lambda_V; }

    scalar_type lambda_L;
    scalar_type lambda_V;
};

template<typename T>
struct BeckmannBase
{
    //conversion between alpha and Phong exponent, Walter et.al.
    static T PhongExponentToAlpha2(T _n)
    {
        return T(2.0) / (_n + T(2.0));
    }
    //+INF for a2==0.0
    static T Alpha2ToPhongExponent(T a2)
    {
        return T(2.0) / a2 - T(2.0);
    }
};

template<typename T, bool IsAnisotropic=false NBL_STRUCT_CONSTRAINABLE>
struct BeckmannCommon;

template<typename T>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<T>)
struct BeckmannCommon<T,false NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<T>) > : BeckmannBase<T>
{
    using scalar_type = T;

    template<class MicrofacetCache NBL_FUNC_REQUIRES(ReadableIsotropicMicrofacetCache<MicrofacetCache>)
    scalar_type D(NBL_CONST_REF_ARG(MicrofacetCache) cache, NBL_REF_ARG(bool) isInfinity)
    {
        if (a2 < numeric_limits<scalar_type>::min)
        {
            isInfinity = true;
            return bit_cast<scalar_type>(numeric_limits<scalar_type>::infinity);
        }
        scalar_type NdotH2 = cache.getNdotH2();
        scalar_type nom = exp2<scalar_type>((NdotH2 - scalar_type(1.0)) / (log<scalar_type>(2.0) * a2 * NdotH2));
        scalar_type denom = a2 * NdotH2 * NdotH2;
        scalar_type ndf = numbers::inv_pi<scalar_type> * nom / denom;
        isInfinity = hlsl::isinf(ndf);
        return ndf;
    }

    scalar_type C2(scalar_type NdotX2)
    {
        assert(NdotX2 >= scalar_type(0.0));
        return NdotX2 / (a2 * (scalar_type(1.0) - NdotX2));
    }

    scalar_type a2;
};

template<typename T>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<T>)
struct BeckmannCommon<T,true NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<T>) > : BeckmannBase<T>
{
    using scalar_type = T;

    template<class MicrofacetCache NBL_FUNC_REQUIRES(AnisotropicMicrofacetCache<MicrofacetCache>)
    scalar_type D(NBL_CONST_REF_ARG(MicrofacetCache) cache, NBL_REF_ARG(bool) isInfinity)
    {
        if (a2 < numeric_limits<scalar_type>::min)
        {
            isInfinity = true;
            return bit_cast<scalar_type>(numeric_limits<scalar_type>::infinity);
        }
        isInfinity = false;
        scalar_type NdotH2 = cache.getNdotH2();
        scalar_type nom = exp<scalar_type>(-(cache.getTdotH2() / ax2 + cache.getBdotH2() / ay2) / NdotH2);
        scalar_type denom = a2 * NdotH2 * NdotH2;
        scalar_type ndf = numbers::inv_pi<scalar_type> * nom / denom;
        isInfinity = hlsl::isinf(ndf);
        return ndf;
    }

    scalar_type C2(scalar_type TdotX2, scalar_type BdotX2, scalar_type NdotX2)
    {
        return NdotX2 / (TdotX2 * ax2 + BdotX2 * ay2);
    }

    scalar_type ax2;
    scalar_type ay2;
    scalar_type a2;
};

template<typename T>
struct BeckmannGenerateH
{
    using scalar_type = T;
    using vector2_type = vector<T, 2>;
    using vector3_type = vector<T, 3>;

    vector3_type __call(const vector3_type localV, const vector2_type u)
    {
        //stretch
        vector3_type V = nbl::hlsl::normalize<vector3_type>(vector3_type(ax * localV.x, ay * localV.y, localV.z));

        vector2_type slope;
        if (V.z > 0.9999)   // V.z=NdotV=cosTheta in tangent space
        {
            scalar_type r = hlsl::sqrt(-hlsl::log(1.0 - u.x));
            scalar_type sinPhi = hlsl::sin(2.0 * numbers::pi<scalar_type> * u.y);
            scalar_type cosPhi = hlsl::cos(2.0 * numbers::pi<scalar_type> * u.y);
            slope = hlsl::promote<vector2_type>(r) * vector2_type(cosPhi,sinPhi);
        }
        else
        {
            scalar_type cosTheta = V.z;
            scalar_type sinTheta = hlsl::sqrt(1.0 - cosTheta * cosTheta);
            scalar_type tanTheta = sinTheta / cosTheta;
            scalar_type cotTheta = 1.0 / tanTheta;

            scalar_type a = -1.0;
            scalar_type c = hlsl::erf(cotTheta);
            scalar_type sample_x = hlsl::max(u.x, scalar_type(1e-6));
            scalar_type theta = hlsl::acos(cosTheta);
            scalar_type fit = 1.0 + theta * (-0.876 + theta * (0.4265 - 0.0594*theta));
            scalar_type b = c - (1.0 + c) * hlsl::pow(scalar_type(1.0)-sample_x, fit);

            scalar_type normalization = 1.0 / (1.0 + c + numbers::inv_sqrtpi<scalar_type> * tanTheta * hlsl::exp(-cotTheta*cotTheta));

            const uint32_t ITER_THRESHOLD = 10;
            const scalar_type MAX_ACCEPTABLE_ERR = 1e-5;
            uint32_t it = 0;
            scalar_type value = 1000.0;
            while (++it < ITER_THRESHOLD && hlsl::abs(value) > MAX_ACCEPTABLE_ERR)
            {
                if (!(b >= a && b <= c))
                    b = 0.5 * (a + c);

                scalar_type invErf = hlsl::erfInv(b);
                value = normalization * (1.0 + b + numbers::inv_sqrtpi<scalar_type> * tanTheta * hlsl::exp(-invErf * invErf)) - sample_x;
                scalar_type derivative = normalization * (1.0 - invErf * cosTheta);

                if (value > 0.0)
                    c = b;
                else
                    a = b;

                b -= value/derivative;
            }
            // TODO: investigate if we can replace these two erf^-1 calls with a box muller transform
            slope.x = hlsl::erfInv(b);
            slope.y = hlsl::erfInv(2.0 * hlsl::max(u.y, scalar_type(1e-6)) - 1.0);
        }

        scalar_type sinTheta = hlsl::sqrt(1.0 - V.z*V.z);
        scalar_type cosPhi = sinTheta==0.0 ? 1.0 : hlsl::clamp<scalar_type>(V.x/sinTheta, -1.0, 1.0);
        scalar_type sinPhi = sinTheta==0.0 ? 0.0 : hlsl::clamp<scalar_type>(V.y/sinTheta, -1.0, 1.0);
        //rotate
        scalar_type tmp = cosPhi*slope.x - sinPhi*slope.y;
        slope.y = sinPhi*slope.x + cosPhi*slope.y;
        slope.x = tmp;

        //unstretch
        slope = vector2_type(ax, ay) * slope;

        return nbl::hlsl::normalize<vector3_type>(vector3_type(-slope, 1.0));
    }

    scalar_type ax;
    scalar_type ay;
};
}


template<typename T, bool _IsAnisotropic, MicrofacetTransformTypes reflect_refract NBL_PRIMARY_REQUIRES(concepts::FloatingPointScalar<T>)
struct Beckmann
{
    NBL_HLSL_NDF_CONSTEXPR_DECLS(_IsAnisotropic,reflect_refract);
    NBL_HLSL_NDF_TYPE_ALIASES(((Beckmann<T,IsAnisotropic,SupportedPaths>))((impl::BeckmannCommon<T,IsAnisotropic>))((impl::SBeckmannDG1Query<scalar_type>))((impl::SBeckmannG2overG1Query<scalar_type>))((DualMeasureQuantQuery<scalar_type>)));
    NBL_CONSTEXPR_STATIC_INLINE bool GuaranteedVNDF = false;

    template<typename C=bool_constant<!IsAnisotropic> >
    static enable_if_t<C::value && !IsAnisotropic, this_t> create(scalar_type A)
    {
        this_t retval;
        retval.__ndf_base.a2 = A*A;
        retval.__generate_base.ax = A;
        retval.__generate_base.ay = A;
        return retval;
    }
    template<typename C=bool_constant<IsAnisotropic> >
    static enable_if_t<C::value && IsAnisotropic, this_t> create(scalar_type ax, scalar_type ay)
    {
        this_t retval;
        retval.__ndf_base.ax2 = ax*ax;
        retval.__ndf_base.ay2 = ay*ay;
        retval.__ndf_base.a2 = ax*ay;
        retval.__generate_base.ax = ax;
        retval.__generate_base.ay = ay;
        return retval;
    }

    static scalar_type Lambda(scalar_type c2)
    {
        scalar_type c = sqrt<scalar_type>(c2);
        scalar_type nom = scalar_type(1.0) - scalar_type(1.259) * c + scalar_type(0.396) * c2;
        scalar_type denom = scalar_type(2.181) * c2 + scalar_type(3.535) * c;
        return hlsl::mix<scalar_type>(scalar_type(0.0), nom / denom, c < scalar_type(1.6));
    }

    template<class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(RequiredInteraction<Interaction> && RequiredMicrofacetCache<MicrofacetCache>)
    quant_query_type createQuantQuery(NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache, scalar_type orientedEta)
    {
        quant_query_type quant_query;   // only has members for refraction
        NBL_IF_CONSTEXPR(SupportsTransmission)
        {
            quant_query = quant_query_type::template create<Interaction, MicrofacetCache>(interaction, cache, orientedEta);
        }
        return quant_query;
    }
    template<class Interaction, class MicrofacetCache, typename C=bool_constant<!IsAnisotropic> NBL_FUNC_REQUIRES(RequiredInteraction<Interaction> && RequiredMicrofacetCache<MicrofacetCache>)
    enable_if_t<C::value && !IsAnisotropic, dg1_query_type> createDG1Query(NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        dg1_query_type dg1_query;
        bool dummy;
        dg1_query.ndf = __ndf_base.template D<MicrofacetCache>(cache, dummy);
        dg1_query.lambda_V = Lambda(__ndf_base.C2(interaction.getNdotV2()));
        return dg1_query;
    }
    template<class LS, class Interaction, typename C=bool_constant<!IsAnisotropic> NBL_FUNC_REQUIRES(LightSample<LS> && RequiredInteraction<Interaction>)
    enable_if_t<C::value && !IsAnisotropic, g2g1_query_type> createG2G1Query(NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        g2g1_query_type g2_query;
        g2_query.lambda_L = Lambda(__ndf_base.C2(_sample.getNdotL2()));
        g2_query.lambda_V = Lambda(__ndf_base.C2(interaction.getNdotV2()));
        return g2_query;
    }
    template<class Interaction, class MicrofacetCache, typename C=bool_constant<IsAnisotropic> NBL_FUNC_REQUIRES(RequiredInteraction<Interaction> && RequiredMicrofacetCache<MicrofacetCache>)
    enable_if_t<C::value && IsAnisotropic, dg1_query_type> createDG1Query(NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        dg1_query_type dg1_query;
        bool dummy;
        dg1_query.ndf = __ndf_base.template D<MicrofacetCache>(cache, dummy);
        dg1_query.lambda_V = Lambda(__ndf_base.C2(interaction.getTdotV2(), interaction.getBdotV2(), interaction.getNdotV2()));
        return dg1_query;
    }
    template<class LS, class Interaction, typename C=bool_constant<IsAnisotropic> NBL_FUNC_REQUIRES(LightSample<LS> && RequiredInteraction<Interaction>)
    enable_if_t<C::value && IsAnisotropic, g2g1_query_type> createG2G1Query(NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction)
    {
        g2g1_query_type g2_query;
        g2_query.lambda_L = Lambda(__ndf_base.C2(_sample.getTdotL2(), _sample.getBdotL2(), _sample.getNdotL2()));
        g2_query.lambda_V = Lambda(__ndf_base.C2(interaction.getTdotV2(), interaction.getBdotV2(), interaction.getNdotV2()));
        return g2_query;
    }

    vector3_type generateH(const vector3_type localV, const vector2_type u)
    {
        return __generate_base.__call(localV, u);
    }

    template<class LS, class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(LightSample<LS> && RequiredInteraction<Interaction> && RequiredMicrofacetCache<MicrofacetCache>)
    quant_type D(NBL_CONST_REF_ARG(quant_query_type) quant_query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache, NBL_REF_ARG(bool) isInfinity)
    {
        scalar_type d = __ndf_base.template D<MicrofacetCache>(cache, isInfinity);
        return createDualMeasureQuantity<T, reflect_refract, quant_query_type>(d, interaction.getNdotV(BxDFClampMode::BCM_ABS), _sample.getNdotL(BxDFClampMode::BCM_ABS), quant_query);
    }

    template<class LS, class Interaction NBL_FUNC_REQUIRES(LightSample<LS> && RequiredInteraction<Interaction>)
    quant_type DG1(NBL_CONST_REF_ARG(dg1_query_type) query, NBL_CONST_REF_ARG(quant_query_type) quant_query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_REF_ARG(bool) isInfinity)
    {
        scalar_type D = query.getNdf();
        isInfinity = hlsl::isinf(D);
        if (isInfinity)
        {
            quant_type dmq;
            return dmq;
        }
        scalar_type dg1 = D / (scalar_type(1.0) + query.getLambdaV());
        return createDualMeasureQuantity<T, reflect_refract, quant_query_type>(dg1, interaction.getNdotV(BxDFClampMode::BCM_ABS), _sample.getNdotL(BxDFClampMode::BCM_ABS), quant_query);
    }

    template<class LS, class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(LightSample<LS> && RequiredInteraction<Interaction> && RequiredMicrofacetCache<MicrofacetCache>)
    scalar_type correlated(NBL_CONST_REF_ARG(g2g1_query_type) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        scalar_type onePlusLambda_V = scalar_type(1.0) + query.getLambdaV();
        scalar_type lambda_L = query.getLambdaL();
        return hlsl::mix(scalar_type(1.0)/(onePlusLambda_V + lambda_L), bxdf::beta<scalar_type>(onePlusLambda_V, scalar_type(1.0) + lambda_L), cache.isTransmission());
    }

    template<class LS, class Interaction, class MicrofacetCache NBL_FUNC_REQUIRES(LightSample<LS> && RequiredInteraction<Interaction> && RequiredMicrofacetCache<MicrofacetCache>)
    scalar_type G2_over_G1(NBL_CONST_REF_ARG(g2g1_query_type) query, NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(Interaction) interaction, NBL_CONST_REF_ARG(MicrofacetCache) cache)
    {
        scalar_type onePlusLambda_V = scalar_type(1.0) + query.getLambdaV();
        scalar_type lambda_L = query.getLambdaL();
        return onePlusLambda_V * hlsl::mix(scalar_type(1.0)/(onePlusLambda_V + lambda_L), bxdf::beta<scalar_type>(onePlusLambda_V, scalar_type(1.0) + lambda_L), cache.isTransmission());
    }

    base_type __ndf_base;
    impl::BeckmannGenerateH<scalar_type> __generate_base;
};

}
}
}
}

#endif
