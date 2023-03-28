
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_BXDF_NDF_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_NDF_INCLUDED_

#include <nbl/builtin/hlsl/bxdf/common.hlsl>

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace ndf
{

// general path
float microfacet_to_light_measure_transform(in float NDFcos, in float absNdotV, in bool transmitted, in float VdotH, in float LdotH, in float VdotHLdotH, in float orientedEta)
{
    float denominator = absNdotV;
    if (transmitted)
    {
        const float VdotH_etaLdotH = (VdotH+orientedEta*LdotH);
        // VdotHLdotH is negative under transmission, so thats denominator is negative
        denominator *= -VdotH_etaLdotH*VdotH_etaLdotH;
    }
    return NDFcos*(transmitted ? VdotHLdotH:0.25)/denominator;
}
float microfacet_to_light_measure_transform(in float NDFcos, in float maxNdotV)
{
    return 0.25*NDFcos/maxNdotV;
}


namespace ggx
{

// specialized factorizations for GGX
float microfacet_to_light_measure_transform(in float NDFcos_already_in_reflective_dL_measure, in float absNdotL, in bool transmitted, in float VdotH, in float LdotH, in float VdotHLdotH, in float orientedEta)
{
    float factor = absNdotL;
    if (transmitted)
    {
        const float VdotH_etaLdotH = (VdotH+orientedEta*LdotH);
        // VdotHLdotH is negative under transmission, so thats denominator is negative
        factor *= -4.0*VdotHLdotH/(VdotH_etaLdotH*VdotH_etaLdotH);
    }
    return NDFcos_already_in_reflective_dL_measure*factor;
}
float microfacet_to_light_measure_transform(in float NDFcos_already_in_reflective_dL_measure, in float maxNdotL)
{
    return NDFcos_already_in_reflective_dL_measure*maxNdotL;
}

}


// Utility class
template <class Scalar = float>
struct NDFBase
{
    // NDFs must define such typenames:
    using scalar_t = Scalar;

    /**
    * NDFs must define such member functions:
    *
    * // Note that generation is always anisotropic,
    * // hence both interaction and microfacet cache must be anisotropic ones.
    * template <class IncomingrayDirInfo>
    * float3 generateH(in surface_interactions::Anisotropic<IncomingrayDirInfo>, inout float3 u, out AnisotropicMicrofacetCache);
    *
    * // isotropic NDF evaluators:
    * scalar_t D(in float NdotH2);
    * scalar_t Lambda(in float NdotX2);
    *
    * // anisotropic NDF evaluators:
    * scalar_t D(in float TdotH2, in float BdotH2, in float NdotH2);
    * scalar_t Lambda(in float TdotX2, in float BdotX2, in float NdotX2);
    */
};


// forward declaration so we can explicitly specialize, e.g. for GGX where optimized forms of the functions provided by the trait exist
template<class ndf_t>
struct ndf_traits;

namespace impl
{
    template<class ndf_t>
    struct ndf_traits
    {
        using scalar_t = typename ndf_t::scalar_t;

        scalar_t G1(in float NdotX2) { return scalar_t(1) / (scalar_t(1) + ndf.Lambda(NdotX2)); }
        scalar_t G2(in float NdotV2, in float NdotL2) { return scalar_t(1) / (scalar_t(1) + ndf.Lambda(NdotV2) + ndf.Lambda(NdotL2)); }

        scalar_t G2_over_G1(in float NdotV2, in float NdotL2) 
        {
            const scalar_t lambdaV_plus_one = ndf.Lambda(NdotV2) + scalar_t(1);
            return lambdaV_plus_one / (ndf.Lambda(NdotL2) + lambdaV_plus_one);
        }

        //
        // dHdL functions
        //
        // For BRDFs only:
        scalar_t dHdL(in float NDFcos, in float maxNdotV)
        {
            return microfacet_to_light_measure_transform(NDFcos, maxNdotV);
        }
        // For all BxDFs:
        scalar_t dHdL(in float NDFcos, in float absNdotV, in bool transmitted, in float VdotH, in float LdotH, in float VdotHLdotH, in float orientedEta)
        {
            return microfacet_to_light_measure_transform(NDFcos, absNdotV, transmitted, VdotH, LdotH, VdotHLdotH, orientedEta);
        }

        //
        // VNDF functions
        //
        // Statics:
        static scalar_t VNDF_static(in scalar_t d, in scalar_t lambda_V, in float maxNdotV, out scalar_t onePlusLambda_V)
        {
            onePlusLambda_V = scalar_t(1) + lambda_V;

            return microfacet_to_light_measure_transform(d / onePlusLambda_V, maxNdotV);
        }
        static scalar_t VNDF_static(in scalar_t d, in scalar_t lambda_V, in float absNdotV, in bool transmitted, in float VdotH, in float LdotH, in float VdotHLdotH, in float orientedEta, in float reflectance, out float onePlusLambda_V)
        {
            onePlusLambda_V = scalar_t(1) + lambda_V;

            return microfacet_to_light_measure_transform((transmitted ? (1.0 - reflectance) : reflectance) * d / onePlusLambda_V, absNdotV, transmitted, VdotH, LdotH, VdotHLdotH, orientedEta);
        }
        static scalar_t VNDF_static(in scalar_t d, in scalar_t G1_over_2NdotV)
        {
            return d * 0.5 * G1_over_2NdotV;
        }

        static scalar_t VNDF_fromLambda_impl(in scalar_t d, in scalar_t lambda, in float maxNdotV)
        {
            scalar_t dummy;
            return VNDF_static(d, lambda, maxNdotV, dummy);
        }
        static scalar_t VNDF_fromG1_over_2NdotV_impl(in scalar_t d, in scalar_t G1_over_2NdotV)
        {
            return VNDF_static(d, G1_over_2NdotV);
        }

        

        // VNDF isotropic variants
        scalar_t VNDF(in float NdotH2, in float NdotV2, in float maxNdotV)
        {
            const float d = ndf.D(NdotH2);
            const float lambda = ndf.Lambda(NdotV2);
            return VNDF_fromLambda_impl(d, lambda, maxNdotV);
        }
        scalar_t VNDF(in float NdotH2, in float G1_over_2NdotV)
        {
            const float d = ndf.D(NdotH2);
            return VNDF_fromG1_over_2NdotV_impl(d, G1_over_2NdotV);
        }
        scalar_t VNDF(
            in float NdotH2, in float NdotV2, 
            in float absNdotV, in bool transmitted, in float VdotH, in float LdotH, in float VdotHLdotH, in float orientedEta, in float reflectance, out float onePlusLambda_V)
        {
            const float d = ndf.D(NdotH2);
            const float lambda = ndf.Lambda(NdotV2);

            return VNDF_static(d, lambda, absNdotV, transmitted, VdotH, LdotH, VdotHLdotH, orientedEta, reflectance, onePlusLambda_V);
        }

        // VNDF anisotropic variants
        scalar_t VNDF(
            in float TdotH2, in float BdotH2, in float NdotH2,
            in float TdotV2, in float BdotV2, in float NdotV2,
            in float maxNdotV)
        {
            const float d = ndf.D(TdotH2, BdotH2, NdotH2);
            const float lambda = ndf.Lambda(TdotV2, BdotV2, NdotV2);
            return VNDF_fromLambda_impl(d, lambda, maxNdotV);
        }
        scalar_t VNDF(
            in float TdotH2, in float BdotH2, in float NdotH2, 
            in float G1_over_2NdotV)
        {
            const float d = ndf.D(TdotH2, BdotH2, NdotH2);
            return VNDF_fromG1_over_2NdotV_impl(d, G1_over_2NdotV);
        }
        scalar_t VNDF(
            in float TdotH2, in float BdotH2, in float NdotH2,
            in float TdotV2, in float BdotV2, in float NdotV2,
            in float absNdotV, in bool transmitted, in float VdotH, in float LdotH, in float VdotHLdotH, in float orientedEta, in float reflectance, out float onePlusLambda_V)
        {
            const float d = ndf.D(TdotH2, BdotH2, NdotH2);
            const float lambda = ndf.Lambda(TdotV2, BdotV2, NdotV2);

            return VNDF_static(d, lambda, absNdotV, transmitted, VdotH, LdotH, VdotHLdotH, orientedEta, reflectance, onePlusLambda_V);
        }

        ndf_t ndf;
    };
}

// default specialization
template<class ndf_t>
struct ndf_traits : impl::ndf_traits<ndf_t> {};

}
}
}
}

#include <nbl/builtin/hlsl/bxdf/ndf/beckmann.hlsl>

#endif