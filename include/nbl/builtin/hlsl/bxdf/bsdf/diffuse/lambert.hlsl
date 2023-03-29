// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_BXDF_BSDF_DIFFUSE_LAMBERT_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_BSDF_DIFFUSE_LAMBERT_INCLUDED_

#include <nbl/builtin/hlsl/bxdf/common.hlsl>
#include <nbl/builtin/hlsl/sampling/cos_weighted.hlsl>

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace bsdf
{
namespace diffuse
{

float lambertian()
{
    return math::RECIPROCAL_PI * 0.5;
}

float lambertian_cos_eval_rec_2pi_factored_out_wo_clamps(in float absNdotL)
{
    return absNdotL;
}
float lambertian_cos_eval_rec_2pi_factored_out(in float NdotL)
{
    return lambertian_cos_eval_rec_2pi_factored_out_wo_clamps(abs(NdotL));
}

float lambertian_cos_eval_wo_clamps(in float absNdotL)
{
    return lambertian_cos_eval_rec_2pi_factored_out_wo_clamps(absNdotL) * lambertian();
}
template <class IncomingRayDirInfo>
float lambertian_cos_eval(in LightSample<IncomingRayDirInfo> _sample)
{
    return lambertian_cos_eval_rec_2pi_factored_out(_sample.NdotL) * lambertian();
}

template <class IncomingRayDirInfo>
LightSample<IncomingRayDirInfo> lambertian_cos_generate_wo_clamps(in float3 tangentSpaceV, in float3x3 m, inout float3 u)
{
    float3 L = sampling::projected_sphere_generate(u);

    return LightSample<IncomingRayDirInfo>::createTangentSpace(tangentSpaceV, IncomingRayDirInfo::create(L), m);
}
template <class IncomingRayDirInfo>
LightSample<IncomingRayDirInfo> lambertian_cos_generate(in surface_interactions::Anisotropic<IncomingRayDirInfo> interaction, inout float3 u)
{
    return lambertian_cos_generate_wo_clamps(interaction.getTangentSpaceV(), interaction.getTangentFrame(), u);
}

template <class IncomingRayDirInfo>
quotient_and_pdf_scalar lambertian_cos_quotient_and_pdf_wo_clamps(in float absNdotL)
{
    float pdf;
    float q = sampling::projected_sphere_quotient_and_pdf(pdf, absNdotL);
    return quotient_and_pdf_scalar::create(q, pdf);
}
template <class IncomingRayDirInfo>
quotient_and_pdf_scalar lambertian_cos_quotient_and_pdf(in LightSample<IncomingRayDirInfo> s)
{
    float pdf;
    float q = lambertian_cos_quotient_and_pdf_wo_clamps(pdf, abs(s.NdotL));
    return quotient_and_pdf_scalar::create(q, pdf);
}

float lambertian_pdf_wo_clamps(in float absNdotL)
{
    return sampling::projected_sphere_pdf(absNdotL);
}


template <class IncomingRayDirInfo>
struct Lambertian : BxDFBase<float, float, LightSample<IncomingRayDirInfo>, surface_interactions::Isotropic<IncomingRayDirInfo> >
{
    using base_t = BxDFBase<float, float, LightSample<IncomingRayDirInfo>, surface_interactions::Isotropic<IncomingRayDirInfo> >;

    static Lambertian create()
    {
        Lambertian lambertian;
        return lambertian;
    }

    typename base_t::spectrum_t      cos_eval(in typename base_t::sample_t s, in typename base_t::interaction_t interaction)
    {
        return 0.5f * math::RECIPROCAL_PI * max(s.NdotL, 0.0f);
    }

    static
    typename base_t::sample_t        generate(in surface_interactions::Anisotropic<IncomingRayDirInfo> interaction, inout float3 u)
    {
        float3 L = sampling::projected_sphere_generate(u);

        const float3 tangentSpaceV = interaction.getTangentSpaceV();
        const float3x3 tangentFrame = interaction.getTangentFrame();

        return LightSample<IncomingRayDirInfo>::createTangentSpace(tangentSpaceV, IncomingRayDirInfo::create(L), tangentFrame);
    }

    typename base_t::q_pdf_t cos_quotient_and_pdf(in typename base_t::sample_t s, in typename base_t::interaction_t interaction)
    {
        float pdf;
        float q = sampling::projected_sphere_quotient_and_pdf(pdf, abs(s.NdotL));

        return quotient_and_pdf_scalar::create(q, pdf);
    }
};

}
}
}
}
}

#endif
