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
float lambertian_cos_remainder_and_pdf_wo_clamps(out float pdf, in float absNdotL)
{
    return sampling::projected_sphere_remainder_and_pdf(pdf, absNdotL);
}
template <class IncomingRayDirInfo>
float lambertian_cos_remainder_and_pdf(out float pdf, in LightSample<IncomingRayDirInfo> s)
{
    return lambertian_cos_remainder_and_pdf_wo_clamps(pdf, abs(s.NdotL));
}

float lambertian_pdf_wo_clamps(in float absNdotL)
{
    return sampling::projected_sphere_pdf(absNdotL);
}

}
}
}
}
}

#endif
