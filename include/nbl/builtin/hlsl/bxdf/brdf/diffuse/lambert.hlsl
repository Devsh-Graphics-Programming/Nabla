// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_BRDF_DIFFUSE_LAMBERT_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_BRDF_DIFFUSE_LAMBERT_INCLUDED_

#include <nbl/builtin/hlsl/bxdf/common.hlsl>
#include <nbl/builtin/hlsl/sampling/cos_weighted.hlsl>

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace brdf
{
namespace diffuse
{

float lambertian()
{
    return math::RECIPROCAL_PI;
}

float lambertian_cos_eval_rec_pi_factored_out_wo_clamps(in float maxNdotL)
{
   return maxNdotL;
}
float lambertian_cos_eval_rec_pi_factored_out(in float NdotL)
{
   return lambertian_cos_eval_rec_pi_factored_out_wo_clamps(max(NdotL,0.0f));
}

float lambertian_cos_eval_wo_clamps(in float maxNdotL)
{
   return lambertian_cos_eval_rec_pi_factored_out_wo_clamps(maxNdotL)*lambertian();
}
template <class IncomingRayDirInfo>
float lambertian_cos_eval(in LightSample<IncomingRayDirInfo> _sample)
{
    return lambertian_cos_eval_rec_pi_factored_out(_sample.NdotL)*lambertian();
}

template <class IncomingRayDirInfo>
LightSample<IncomingRayDirInfo> lambertian_cos_generate_wo_clamps(in float3 tangentSpaceV, in float3x3 m, in float2 u)
{
    float3 L = sampling::projected_hemisphere_generate(u);

    return LightSample<IncomingRayDirInfo>::createTangentSpace(tangentSpaceV,IncomingRayDirInfo::create(L),m);
}
template <class IncomingRayDirInfo>
LightSample<IncomingRayDirInfo> lambertian_cos_generate(in surface_interactions::Anisotropic<IncomingRayDirInfo> interaction, in float2 u)
{
    return lambertian_cos_generate_wo_clamps(interaction.getTangentSpaceV(),interaction.getTangentFrame(),u);
}



float lambertian_pdf_wo_clamps(in float maxNdotL)
{
    return sampling::projected_hemisphere_pdf(maxNdotL);
}

template <class IncomingRayDirInfo>
float lambertian_pdf(in LightSample<IncomingRayDirInfo> s, in surface_interactions::Isotropic<IncomingRayDirInfo> i)
{
    return lambertian_pdf_wo_clamps(max(s.NdotL,0.0f));
}


float lambertian_cos_remainder_and_pdf_wo_clamps(out float pdf, in float maxNdotL)
{
    return sampling::projected_hemisphere_remainder_and_pdf(pdf,maxNdotL);
}
template <class IncomingRayDirInfo>
float lambertian_cos_remainder_and_pdf(out float pdf, in LightSample<IncomingRayDirInfo> s)
{
    return sampling::projected_hemisphere_remainder_and_pdf(pdf,max(s.NdotL,0.0f));
}

}
}
}
}
}

#endif 
