// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_BRDF_DIFFUSE_OREN_NAYAR_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_BRDF_DIFFUSE_OREN_NAYAR_INCLUDED_

#include <nbl/builtin/hlsl/bxdf/brdf/diffuse/lambert.hlsl>

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

float oren_nayar_cos_rec_pi_factored_out_wo_clamps(in float _a2, in float VdotL, in float maxNdotL, in float maxNdotV)
{
    // theta - polar angles
    // phi - azimuth angles
    float a2 = _a2*0.5; //todo read about this&
    float2 AB = float2(1.0, 0.0) + float2(-0.5, 0.45) * float2(a2, a2)/float2(a2+0.33, a2+0.09);
    float C = 1.0 / max(maxNdotL, maxNdotV);

    // should be equal to cos(phi)*sin(theta_i)*sin(theta_o)
    // where `phi` is the angle in the tangent plane to N, between L and V
    // and `theta_i` is the sine of the angle between L and N, similarily for `theta_o` but with V
    float cos_phi_sin_theta = max(VdotL-maxNdotL*maxNdotV,0.0f);
    
    return (AB.x + AB.y * cos_phi_sin_theta * C);
}


float oren_nayar_cos_eval_wo_clamps(in float a2, in float VdotL, in float maxNdotL, in float maxNdotV)
{
    return maxNdotL*math::RECIPROCAL_PI*oren_nayar_cos_rec_pi_factored_out_wo_clamps(a2,VdotL,maxNdotL,maxNdotV);
}

template <class IncomingRayDirInfo>
float oren_nayar_cos_eval(in LightSample<IncomingRayDirInfo> _sample, in surface_interactions::Isotropic<IncomingRayDirInfo> inter, in float a2)
{
    return oren_nayar_cos_eval_wo_clamps(a2, _sample.VdotL, max(_sample.NdotL,0.0f), max(inter.NdotV,0.0f));
}


template <class IncomingRayDirInfo>
LightSample<IncomingRayDirInfo> oren_nayar_cos_generate_wo_clamps(in float3 tangentSpaceV, in float3x3 m, in float2 u)
{
    // until we find something better
    return lambertian_cos_generate_wo_clamps<IncomingRayDirInfo>(tangentSpaceV, m, u);
}
template <class IncomingRayDirInfo>
LightSample<IncomingRayDirInfo> oren_nayar_cos_generate(in surface_interactions::Anisotropic<IncomingRayDirInfo> interaction, in float2 u, in float a2)
{
    return oren_nayar_cos_generate_wo_clamps<IncomingRayDirInfo>(getTangentSpaceV(interaction),getTangentFrame(interaction),u);
}


float oren_nayar_pdf_wo_clamps(in float maxNdotL)
{
    return lambertian_pdf_wo_clamps(maxNdotL);
}

template <class IncomingRayDirInfo>
float oren_nayar_pdf(in LightSample<IncomingRayDirInfo> s, in surface_interactions::Isotropic<IncomingRayDirInfo> i)
{
    return lambertian_pdf<IncomingRayDirInfo>(s, i);
}


quotient_and_pdf_scalar oren_nayar_cos_quotient_and_pdf_wo_clamps(in float a2, in float VdotL, in float maxNdotL, in float maxNdotV)
{
    float pdf = oren_nayar_pdf_wo_clamps(maxNdotL);
    return quotient_and_pdf_scalar::create(
        oren_nayar_cos_rec_pi_factored_out_wo_clamps(a2,VdotL,maxNdotL,maxNdotV),
        pdf
    );
}

template <class IncomingRayDirInfo>
quotient_and_pdf_scalar oren_nayar_cos_quotient_and_pdf(in LightSample<IncomingRayDirInfo> s, in surface_interactions::Isotropic<IncomingRayDirInfo> interaction, in float a2)
{
    return oren_nayar_cos_quotient_and_pdf_wo_clamps(a2,dot(interaction.V.getDirection(),s.L), max(s.NdotL,0.0f), max(interaction.NdotV,0.0f));
}

}
}
}
}
}

#endif
