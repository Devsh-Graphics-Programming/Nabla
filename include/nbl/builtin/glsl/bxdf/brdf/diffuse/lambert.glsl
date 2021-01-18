// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BXDF_BRDF_DIFFUSE_LAMBERT_INCLUDED_
#define _NBL_BXDF_BRDF_DIFFUSE_LAMBERT_INCLUDED_

#include <nbl/builtin/glsl/bxdf/common.glsl>
#include <nbl/builtin/glsl/sampling/cos_weighted.glsl>

float nbl_glsl_lambertian()
{
    return nbl_glsl_RECIPROCAL_PI;
}

float nbl_glsl_lambertian_cos_eval_rec_pi_factored_out_wo_clamps(in float maxNdotL)
{
   return maxNdotL;
}
float nbl_glsl_lambertian_cos_eval_rec_pi_factored_out(in float NdotL)
{
   return nbl_glsl_lambertian_cos_eval_rec_pi_factored_out_wo_clamps(max(NdotL,0.0));
}

float nbl_glsl_lambertian_cos_eval_wo_clamps(in float maxNdotL)
{
   return nbl_glsl_lambertian_cos_eval_rec_pi_factored_out_wo_clamps(maxNdotL)*nbl_glsl_lambertian();
}
float nbl_glsl_lambertian_cos_eval(in nbl_glsl_LightSample _sample)
{
    return nbl_glsl_lambertian_cos_eval_rec_pi_factored_out(_sample.NdotL)*nbl_glsl_lambertian();
}

nbl_glsl_LightSample nbl_glsl_lambertian_cos_generate_wo_clamps(in vec3 tangentSpaceV, in mat3 m, in vec2 u)
{
    vec3 L = nbl_glsl_projected_hemisphere_generate(u);

    return nbl_glsl_createLightSampleTangentSpace(tangentSpaceV,L,m);
}
nbl_glsl_LightSample nbl_glsl_lambertian_cos_generate(in nbl_glsl_AnisotropicViewSurfaceInteraction interaction, in vec2 u)
{
    return nbl_glsl_lambertian_cos_generate_wo_clamps(nbl_glsl_getTangentSpaceV(interaction),nbl_glsl_getTangentFrame(interaction),u);
}



float nbl_glsl_lambertian_pdf_wo_clamps(in float maxNdotL)
{
    return nbl_glsl_projected_hemisphere_pdf(maxNdotL);
}

float nbl_glsl_lambertian_pdf(in nbl_glsl_LightSample s, in nbl_glsl_IsotropicViewSurfaceInteraction i)
{
    return nbl_glsl_lambertian_pdf_wo_clamps(max(s.NdotL,0.0));
}


float nbl_glsl_lambertian_cos_remainder_and_pdf_wo_clamps(out float pdf, in float maxNdotL)
{
    return nbl_glsl_projected_hemisphere_remainder_and_pdf(pdf,maxNdotL);
}
float nbl_glsl_lambertian_cos_remainder_and_pdf(out float pdf, in nbl_glsl_LightSample s)
{
    return nbl_glsl_projected_hemisphere_remainder_and_pdf(pdf,max(s.NdotL,0.0));
}

#endif
