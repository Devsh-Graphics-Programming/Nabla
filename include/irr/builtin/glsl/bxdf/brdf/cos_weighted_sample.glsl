#ifndef _IRR_BSDF_BRDF_COS_WEIGHTED_SAMPLE_INCLUDED_
#define _IRR_BSDF_BRDF_COS_WEIGHTED_SAMPLE_INCLUDED_

#include <irr/builtin/glsl/bxdf/common_samples.glsl>

// TODO: functions for sampling according to `abs(NdotL)*0.5/irr_glsl_RECIPROCAL_PI` in `glsl/bxdf/bsdf`
irr_glsl_BSDFSample irr_glsl_cos_weighted_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec2 _sample)
{
    vec2 p = irr_glsl_concentricMapping(_sample);
    
    mat3 m = irr_glsl_getTangentFrame(interaction);
    float z = sqrt(max(0.0, 1.0 - p.x*p.x - p.y*p.y));
    vec3 L = m*vec3(p.x,p.y,z);

    irr_glsl_BSDFSample smpl;
    smpl.L = L;
	smpl.NdotL = z;
    smpl.TdotL = p.x;
    smpl.BdotL = p.y;

    return smpl;
}

vec3 irr_glsl_cos_weighted_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFSample s, in irr_glsl_IsotropicViewSurfaceInteraction interaction)
{
	pdf = s.NdotL*irr_glsl_RECIPROCAL_PI;
	return vec3(1.0);
}

#endif
