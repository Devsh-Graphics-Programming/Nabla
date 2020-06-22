#ifndef _IRR_BSDF_BRDF_COS_WEIGHTED_SAMPLE_INCLUDED_
#define _IRR_BSDF_BRDF_COS_WEIGHTED_SAMPLE_INCLUDED_

#include <irr/builtin/glsl/bxdf/common.glsl>

irr_glsl_BSDFSample irr_glsl_cos_weighted_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec2 _sample)
{
    vec2 p = irr_glsl_concentricMapping(_sample);
    
    mat3 m = irr_glsl_getTangentFrame(interaction);
    float z = sqrt(max(0.0, 1.0 - p.x*p.x - p.y*p.y));
    vec3 L = m*vec3(p.x,p.y,z);

    irr_glsl_BSDFSample smpl;
    smpl.L = L;
	smpl.LdotN = dot(interaction.N,smpl.L);
	//TODO fill other smpl.xxxxx

    return smpl;
}
irr_glsl_BSDFSample irr_glsl_cos_weighted_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in uvec2 _sample)
{
    vec2 u = vec2(_sample)/float(UINT_MAX);
    return irr_glsl_cos_weighted_cos_generate(interaction, u);
}

//returning just 1.0 since we dont know which brdf is being sampled, maybe return irr_glsl_PI instead
//TODO oren-nayar cos_eval with rec_pi factored out
vec3 irr_glsl_cos_weighted_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFAnisotropicParams params, in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in uvec2 u)
{
	irr_glsl_BSDFSample s = irr_glsl_cos_weighted_cos_generate(interaction, u);
	
	float val = s.LdotN*irr_glsl_RECIPROCAL_PI;
	pdf = val;
	
	return 1.0;
}

#endif
