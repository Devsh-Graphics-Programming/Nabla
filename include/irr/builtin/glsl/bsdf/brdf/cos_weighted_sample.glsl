#ifndef _IRR_BSDF_BRDF_COS_WEIGHTED_SAMPLE_INCLUDED_
#define _IRR_BSDF_BRDF_COS_WEIGHTED_SAMPLE_INCLUDED_

#include <irr/builtin/glsl/bsdf/common.glsl>

irr_glsl_BSDFSample irr_glsl_cos_weighted_cos_gen_sample(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec2 _sample)
{
    vec2 p = irr_glsl_concentricMapping(_sample);
    
    mat3 m = irr_glsl_getTangentFrame(interaction);
    float z = sqrt(max(0.0, 1.0 - p.x*p.x - p.y*p.y));
    vec3 L = m*vec3(p.x,p.y,z);

    irr_glsl_BSDFSample smpl;
    smpl.L = L;
    smpl.probability = dot(interaction.N,smpl.L)*irr_glsl_RECIPROCAL_PI;

    return smpl;
}
irr_glsl_BSDFSample irr_glsl_cos_weighted_cos_gen_sample(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in uvec2 _sample)
{
    vec2 u = vec2(_sample)/float(UINT_MAX);
    return irr_glsl_cos_weighted_cos_gen_sample(interaction, u);
}

#endif
