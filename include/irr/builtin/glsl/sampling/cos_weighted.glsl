#ifndef _NBL_BUILTIN_GLSL_SAMPLING_COS_WEIGHTED_INCLUDED_
#define _NBL_BUILTIN_GLSL_SAMPLING_COS_WEIGHTED_INCLUDED_

#include <irr/builtin/glsl/sampling/concentric_mapping.glsl>

vec3 irr_glsl_projected_hemisphere_generate(in vec2 _sample)
{
    vec2 p = irr_glsl_concentricMapping(_sample);
    
    float z = sqrt(max(0.0, 1.0 - p.x*p.x - p.y*p.y));
    
    return vec3(p.x,p.y,z);
}

float irr_glsl_projected_hemisphere_remainder_and_pdf(out float pdf, in float L_z)
{
	pdf = L_z*irr_glsl_RECIPROCAL_PI;
	return 1.0;
}
float irr_glsl_projected_hemisphere_remainder_and_pdf(out float pdf, in vec3 L)
{
	return irr_glsl_projected_hemisphere_remainder_and_pdf(pdf,L.z);
}

vec3 irr_glsl_projected_sphere_generate(in vec3 _sample)
{
    vec3 retval = irr_glsl_projected_hemisphere_generate(_sample.xy);
    retval.z = _sample.z>0.5 ? (-retval.z):retval.z;
    return retval;
}

float irr_glsl_projected_sphere_remainder_and_pdf(out float pdf, in float L_z)
{
    float retval = irr_glsl_projected_hemisphere_remainder_and_pdf(pdf,L_z);
    pdf *= 0.5;
	return retval;
}
float irr_glsl_projected_sphere_remainder_and_pdf(out float pdf, in vec3 L)
{
    return irr_glsl_projected_sphere_remainder_and_pdf(pdf,L.z);
}

#endif
