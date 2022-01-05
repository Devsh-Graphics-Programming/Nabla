// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_SAMPLING_COS_WEIGHTED_INCLUDED_
#define _NBL_BUILTIN_GLSL_SAMPLING_COS_WEIGHTED_INCLUDED_

#include <nbl/builtin/glsl/sampling/concentric_mapping.glsl>

vec3 nbl_glsl_projected_hemisphere_generate(in vec2 _sample)
{
    vec2 p = nbl_glsl_concentricMapping(_sample*0.99999+0.000005);
    
    float z = sqrt(max(0.0, 1.0 - p.x*p.x - p.y*p.y));
    
    return vec3(p.x,p.y,z);
}

float nbl_glsl_projected_hemisphere_pdf(in float L_z)
{
    return L_z * nbl_glsl_RECIPROCAL_PI;
}

float nbl_glsl_projected_hemisphere_remainder_and_pdf(out float pdf, in float L_z)
{
	pdf = nbl_glsl_projected_hemisphere_pdf(L_z);
	return 1.0;
}
float nbl_glsl_projected_hemisphere_remainder_and_pdf(out float pdf, in vec3 L)
{
	return nbl_glsl_projected_hemisphere_remainder_and_pdf(pdf,L.z);
}

vec3 nbl_glsl_projected_sphere_generate(inout vec3 _sample)
{
    vec3 retval = nbl_glsl_projected_hemisphere_generate(_sample.xy);
    const bool chooseLower = _sample.z>0.5;
    retval.z = chooseLower ? (-retval.z):retval.z;
    if (chooseLower)
        _sample.z -= 0.5f;
    _sample.z *= 2.f;
    return retval;
}

float nbl_glsl_projected_sphere_remainder_and_pdf(out float pdf, in float L_z)
{
    float retval = nbl_glsl_projected_hemisphere_remainder_and_pdf(pdf,L_z);
    pdf *= 0.5;
	return retval;
}
float nbl_glsl_projected_sphere_remainder_and_pdf(out float pdf, in vec3 L)
{
    return nbl_glsl_projected_sphere_remainder_and_pdf(pdf,L.z);
}

float nbl_glsl_projected_sphere_pdf(in float L_z)
{
    return 0.5*nbl_glsl_projected_hemisphere_pdf(L_z);
}

#endif
