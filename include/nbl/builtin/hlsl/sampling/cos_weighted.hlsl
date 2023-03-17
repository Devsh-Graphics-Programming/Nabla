// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SAMPLING_COS_WEIGHTED_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_COS_WEIGHTED_INCLUDED_

#include <nbl/builtin/hlsl/sampling/concentric_mapping.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{

float3 projected_hemisphere_generate(in float2 _sample)
{
    float2 p = concentricMapping(_sample*0.99999f+0.000005f);
    
    float z = sqrt(max(0.0f, 1.0f - p.x*p.x - p.y*p.y));
    
    return float3(p.x,p.y,z);
}

float projected_hemisphere_pdf(in float L_z)
{
    return L_z * math::RECIPROCAL_PI;
}

float projected_hemisphere_remainder_and_pdf(out float pdf, in float L_z)
{
	pdf = projected_hemisphere_pdf(L_z);
	return 1.0f;
}
float projected_hemisphere_remainder_and_pdf(out float pdf, in float3 L)
{
	return projected_hemisphere_remainder_and_pdf(pdf,L.z);
}

float3 projected_sphere_generate(inout float3 _sample) // TODO, it should be `inout`, right?
{
    float3 retval = projected_hemisphere_generate(_sample.xy);
    const bool chooseLower = _sample.z>0.5f;
    retval.z = chooseLower ? (-retval.z):retval.z;
    if (chooseLower)
        _sample.z -= 0.5f;
    _sample.z *= 2.f;
    return retval;
}

float projected_sphere_remainder_and_pdf(out float pdf, in float L_z)
{
    float retval = projected_hemisphere_remainder_and_pdf(pdf,L_z);
    pdf *= 0.5f;
	return retval;
}
float projected_sphere_remainder_and_pdf(out float pdf, in float3 L)
{
    return projected_sphere_remainder_and_pdf(pdf,L.z);
}

float projected_sphere_pdf(in float L_z)
{
    return 0.5f*projected_hemisphere_pdf(L_z);
}
}
}
}

#endif