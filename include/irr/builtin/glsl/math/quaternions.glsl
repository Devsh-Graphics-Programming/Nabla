#ifndef _NBL_BUILTIN_GLSL_MATH_QUATERNIONS_INCLUDED_
#define _NBL_BUILTIN_GLSL_MATH_QUATERNIONS_INCLUDED_

#include <irr/builtin/glsl/math/functions.glsl>

vec3 irr_glsl_slerp_impl_impl(in vec3 start, in vec3 preScaledWaypoint, float cosAngleFromStart)
{
    vec3 planeNormal = cross(start,preScaledWaypoint);
    
    cosAngleFromStart *= 0.5;
    const float sinAngle = sqrt(0.5-cosAngleFromStart);
    const float cosAngle = sqrt(0.5+cosAngleFromStart);
    
    planeNormal *= sinAngle;
    const vec3 precompPart = cross(planeNormal,start)*2.0;

    return start+precompPart*cosAngle+cross(planeNormal,precompPart);
}

#endif
