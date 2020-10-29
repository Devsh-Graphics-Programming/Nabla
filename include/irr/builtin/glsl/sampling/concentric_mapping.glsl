#ifndef _NBL_BUILTIN_GLSL_SAMPLING_CONCENTRIC_MAPPING_INCLUDED_
#define _NBL_BUILTIN_GLSL_SAMPLING_CONCENTRIC_MAPPING_INCLUDED_

#include <irr/builtin/glsl/math/functions.glsl>

vec2 irr_glsl_concentricMapping(in vec2 _u)
{
    //map [0;1]^2 to [-1;1]^2
    vec2 u = 2.0*_u - 1.0;
    
    vec2 p;
    if (u==vec2(0.0))
        p = vec2(0.0);
    else
    {
        float r;
        float theta;
        if (abs(u.x)>abs(u.y)) {
            r = u.x;
            theta = 0.25*irr_glsl_PI * (u.y/u.x);
        } else {
            r = u.y;
            theta = 0.5*irr_glsl_PI - 0.25*irr_glsl_PI*(u.x/u.y);
        }
		// TODO: use irr_glsl_sincos, but check that theta is in [-PI,PI]
        p = r*vec2(cos(theta),sin(theta));
    }

    return p;
}

#endif
