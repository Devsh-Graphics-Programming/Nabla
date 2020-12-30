#ifndef _NBL_BUILTIN_GLSL_UTILS_NORMAL_DECODE_INCLUDED_
#define _NBL_BUILTIN_GLSL_UTILS_NORMAL_DECODE_INCLUDED_

#include "nbl/builtin/glsl/math/constants.glsl"

vec3 nbl_glsl_NormalDecode_signedSpherical(in vec2 enc)
{
	float ang = enc.x*nbl_glsl_PI;
    return vec3(vec2(cos(ang),sin(ang))*sqrt(1.0-enc.y*enc.y), enc.y);
}

#endif