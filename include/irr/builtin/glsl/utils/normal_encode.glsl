#ifndef _IRR_BUILTIN_GLSL_UTILS_NORMAL_ENCODE_INCLUDED_
#define _IRR_BUILTIN_GLSL_UTILS_NORMAL_ENCODE_INCLUDED_

#include "irr/builtin/glsl/math/constants.glsl"

vec2 irr_glsl_NormalEncode_signedSpherical(in vec3 n)
{
    return vec2(atan(n.y,n.x)/kPI, n.z);
}

#endif