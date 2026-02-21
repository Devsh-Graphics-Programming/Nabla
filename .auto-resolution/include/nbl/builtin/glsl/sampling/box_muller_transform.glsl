#ifndef _NBL_BUILTIN_GLSL_BOX_MULLER_TRANSFORM_INCLUDED_
#define _NBL_BUILTIN_GLSL_BOX_MULLER_TRANSFORM_INCLUDED_

#include <nbl/builtin/glsl/math/functions.glsl>

vec2 nbl_glsl_BoxMullerTransform(in vec2 xi, in float stddev)
{
    float sinPhi, cosPhi;
    nbl_glsl_sincos(2.0 * nbl_glsl_PI * xi.y - nbl_glsl_PI, sinPhi, cosPhi);
    return vec2(cosPhi, sinPhi) * sqrt(-2.0 * log(xi.x)) * stddev;
}

#endif
