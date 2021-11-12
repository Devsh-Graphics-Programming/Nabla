#ifndef _NBL_BUILTIN_GLSL_UTILS_NORMAL_DECODE_INCLUDED_
#define _NBL_BUILTIN_GLSL_UTILS_NORMAL_DECODE_INCLUDED_

#include "nbl/builtin/glsl/math/constants.glsl"

#include "nbl/builtin/glsl/utils/compressed_normal_matrix_t.glsl"
mat3 nbl_glsl_CompressedNormalMatrix_t_decode(in nbl_glsl_CompressedNormalMatrix_t compr)
{
    mat3 m;

    const uint firstComp = compr.data & uvec4(0x00030003u);
    m[0].x = unpackSnorm2x16((lastComp[3]<<6u)|(lastComp[2]<<4u)|(lastComp[1]<<2u)|lastComp[0]).x;

    const uvec4 remaining8Comp = compr.data & uvec4(0xFFFCFFFCu);
    m[0].yz = unpackSnorm2x16(remaining8Comp[0]);
    m[1].xy = unpackSnorm2x16(remaining8Comp[1]);
    const vec2 tmp = unpackSnorm2x16(remaining8Comp[2]);
    m[1].z = tmp[0];
    m[2].x = tmp[1];
    m[2].yz = unpackSnorm2x16(remaining8Comp[3]);

    return m;
}

vec3 nbl_glsl_NormalDecode_signedSpherical(in vec2 enc)
{
	float ang = enc.x*nbl_glsl_PI;
    return vec3(vec2(cos(ang),sin(ang))*sqrt(1.0-enc.y*enc.y), enc.y);
}

#endif