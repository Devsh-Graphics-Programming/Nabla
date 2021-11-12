#ifndef _NBL_BUILTIN_GLSL_UTILS_NORMAL_ENCODE_INCLUDED_
#define _NBL_BUILTIN_GLSL_UTILS_NORMAL_ENCODE_INCLUDED_

#include "nbl/builtin/glsl/math/constants.glsl"

#include "nbl/builtin/glsl/utils/compressed_normal_matrix_t.glsl"
nbl_glsl_CompressedNormalMatrix_t nbl_glsl_CompressedNormalMatrix_t_encode(in uint signFlipMask, mat3 m)
{
    const vec3 colmax = max(max(abs(m[0]),abs(m[1])),abs(m[2]));
    m /= uintBitsToFloat(floatBitsToUint(max(max(colmax.x,colmax.y),colmax.z))^signFlipMask);

    nbl_glsl_CompressedNormalMatrix_t compr;

    compr.data[0] = packSnorm2x16(m[0].yz);
    compr.data[1] = packSnorm2x16(m[1].xy);
    compr.data[2] = packSnorm2x16(vec2(m[1].z,m[2].x));
    compr.data[3] = packSnorm2x16(m[2].yz);
    compr.data &= uvec4(0xFFFCFFFCu);
    
    const uint firstComp = packSnorm2x16(vec2(m[0].x,0.f));
    compr.data[0] |= firstComp & 0x00030003u;
    compr.data[1] |= (firstComp>>2u) & 0x00030003u;
    compr.data[2] |= (firstComp>>4u) & 0x00030003u;
    compr.data[3] |= (firstComp>>6u) & 0x00030003u;

    return compr;
}

vec2 nbl_glsl_NormalEncode_signedSpherical(in vec3 n)
{
    return vec2(atan(n.y,n.x)/nbl_glsl_PI, n.z);
}

#endif