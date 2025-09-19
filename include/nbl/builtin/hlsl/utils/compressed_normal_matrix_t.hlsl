
#ifndef _NBL_BUILTIN_HLSL_UTILS_COMPRESSED_NORMAL_MATRIX_T_INCLUDED_
#define _NBL_BUILTIN_HLSL_UTILS_COMPRESSED_NORMAL_MATRIX_T_INCLUDED_

#include "nbl/builtin/hlsl/common.hlsl"
#include "nbl/builtin/hlsl/math/constants.hlsl"

namespace nbl
{
namespace hlsl
{

struct CompressedNormalMatrix_t
{
    uint4 data;


    float3x3 decode(in CompressedNormalMatrix_t compr)
    {
        float3x3 m;

        const uint4 bottomBits = compr.data & (0x00030003u).xxxx;
        const uint firstComp = (bottomBits[3]<<6u)|(bottomBits[2]<<4u)|(bottomBits[1]<<2u)|bottomBits[0];
        m[0].x = unpackSnorm2x16((firstComp>>8u)|firstComp).x;

        const uint4 remaining8Comp = compr.data & (0xFFFCFFFCu).xxxx;
        m[0].yz = unpackSnorm2x16(remaining8Comp[0]);
        m[1].xy = unpackSnorm2x16(remaining8Comp[1]);
        const float2 tmp = unpackSnorm2x16(remaining8Comp[2]);
        m[1].z = tmp[0];
        m[2].x = tmp[1];
        m[2].yz = unpackSnorm2x16(remaining8Comp[3]);

        return m;
    }
};

}
}

#endif