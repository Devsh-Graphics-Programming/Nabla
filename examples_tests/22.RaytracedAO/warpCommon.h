#ifndef _WARP_COMMON_H_INCLUDED_
#define _WARP_COMMON_H_INCLUDED_

#include "common.h"

#define LUMA_MIP_MAP_GEN_WORKGROUP_DIM 16
#define WARP_MAP_GEN_WORKGROUP_DIM 16

struct LumaMipMapGenShaderData_t
{
	vec4 luminanceScales;
    uint calcLuma;
    vec3 padding;
};

struct WarpMapGenShaderData_t
{
    uint lumaMipCount;
    vec3 padding;
};

#endif
