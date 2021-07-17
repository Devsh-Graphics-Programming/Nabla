#ifndef _RASTERIZATION_COMMON_H_INCLUDED_
#define _RASTERIZATION_COMMON_H_INCLUDED_

#include "cullShaderCommon.h"

struct CullShaderData_t
{
    mat4    viewProjMatrix;
    uint    maxBatchCount;
    uint    freezeCulling;
};

struct CullData_t
{
    vec3    aabbMinEdge;
    uint    padding;
    vec3    aabbMaxEdge;
    uint    drawCommandGUID;
};

#endif
