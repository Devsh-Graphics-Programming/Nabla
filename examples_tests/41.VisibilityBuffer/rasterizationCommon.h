#ifndef _RASTERIZATION_COMMON_H_INCLUDED_
#define _RASTERIZATION_COMMON_H_INCLUDED_

#include "cullShaderCommon.h"

struct CullShaderData_t
{
    mat4    viewProjMatrix;
    float   viewProjDeterminant;
    uint    maxBatchCount;
};

struct CullData_t
{
    vec3    aabbMinEdge;
    vec3    aabbMaxEdge;
    uint    drawCommandGUID;
};

struct DrawData_t
{
    mat4 MVP;
    uint backfacingBit_batchInstanceGUID;
    uint firstIndex;
    uint padding1;
    uint padding2;
};

#endif
