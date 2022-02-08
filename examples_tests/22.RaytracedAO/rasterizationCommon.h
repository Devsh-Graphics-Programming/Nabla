#ifndef _RASTERIZATION_COMMON_H_INCLUDED_
#define _RASTERIZATION_COMMON_H_INCLUDED_

#include "common.h"

struct CullShaderData_t
{
    mat4    viewProjMatrix;
    float   viewProjDeterminant;
    uint    currentCommandBufferIx;
    uint    maxDrawCommandCount;
    uint    maxGlobalInstanceCount;
};

struct CullData_t
{
    vec3    aabbMinEdge;
    uint    batchInstanceGUID;
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
