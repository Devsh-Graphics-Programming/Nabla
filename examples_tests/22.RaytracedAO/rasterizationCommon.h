#ifndef _RASTERIZATION_COMMON_H_INCLUDED_
#define _RASTERIZATION_COMMON_H_INCLUDED_

#include "common.h"

struct CullShaderData_t
{
    mat4    viewProjMatrix;
    float   viewProjDeterminant;
    uint    currentCommandBufferIx;
    uint    maxDrawCount;
    uint    maxObjectCount;
};

struct CullData_t
{
    vec3    aabbMinEdge;
    uint    globalObjectID;
    vec3    aabbMaxEdge;
    uint    drawID;
};

struct DrawData_t
{
    mat4 MVP;
    uint backfacingBit_objectID;
    uint padding0;
    uint padding1;
    uint padding2;
};

#endif
