#ifndef _INSTANCE_DATA_PER_CAMERA_INCLUDED_
#define _INSTANCE_DATA_PER_CAMERA_INCLUDED_

#include "common.glsl"

struct CullShaderData_t
{
    mat4    viewProjMatrix;
    uint    maxObjectCount;
    uint    currentCommandBufferIx;
    float   viewProjDeterminant;
    uint    padding;
};

struct ObjectStaticData_t
{
    vec3    normalMatrixRow0;
    float   detWorldMatrix;
    vec3    normalMatrixRow1;
    uint    padding0;
    vec3    normalMatrixRow2;
    uint    padding1;
};

struct CullData_t
{
    mat4x3  worldMatrix;
    vec3    aabbMinEdge;
    uint    drawID;
    vec3    aabbMaxEdge;
    uint    baseInstance;
};

struct DrawData_t
{
    mat4 MVP;
    float detMVP;
    uint objectID;
    uint padding0;
    uint padding1;
};

#endif
