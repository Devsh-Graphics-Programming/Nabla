#version 400 core
layout(points) in;
layout(points, max_vertices = 1) out;

uniform mat4 ProjViewWorldMat;
uniform mat4x3 ViewWorldMat;
uniform mat4x3 WorldMat;
uniform vec3 eyePos;
uniform vec3 LoDInvariantMinEdge;
uniform vec3 LoDInvariantBBoxCenter;
uniform vec3 LoDInvariantMaxEdge;
uniform vec2 instanceLoDDistancesSQ;

in vec4 gWorldMatPart0[];
in vec4 gWorldMatPart1[];
in vec4 gWorldMatPart2[];
/*
struct InstanceData
{
    vec4 worldViewProjMatCol0;
    vec4 worldViewProjMatCol1;
    vec4 worldViewProjMatCol2;
    vec4 worldViewProjMatCol3;

    vec3 worldViewMatCol0;
    vec3 worldViewMatCol1;
    vec3 worldViewMatCol2;
    vec3 worldViewMatCol3;
};*/

layout(stream = 0) out vec4 outLoD0_worldViewProjMatCol0;
layout(stream = 0) out vec4 outLoD0_worldViewProjMatCol1;
layout(stream = 0) out vec4 outLoD0_worldViewProjMatCol2;
layout(stream = 0) out vec4 outLoD0_worldViewProjMatCol3;
layout(stream = 0) out vec3 outLoD0_worldViewMatCol0;
layout(stream = 0) out vec3 outLoD0_worldViewMatCol1;
layout(stream = 0) out vec3 outLoD0_worldViewMatCol2;
layout(stream = 0) out vec3 outLoD0_worldViewMatCol3;

layout(stream = 1) out vec4 outLoD1_worldViewProjMatCol0;
layout(stream = 1) out vec4 outLoD1_worldViewProjMatCol1;
layout(stream = 1) out vec4 outLoD1_worldViewProjMatCol2;
layout(stream = 1) out vec4 outLoD1_worldViewProjMatCol3;
layout(stream = 1) out vec3 outLoD1_worldViewMatCol0;
layout(stream = 1) out vec3 outLoD1_worldViewMatCol1;
layout(stream = 1) out vec3 outLoD1_worldViewMatCol2;
layout(stream = 1) out vec3 outLoD1_worldViewMatCol3;

void main()
{
    vec3 instancePos = gWorldMatPart0[0].xyz*LoDInvariantBBoxCenter.x+vec3(gWorldMatPart0[0].w,gWorldMatPart1[0].xy)*LoDInvariantBBoxCenter.y+vec3(gWorldMatPart1[0].zw,gWorldMatPart2[0].x)*LoDInvariantBBoxCenter.z+gWorldMatPart2[0].yzw;
    instancePos = WorldMat[0]*instancePos.x+WorldMat[1]*instancePos.y+WorldMat[2]*instancePos.z+WorldMat[3];
    vec3 eyeToInstance = instancePos-eyePos;
    float distanceToInstance = dot(eyeToInstance,eyeToInstance);
    if (distanceToInstance>=instanceLoDDistancesSQ.y)
        return;

    vec4 instanceWorldViewProjMatCol0 = ProjViewWorldMat[0]*gWorldMatPart0[0].x+ProjViewWorldMat[1]*gWorldMatPart0[0].y+ProjViewWorldMat[2]*gWorldMatPart0[0].z;
    vec4 instanceWorldViewProjMatCol1 = ProjViewWorldMat[0]*gWorldMatPart0[0].w+ProjViewWorldMat[1]*gWorldMatPart1[0].x+ProjViewWorldMat[2]*gWorldMatPart1[0].y;
    vec4 instanceWorldViewProjMatCol2 = ProjViewWorldMat[0]*gWorldMatPart1[0].z+ProjViewWorldMat[1]*gWorldMatPart1[0].w+ProjViewWorldMat[2]*gWorldMatPart2[0].x;
    vec4 instanceWorldViewProjMatCol3 = ProjViewWorldMat[0]*gWorldMatPart2[0].y+ProjViewWorldMat[1]*gWorldMatPart2[0].z+ProjViewWorldMat[2]*gWorldMatPart2[0].w+ProjViewWorldMat[3];

    ///Do frustum Culling, can stay because its correct, but will need a rewrite for the compute shader version for better readability
    // this might be wrong
    float tmp = instanceWorldViewProjMatCol0.w*(instanceWorldViewProjMatCol0.w<0.0 ? LoDInvariantMinEdge.x:LoDInvariantMaxEdge.x)+
                instanceWorldViewProjMatCol1.w*(instanceWorldViewProjMatCol1.w<0.0 ? LoDInvariantMinEdge.y:LoDInvariantMaxEdge.y)+
                instanceWorldViewProjMatCol2.w*(instanceWorldViewProjMatCol2.w<0.0 ? LoDInvariantMinEdge.z:LoDInvariantMaxEdge.z);
    if (tmp<=-instanceWorldViewProjMatCol3.w)
        return;


    vec2 columnMin[4];
    columnMin[0] = instanceWorldViewProjMatCol0.xy+instanceWorldViewProjMatCol0.ww;
    columnMin[1] = instanceWorldViewProjMatCol1.xy+instanceWorldViewProjMatCol1.ww;
    columnMin[2] = instanceWorldViewProjMatCol2.xy+instanceWorldViewProjMatCol2.ww;
    columnMin[3] =-instanceWorldViewProjMatCol3.xy-instanceWorldViewProjMatCol3.ww;

    tmp = columnMin[0].x*(columnMin[0].x<0.0 ? LoDInvariantMinEdge.x:LoDInvariantMaxEdge.x)+columnMin[1].x*(columnMin[1].x<0.0 ? LoDInvariantMinEdge.y:LoDInvariantMaxEdge.y)+columnMin[2].x*(columnMin[2].x<0.0 ? LoDInvariantMinEdge.z:LoDInvariantMaxEdge.z);
    if (tmp<=columnMin[3].x)
        return;
    tmp = columnMin[0].y*(columnMin[0].y<0.0 ? LoDInvariantMinEdge.x:LoDInvariantMaxEdge.x)+columnMin[1].y*(columnMin[1].y<0.0 ? LoDInvariantMinEdge.y:LoDInvariantMaxEdge.y)+columnMin[2].y*(columnMin[2].y<0.0 ? LoDInvariantMinEdge.z:LoDInvariantMaxEdge.z);
    if (tmp<=columnMin[3].y)
        return;


    vec2 columnMax[4];
    columnMax[0] = instanceWorldViewProjMatCol0.xy-instanceWorldViewProjMatCol0.ww;
    columnMax[1] = instanceWorldViewProjMatCol1.xy-instanceWorldViewProjMatCol1.ww;
    columnMax[2] = instanceWorldViewProjMatCol2.xy-instanceWorldViewProjMatCol2.ww;
    columnMax[3] = instanceWorldViewProjMatCol3.ww-instanceWorldViewProjMatCol3.xy;

    tmp = columnMax[0].x*(columnMax[0].x>=0.0 ? LoDInvariantMinEdge.x:LoDInvariantMaxEdge.x)+columnMax[1].x*(columnMax[1].x>=0.0 ? LoDInvariantMinEdge.y:LoDInvariantMaxEdge.y)+columnMax[2].x*(columnMax[2].x>=0.0 ? LoDInvariantMinEdge.z:LoDInvariantMaxEdge.z);
    if (tmp>=columnMax[3].x)
        return;
    tmp = columnMax[0].y*(columnMax[0].y>=0.0 ? LoDInvariantMinEdge.x:LoDInvariantMaxEdge.x)+columnMax[1].y*(columnMax[1].y>=0.0 ? LoDInvariantMinEdge.y:LoDInvariantMaxEdge.y)+columnMax[2].y*(columnMax[2].y>=0.0 ? LoDInvariantMinEdge.z:LoDInvariantMaxEdge.z);
    if (tmp>=columnMax[3].y)
        return;
    // no far-Z culling because distanceToInstance Culling should have taken care of it

    vec3 instanceWorldViewMatCol0 = ViewWorldMat[0]*gWorldMatPart0[0].x+ViewWorldMat[1]*gWorldMatPart0[0].y+ViewWorldMat[2]*gWorldMatPart0[0].z;
    vec3 instanceWorldViewMatCol1 = ViewWorldMat[0]*gWorldMatPart0[0].w+ViewWorldMat[1]*gWorldMatPart1[0].x+ViewWorldMat[2]*gWorldMatPart1[0].y;
    vec3 instanceWorldViewMatCol2 = ViewWorldMat[0]*gWorldMatPart1[0].z+ViewWorldMat[1]*gWorldMatPart1[0].w+ViewWorldMat[2]*gWorldMatPart2[0].x;
    vec3 instanceWorldViewMatCol3 = ViewWorldMat[0]*gWorldMatPart2[0].y+ViewWorldMat[1]*gWorldMatPart2[0].z+ViewWorldMat[2]*gWorldMatPart2[0].w+ViewWorldMat[3];

    if (distanceToInstance<instanceLoDDistancesSQ.x)
    {
        outLoD0_worldViewProjMatCol0 = instanceWorldViewProjMatCol0;
        outLoD0_worldViewProjMatCol1 = instanceWorldViewProjMatCol1;
        outLoD0_worldViewProjMatCol2 = instanceWorldViewProjMatCol2;
        outLoD0_worldViewProjMatCol3 = instanceWorldViewProjMatCol3;
        outLoD0_worldViewMatCol0 = instanceWorldViewMatCol0;
        outLoD0_worldViewMatCol1 = instanceWorldViewMatCol1;
        outLoD0_worldViewMatCol2 = instanceWorldViewMatCol2;
        outLoD0_worldViewMatCol3 = instanceWorldViewMatCol3;
        EmitStreamVertex(0);
        EndStreamPrimitive(0);
    }
    else
    {
        outLoD1_worldViewProjMatCol0 = instanceWorldViewProjMatCol0;
        outLoD1_worldViewProjMatCol1 = instanceWorldViewProjMatCol1;
        outLoD1_worldViewProjMatCol2 = instanceWorldViewProjMatCol2;
        outLoD1_worldViewProjMatCol3 = instanceWorldViewProjMatCol3;
        outLoD1_worldViewMatCol0 = instanceWorldViewMatCol0;
        outLoD1_worldViewMatCol1 = instanceWorldViewMatCol1;
        outLoD1_worldViewMatCol2 = instanceWorldViewMatCol2;
        outLoD1_worldViewMatCol3 = instanceWorldViewMatCol3;
        EmitStreamVertex(1);
        EndStreamPrimitive(1);
    }
}
