#version 400 core
layout(points) in;
layout(points, max_vertices = 1) out;

uniform mat4 ProjViewWorldMat;
uniform mat4x3 WorldMat;
uniform vec3 eyePos;
uniform vec3 LoDInvariantMinEdge;
uniform vec3 LoDInvariantBBoxCenter;
uniform vec3 LoDInvariantMaxEdge;

in vec4 gWorldMatPart0[];
in vec4 gWorldMatPart1[];
in vec4 gWorldMatPart2[];
in uint gInstanceColor[];
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
layout(stream = 0) out vec3 outLoD0_worldMatCol0;
layout(stream = 0) out vec3 outLoD0_worldMatCol1;
layout(stream = 0) out vec3 outLoD0_worldMatCol2;
layout(stream = 0) out vec3 outLoD0_worldMatCol3;
layout(stream = 0) out uint outLoD0_instanceColor;

void main()
{
    vec4 instanceWorldViewProjMatCol0 = ProjViewWorldMat[0]*gWorldMatPart0[0].x+ProjViewWorldMat[1]*gWorldMatPart0[0].y+ProjViewWorldMat[2]*gWorldMatPart0[0].z;
    vec4 instanceWorldViewProjMatCol1 = ProjViewWorldMat[0]*gWorldMatPart0[0].w+ProjViewWorldMat[1]*gWorldMatPart1[0].x+ProjViewWorldMat[2]*gWorldMatPart1[0].y;
    vec4 instanceWorldViewProjMatCol2 = ProjViewWorldMat[0]*gWorldMatPart1[0].z+ProjViewWorldMat[1]*gWorldMatPart1[0].w+ProjViewWorldMat[2]*gWorldMatPart2[0].x;
    vec4 instanceWorldViewProjMatCol3 = ProjViewWorldMat[0]*gWorldMatPart2[0].y+ProjViewWorldMat[1]*gWorldMatPart2[0].z+ProjViewWorldMat[2]*gWorldMatPart2[0].w+ProjViewWorldMat[3];

    float tmp;
/*
    ///Do frustum Culling
    float tmp = instanceWorldViewProjMatCol0.w*(instanceWorldViewProjMatCol0.w<0.0 ? LoDInvariantMinEdge.x:LoDInvariantMaxEdge.x)+
                instanceWorldViewProjMatCol1.w*(instanceWorldViewProjMatCol1.w<0.0 ? LoDInvariantMinEdge.y:LoDInvariantMaxEdge.y)+
                instanceWorldViewProjMatCol2.w*(instanceWorldViewProjMatCol2.w<0.0 ? LoDInvariantMinEdge.z:LoDInvariantMaxEdge.z);
    if (tmp<=-instanceWorldViewProjMatCol3.w)
        return;
*/

    vec2 columnMin[4];
    columnMin[0] = instanceWorldViewProjMatCol0.xy+instanceWorldViewProjMatCol0.ww;
    columnMin[1] = instanceWorldViewProjMatCol1.xy+instanceWorldViewProjMatCol1.ww;
    columnMin[2] = instanceWorldViewProjMatCol2.xy+instanceWorldViewProjMatCol2.ww;
    columnMin[3] =-instanceWorldViewProjMatCol3.xy-instanceWorldViewProjMatCol3.ww;
/*
    tmp = columnMin[0].x*(columnMin[0].x<0.0 ? LoDInvariantMinEdge.x:LoDInvariantMaxEdge.x)+columnMin[1].x*(columnMin[1].x<0.0 ? LoDInvariantMinEdge.y:LoDInvariantMaxEdge.y)+columnMin[2].x*(columnMin[2].x<0.0 ? LoDInvariantMinEdge.z:LoDInvariantMaxEdge.z);
    if (tmp<=columnMin[3].x)
        return;
    tmp = columnMin[0].y*(columnMin[0].y<0.0 ? LoDInvariantMinEdge.x:LoDInvariantMaxEdge.x)+columnMin[1].y*(columnMin[1].y<0.0 ? LoDInvariantMinEdge.y:LoDInvariantMaxEdge.y)+columnMin[2].y*(columnMin[2].y<0.0 ? LoDInvariantMinEdge.z:LoDInvariantMaxEdge.z);
    if (tmp<=columnMin[3].y)
        return;
*/

    vec3 columnMax[4];
    columnMax[0] = instanceWorldViewProjMatCol0.xyz-instanceWorldViewProjMatCol0.www;
    columnMax[1] = instanceWorldViewProjMatCol1.xyz-instanceWorldViewProjMatCol1.www;
    columnMax[2] = instanceWorldViewProjMatCol2.xyz-instanceWorldViewProjMatCol2.www;
    columnMax[3] = instanceWorldViewProjMatCol3.www-instanceWorldViewProjMatCol3.xyz;
/*
    tmp = columnMax[0].x*(columnMax[0].x>=0.0 ? LoDInvariantMinEdge.x:LoDInvariantMaxEdge.x)+columnMax[1].x*(columnMax[1].x>=0.0 ? LoDInvariantMinEdge.y:LoDInvariantMaxEdge.y)+columnMax[2].x*(columnMax[2].x>=0.0 ? LoDInvariantMinEdge.z:LoDInvariantMaxEdge.z);
    if (tmp>=columnMax[3].x)
        return;
    tmp = columnMax[0].y*(columnMax[0].y>=0.0 ? LoDInvariantMinEdge.x:LoDInvariantMaxEdge.x)+columnMax[1].y*(columnMax[1].y>=0.0 ? LoDInvariantMinEdge.y:LoDInvariantMaxEdge.y)+columnMax[2].y*(columnMax[2].y>=0.0 ? LoDInvariantMinEdge.z:LoDInvariantMaxEdge.z);
    if (tmp>=columnMax[3].y)
        return;

    tmp = columnMax[0].z*(columnMax[0].z>=0.0 ? LoDInvariantMinEdge.x:LoDInvariantMaxEdge.x)+columnMax[1].z*(columnMax[1].z>=0.0 ? LoDInvariantMinEdge.y:LoDInvariantMaxEdge.y)+columnMax[2].z*(columnMax[2].z>=0.0 ? LoDInvariantMinEdge.z:LoDInvariantMaxEdge.z);
    if (tmp>=columnMax[3].z)
        return;
*/
    vec3 instanceWorldMatCol0 = WorldMat[0]*gWorldMatPart0[0].x+WorldMat[1]*gWorldMatPart0[0].y+WorldMat[2]*gWorldMatPart0[0].z;
    vec3 instanceWorldMatCol1 = WorldMat[0]*gWorldMatPart0[0].w+WorldMat[1]*gWorldMatPart1[0].x+WorldMat[2]*gWorldMatPart1[0].y;
    vec3 instanceWorldMatCol2 = WorldMat[0]*gWorldMatPart1[0].z+WorldMat[1]*gWorldMatPart1[0].w+WorldMat[2]*gWorldMatPart2[0].x;
    vec3 instanceWorldMatCol3 = WorldMat[0]*gWorldMatPart2[0].y+WorldMat[1]*gWorldMatPart2[0].z+WorldMat[2]*gWorldMatPart2[0].w+WorldMat[3];


    outLoD0_worldViewProjMatCol0 = instanceWorldViewProjMatCol0;
    outLoD0_worldViewProjMatCol1 = instanceWorldViewProjMatCol1;
    outLoD0_worldViewProjMatCol2 = instanceWorldViewProjMatCol2;
    outLoD0_worldViewProjMatCol3 = instanceWorldViewProjMatCol3;
    outLoD0_worldMatCol0 = instanceWorldMatCol0;
    outLoD0_worldMatCol1 = instanceWorldMatCol1;
    outLoD0_worldMatCol2 = instanceWorldMatCol2;
    outLoD0_worldMatCol3 = instanceWorldMatCol3;
    outLoD0_instanceColor = gInstanceColor[0];
    EmitStreamVertex(0);
    EndStreamPrimitive(0);
}
