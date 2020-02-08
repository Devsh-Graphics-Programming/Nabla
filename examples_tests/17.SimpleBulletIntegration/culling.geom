#version 430 core
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
    vec4 instanceWorldViewProjMatCol0 = ProjViewWorldMat[0]*gWorldMatPart0[0].x+ProjViewWorldMat[1]*gWorldMatPart1[0].x+ProjViewWorldMat[2]*gWorldMatPart2[0].x;
    vec4 instanceWorldViewProjMatCol1 = ProjViewWorldMat[0]*gWorldMatPart0[0].y+ProjViewWorldMat[1]*gWorldMatPart1[0].y+ProjViewWorldMat[2]*gWorldMatPart2[0].y;
    vec4 instanceWorldViewProjMatCol2 = ProjViewWorldMat[0]*gWorldMatPart0[0].z+ProjViewWorldMat[1]*gWorldMatPart1[0].z+ProjViewWorldMat[2]*gWorldMatPart2[0].z;
    vec4 instanceWorldViewProjMatCol3 = ProjViewWorldMat[0]*gWorldMatPart0[0].w+ProjViewWorldMat[1]*gWorldMatPart1[0].w+ProjViewWorldMat[2]*gWorldMatPart2[0].w+ProjViewWorldMat[3];

	
    vec3 instanceWorldMatCol0 = WorldMat[0]*gWorldMatPart0[0].x+WorldMat[1]*gWorldMatPart1[0].x+WorldMat[2]*gWorldMatPart2[0].x;
    vec3 instanceWorldMatCol1 = WorldMat[0]*gWorldMatPart0[0].y+WorldMat[1]*gWorldMatPart1[0].y+WorldMat[2]*gWorldMatPart2[0].y;
    vec3 instanceWorldMatCol2 = WorldMat[0]*gWorldMatPart0[0].z+WorldMat[1]*gWorldMatPart1[0].z+WorldMat[2]*gWorldMatPart2[0].z;
    vec3 instanceWorldMatCol3 = WorldMat[0]*gWorldMatPart0[0].w+WorldMat[1]*gWorldMatPart1[0].w+WorldMat[2]*gWorldMatPart2[0].w+WorldMat[3];


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
