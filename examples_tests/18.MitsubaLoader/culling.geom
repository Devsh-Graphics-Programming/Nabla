// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 430 core
layout(points) in;
layout(points, max_vertices = 1) out;

uniform mat4 ProjViewWorldMat;
uniform vec3 LoDInvariantMinEdge;
uniform vec3 LoDInvariantMaxEdge;

in vec4 gWorldMatPart0[];
in vec4 gWorldMatPart1[];
in vec4 gWorldMatPart2[];

layout(stream = 0) out vec4 worldViewProjMatCol0;
layout(stream = 0) out vec4 worldViewProjMatCol1;
layout(stream = 0) out vec4 worldViewProjMatCol2;
layout(stream = 0) out vec4 worldViewProjMatCol3;
layout(stream = 0) out vec3 tposeInverseWorldMatCol0;
layout(stream = 0) out vec3 tposeInverseWorldMatCol1;
layout(stream = 0) out vec3 tposeInverseWorldMatCol2;

void main()
{
    vec4 instanceWorldViewProjMatCol0 = ProjViewWorldMat[0]*gWorldMatPart0[0].x+ProjViewWorldMat[1]*gWorldMatPart1[0].x+ProjViewWorldMat[2]*gWorldMatPart2[0].x;
    vec4 instanceWorldViewProjMatCol1 = ProjViewWorldMat[0]*gWorldMatPart0[0].y+ProjViewWorldMat[1]*gWorldMatPart1[0].y+ProjViewWorldMat[2]*gWorldMatPart2[0].y;
    vec4 instanceWorldViewProjMatCol2 = ProjViewWorldMat[0]*gWorldMatPart0[0].z+ProjViewWorldMat[1]*gWorldMatPart1[0].z+ProjViewWorldMat[2]*gWorldMatPart2[0].z;
    vec4 instanceWorldViewProjMatCol3 = ProjViewWorldMat[0]*gWorldMatPart0[0].w+ProjViewWorldMat[1]*gWorldMatPart1[0].w+ProjViewWorldMat[2]*gWorldMatPart2[0].w+ProjViewWorldMat[3];

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


    vec3 columnMax[4];
    columnMax[0] = instanceWorldViewProjMatCol0.xyz-instanceWorldViewProjMatCol0.www;
    columnMax[1] = instanceWorldViewProjMatCol1.xyz-instanceWorldViewProjMatCol1.www;
    columnMax[2] = instanceWorldViewProjMatCol2.xyz-instanceWorldViewProjMatCol2.www;
    columnMax[3] = instanceWorldViewProjMatCol3.www-instanceWorldViewProjMatCol3.xyz;

    tmp = columnMax[0].x*(columnMax[0].x>=0.0 ? LoDInvariantMinEdge.x:LoDInvariantMaxEdge.x)+columnMax[1].x*(columnMax[1].x>=0.0 ? LoDInvariantMinEdge.y:LoDInvariantMaxEdge.y)+columnMax[2].x*(columnMax[2].x>=0.0 ? LoDInvariantMinEdge.z:LoDInvariantMaxEdge.z);
    if (tmp>=columnMax[3].x)
        return;
    tmp = columnMax[0].y*(columnMax[0].y>=0.0 ? LoDInvariantMinEdge.x:LoDInvariantMaxEdge.x)+columnMax[1].y*(columnMax[1].y>=0.0 ? LoDInvariantMinEdge.y:LoDInvariantMaxEdge.y)+columnMax[2].y*(columnMax[2].y>=0.0 ? LoDInvariantMinEdge.z:LoDInvariantMaxEdge.z);
    if (tmp>=columnMax[3].y)
        return;
    tmp = columnMax[0].z*(columnMax[0].z>=0.0 ? LoDInvariantMinEdge.x:LoDInvariantMaxEdge.x)+columnMax[1].z*(columnMax[1].z>=0.0 ? LoDInvariantMinEdge.y:LoDInvariantMaxEdge.y)+columnMax[2].z*(columnMax[2].z>=0.0 ? LoDInvariantMinEdge.z:LoDInvariantMaxEdge.z);
    if (tmp>=columnMax[3].z)
        return;

    worldViewProjMatCol0 = instanceWorldViewProjMatCol0;
    worldViewProjMatCol1 = instanceWorldViewProjMatCol1;
    worldViewProjMatCol2 = instanceWorldViewProjMatCol2;
    worldViewProjMatCol3 = instanceWorldViewProjMatCol3;
	mat3 tmp2 = transpose(inverse(mat3(gWorldMatPart0[0].xyz,gWorldMatPart1[0].xyz,gWorldMatPart2[0].xyz)));
	tposeInverseWorldMatCol0 = tmp2[0];
	tposeInverseWorldMatCol1 = tmp2[1];
	tposeInverseWorldMatCol2 = tmp2[2];
    EmitStreamVertex(0);
    EndStreamPrimitive(0);
}
