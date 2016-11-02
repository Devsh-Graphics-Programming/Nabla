#version 330 core
layout(points) in;
layout(points, max_vertices = 1) out;

uniform mat4 ProjViewWorldMat;
uniform mat4x3 ViewWorldMat;
uniform mat4x3 WorldMat;
uniform mat3 NormalMat;
uniform vec3 eyePos;
uniform vec3 LoDInvariantMinEdge;
uniform vec3 LoDInvariantBBoxCenter;
uniform vec3 LoDInvariantMaxEdge;
vec3 instanceLoDDistanceSQ = vec3(0.0,64.0,2500.0);
uniform int cullingPassLoD;

in vec4 gWorldMatPart0[];
in vec4 gWorldMatPart1[];
in vec4 gWorldMatPart2[];

in mat3 gNormalMat[];


out vec4 instanceWorldViewProjMatCol0;
out vec4 instanceWorldViewProjMatCol1;
out vec4 instanceWorldViewProjMatCol2;
out vec4 instanceWorldViewProjMatCol3;

out vec3 instanceNormalMatCol0;
out vec3 instanceNormalMatCol1;
out vec3 instanceNormalMatCol2;


out vec3 instanceWorldViewMatCol0;
out vec3 instanceWorldViewMatCol1;
out vec3 instanceWorldViewMatCol2;
out vec3 instanceWorldViewMatCol3;


void main()
{
    vec3 instancePos = gWorldMatPart0[0].xyz*LoDInvariantBBoxCenter.x+vec3(gWorldMatPart0[0].w,gWorldMatPart1[0].xy)*LoDInvariantBBoxCenter.y+vec3(gWorldMatPart1[0].zw,gWorldMatPart2[0].x)*LoDInvariantBBoxCenter.z+gWorldMatPart2[0].yzw;
    instancePos = WorldMat[0]*instancePos.x+WorldMat[1]*instancePos.y+WorldMat[2]*instancePos.z+WorldMat[3];
    vec3 eyeToInstance = instancePos-eyePos;
    float distanceToInstance = dot(eyeToInstance,eyeToInstance);
    if (distanceToInstance<instanceLoDDistanceSQ[cullingPassLoD]||distanceToInstance>=instanceLoDDistanceSQ[cullingPassLoD+1])
        return;

    instanceWorldViewProjMatCol0 = ProjViewWorldMat[0]*gWorldMatPart0[0].x+ProjViewWorldMat[1]*gWorldMatPart0[0].y+ProjViewWorldMat[2]*gWorldMatPart0[0].z;
    instanceWorldViewProjMatCol1 = ProjViewWorldMat[0]*gWorldMatPart0[0].w+ProjViewWorldMat[1]*gWorldMatPart1[0].x+ProjViewWorldMat[2]*gWorldMatPart1[0].y;
    instanceWorldViewProjMatCol2 = ProjViewWorldMat[0]*gWorldMatPart1[0].z+ProjViewWorldMat[1]*gWorldMatPart1[0].w+ProjViewWorldMat[2]*gWorldMatPart2[0].x;
    instanceWorldViewProjMatCol3 = ProjViewWorldMat[0]*gWorldMatPart2[0].y+ProjViewWorldMat[1]*gWorldMatPart2[0].z+ProjViewWorldMat[2]*gWorldMatPart2[0].w+ProjViewWorldMat[3];

    ///Do frustum Culling
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

    instanceNormalMatCol0 = NormalMat[0]*gNormalMat[0][0].x+NormalMat[1]*gNormalMat[0][0].y+NormalMat[2]*gNormalMat[0][0].z;
    instanceNormalMatCol1 = NormalMat[0]*gNormalMat[0][1].x+NormalMat[1]*gNormalMat[0][1].y+NormalMat[2]*gNormalMat[0][1].z;
    instanceNormalMatCol2 = NormalMat[0]*gNormalMat[0][2].x+NormalMat[1]*gNormalMat[0][2].y+NormalMat[2]*gNormalMat[0][2].z;

    instanceWorldViewMatCol0 = -ViewWorldMat[0]*gWorldMatPart0[0].x-ViewWorldMat[1]*gWorldMatPart0[0].y-ViewWorldMat[2]*gWorldMatPart0[0].z;
    instanceWorldViewMatCol1 = -ViewWorldMat[0]*gWorldMatPart0[0].w-ViewWorldMat[1]*gWorldMatPart1[0].x-ViewWorldMat[2]*gWorldMatPart1[0].y;
    instanceWorldViewMatCol2 = -ViewWorldMat[0]*gWorldMatPart1[0].z-ViewWorldMat[1]*gWorldMatPart1[0].w-ViewWorldMat[2]*gWorldMatPart2[0].x;
    instanceWorldViewMatCol3 = -ViewWorldMat[0]*gWorldMatPart2[0].y-ViewWorldMat[1]*gWorldMatPart2[0].z-ViewWorldMat[2]*gWorldMatPart2[0].w-ViewWorldMat[3];


    EmitVertex();
    EndPrimitive();
}
