#version 460 core

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec4 vMatRow0;
layout(location = 2) in vec4 vMatRow1;
layout(location = 3) in vec4 vMatRow2;
layout(location = 4) in vec4 vMatRow3;
layout(location = 5) in uint vDrawGUID; //TODO: remove it

layout(location = 0) out flat uint instanceID;

void main()
{
    mat4 mvp = mat4(vMatRow0, vMatRow1, vMatRow2, vMatRow3);

    gl_Position = transpose(mvp) * vec4(vPos, 1.0);
    instanceID = gl_InstanceIndex;
}