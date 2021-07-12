#version 460 core

layout(location = 0) in vec3 vPos;
layout(location = 1) in vec4 vMatRow0;
layout(location = 2) in vec4 vMatRow1;
layout(location = 3) in vec4 vMatRow2;
layout(location = 4) in vec4 vMatRow3;
layout(location = 5) in uint vDrawGUID;

layout(location = 0) out flat uint drawGUID;

void main()
{
    mat4 mvp = mat4(vMatRow0, vMatRow1, vMatRow2, vMatRow3);
    //mvp = transpose(mvp);

    gl_Position = transpose(mvp) * vec4(vPos, 1.0);
    drawGUID = vDrawGUID;
}