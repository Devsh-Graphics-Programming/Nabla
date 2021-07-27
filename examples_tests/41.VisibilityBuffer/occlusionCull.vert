#version 460 core
#extension GL_EXT_shader_16bit_storage : require

layout(location = 0) in vec4 vMatRow0;
layout(location = 1) in vec4 vMatRow1;
layout(location = 2) in vec4 vMatRow2;
layout(location = 3) in vec4 vMatRow3;

layout(location = 0) out flat uint instanceID;

void main()
{
    const vec3 pos[8] = {
        vec3(0.0, 0.0, 0.0),
        vec3(1.0, 0.0, 0.0),
        vec3(1.0, 1.0, 0.0),
        vec3(0.0, 1.0, 0.0),

        vec3(0.0, 0.0, 1.0),
        vec3(1.0, 0.0, 1.0),
        vec3(1.0, 1.0, 1.0),
        vec3(0.0, 1.0, 1.0),
    };

    mat4 mvp = mat4(vMatRow0, vMatRow1, vMatRow2, vMatRow3);

    gl_Position = mvp * vec4(pos[gl_VertexIndex], 1.0);
    instanceID = gl_InstanceIndex;
}