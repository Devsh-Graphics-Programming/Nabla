#version 460 core

layout(push_constant, row_major) uniform PushConstants
{
	mat4 vp;
} pc;

layout(location = 0) in vec4 vPos;
layout(location = 2) in vec2 vTexCoord;
layout(location = 3) in vec3 vNormal;

layout(location = 0) out vec3 worldPos;
layout(location = 1) out vec2 texCoord;
layout(location = 2) out vec3 normal;
layout(location = 3) flat out uint drawID;

void main()
{
    gl_Position = pc.vp * vPos;
    
    drawID = gl_DrawID;
    worldPos = vec3(vPos);
    texCoord = vTexCoord;
    normal = normalize(vNormal);
}
