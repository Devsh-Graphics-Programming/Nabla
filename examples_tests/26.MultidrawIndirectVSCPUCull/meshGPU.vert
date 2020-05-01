#version 430 core
#extension ARB_shader_draw_parameters : require

#include "../commonVertexShader.glsl"
#include "../commonIndirect.glsl"

layout(set = 1, binding = 1, std430, row_major) restrict readonly buffer CulledDraws
{
    DrawElementsIndirectCommand_t draws;
};

void main()
{
    uint index = draws.objectUUID[gl_DrawIDARB];
    impl(index);
}

