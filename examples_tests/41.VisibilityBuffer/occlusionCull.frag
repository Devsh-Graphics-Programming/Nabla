#version 460 core
#extension GL_EXT_shader_16bit_storage : require

layout(early_fragment_tests) in;

layout(location = 0) in flat uint instanceID;

#define VISIBLE_BUFF_SET 0
#define VISIBLE_BUFF_BINDING 0
#include "occlusionCullingShaderCommon.glsl"

void main()
{
    visibleBuff.visible[instanceID] = uint16_t(1u);
}