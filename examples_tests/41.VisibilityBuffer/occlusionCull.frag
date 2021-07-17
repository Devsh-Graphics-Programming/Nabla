#version 460 core

layout(location = 0) in flat uint instanceID;

#define ENABLE_VISIBLE_BUFFER
#define VISIBLE_BUFF_SET 0
#define VISIBLE_BUFF_BINDING 0
#include "occlusionCullingShaderCommon.glsl"

void main()
{
    visibleBuff.visible[instanceID] = uint16_t(1);
}