#version 460 core

#extension GL_EXT_shader_16bit_storage : require

layout(location = 0) in flat uint drawGUID;

layout(set = 0, binding = 0, std430) restrict coherent buffer VisibleBuff
{
    uint16_t visible[];
} visibleBuff;

void main()
{
    visibleBuff.visible[drawGUID] = uint16_t(1);
}