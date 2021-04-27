#version 460 core

#include "common.glsl"

layout (push_constant) uniform Block 
{
    uint dataBufferOffset;
} pc;

layout(location = 0) flat out uint drawGUID;

#include <nbl/builtin/glsl/utils/transform.glsl>
void main()
{
    drawGUID = gl_DrawID+pc.dataBufferOffset;
    gl_Position = nbl_glsl_pseudoMul4x4with3x1(CamData.params.MVP,nbl_glsl_fetchVtxPos(gl_VertexIndex,drawGUID));
}