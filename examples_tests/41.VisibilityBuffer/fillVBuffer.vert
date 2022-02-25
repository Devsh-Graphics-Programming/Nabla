#version 460 core

#include "common.glsl"

layout (push_constant) uniform Block 
{
  uint dataBufferOffset;
} pc;

layout(location = 0) flat out uint drawGUID;
#include <nbl/builtin/glsl/barycentric/vert.glsl>

#include <nbl/builtin/glsl/utils/transform.glsl>
void main()
{
    drawGUID = gl_DrawID+pc.dataBufferOffset;
    vec3 modelPos = nbl_glsl_fetchVtxPos(gl_VertexIndex,drawGUID);
    nbl_glsl_barycentric_vert_set(modelPos);
    gl_Position = nbl_glsl_pseudoMul4x4with3x1(CamData.params.MVP,modelPos);
}