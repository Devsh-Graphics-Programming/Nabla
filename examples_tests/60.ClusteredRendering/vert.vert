#version 430 core

// Todo(achal):
// Following aren't required anymore and can be removed:
//      layout (location = 0) in vec3 LocalPos
//      layout (location = 2) out vec3 Normal;

layout (location = 4) flat out vec3 EyePos;
layout (location = 5) out vec3 Normal_WorldSpace;
layout (location = 6) flat out mat4 MVP;

#define _NBL_VERT_MAIN_DEFINED_
#include <nbl/builtin/shader/loader/mtl/vertex_impl.glsl>

void main()
{
    LocalPos = nbl_glsl_fetchVtxPos(gl_VertexIndex);
    gl_Position = nbl_glsl_pseudoMul4x4with3x1(CamData.params.MVP, LocalPos);
    ViewPos = nbl_glsl_pseudoMul3x4with3x1(CamData.params.MV, LocalPos);
    mat3 normalMat = nbl_glsl_SBasicViewParameters_GetNormalMat(CamData.params.NormalMatAndEyePos);
    Normal = normalMat*normalize(nbl_glsl_fetchVtxNormal(gl_VertexIndex));
#ifndef _NO_UV
    UV = nbl_glsl_fetchVtxUV(gl_VertexIndex);
#endif
    EyePos = nbl_glsl_SBasicViewParameters_GetEyePos(CamData.params.NormalMatAndEyePos);
    Normal_WorldSpace = normalize(nbl_glsl_fetchVtxNormal(gl_VertexIndex));

    MVP = CamData.params.MVP;
}