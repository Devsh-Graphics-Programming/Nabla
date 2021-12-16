// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_VERT_INPUTS_DEFINED_
#define _NBL_VERT_INPUTS_DEFINED_
layout (location = 0) in vec3 vPos;
#ifndef _NO_UV
layout (location = 2) in vec2 vUV;
#endif
layout (location = 3) in vec3 vNormal;
#endif //_NBL_VERT_INPUTS_DEFINED_

#ifndef _NBL_VERT_OUTPUTS_DEFINED_
#define _NBL_VERT_OUTPUTS_DEFINED_
layout (location = 0) out vec3 LocalPos;
layout (location = 1) out vec3 ViewPos;
layout (location = 2) out vec3 Normal;
#ifndef _NO_UV
layout (location = 3) out vec2 UV;
#endif
#endif //_NBL_VERT_OUTPUTS_DEFINED_

#include <nbl/builtin/glsl/utils/common.glsl>
#include <nbl/builtin/glsl/utils/transform.glsl>

#ifndef _NBL_BASIC_VTX_ATTRIB_FETCH_FUCTIONS_DEFINED_
#ifndef _NBL_POS_FETCH_FUNCTION_DEFINED
vec3 nbl_glsl_fetchVtxPos(in uint vtxID, in uint drawID)    { return vPos; }
#endif
#if !defined(_NBL_UV_FETCH_FUNCTION_DEFINED) && !defined(_NO_UV)
vec2 nbl_glsl_fetchVtxUV(in uint vtxID, in uint drawID)     { return vUV; }
#endif
#ifndef _NBL_NORMAL_FETCH_FUNCTION_DEFINED
vec3 nbl_glsl_fetchVtxNormal(in uint vtxID, in uint drawID) { return vNormal; }
#endif
#endif

#ifndef _NBL_VERT_SET1_BINDINGS_DEFINED_
#define _NBL_VERT_SET1_BINDINGS_DEFINED_
layout (set = 1, binding = 0, row_major, std140) uniform UBO
{
    nbl_glsl_SBasicViewParameters params;
} CamData;
#endif //_NBL_VERT_SET1_BINDINGS_DEFINED_

#ifndef _NBL_VERT_MAIN_DEFINED_
#define _NBL_VERT_MAIN_DEFINED_
void main()
{
    LocalPos = nbl_glsl_fetchVtxPos(gl_VertexIndex, gl_DrawID);
    gl_Position = nbl_glsl_pseudoMul4x4with3x1(CamData.params.MVP, LocalPos);
    ViewPos = nbl_glsl_pseudoMul3x4with3x1(CamData.params.MV, LocalPos);
    mat3 normalMat = nbl_glsl_SBasicViewParameters_GetNormalMat(CamData.params.NormalMatAndEyePos);
    Normal = normalMat*normalize(nbl_glsl_fetchVtxNormal(gl_VertexIndex, gl_DrawID));
#ifndef _NO_UV
    UV = nbl_glsl_fetchVtxUV(gl_VertexIndex, gl_DrawID);
#endif
}
#endif //_NBL_VERT_MAIN_DEFINED_