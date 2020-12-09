// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define LINE_VERTEX_LIMIT 11184810u

#ifndef __cplusplus
#include "nbl/builtin/glsl/limits/numeric.glsl"
#include "nbl/builtin/glsl/utils/indirect_commands.glsl"
layout(set = 0, binding = 0) coherent buffer LineCount
{
    nbl_glsl_DrawArraysIndirectCommand_t lineDraw[2];
};
layout(set = 0, binding = 1) writeonly buffer Lines
{
    float linePoints[]; // 6 floats decribe a line, 3d start, 3d end
};
layout(set = 0, binding = 2) uniform GlobalUniforms
{
    uint mdiIndex;
    float intersectionPlaneSpacing;
};
#else
struct GlobalUniforms
{
    uint32_t mdiIndex;
    float intersectionPlaneSpacing;
};
#endif