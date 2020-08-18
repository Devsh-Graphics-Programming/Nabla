#define LINE_VERTEX_LIMIT 22369620u

#ifndef __cplusplus
#include "irr/builtin/glsl/limits/numeric.glsl"
#include "irr/builtin/glsl/utils/indirect_commands.glsl"
layout(set = 0, binding = 0) coherent buffer LineCount
{
    irr_glsl_DrawArraysIndirectCommand_t lineDraw;
};
layout(set = 0, binding = 1) writeonly buffer Lines
{
    float linePoints[]; // 6 floats decribe a line, 3d start, 3d end
};
layout(set = 0, binding = 2) uniform LevelCurveSettings
{
    float intersectionPlaneSpacing;
};
#endif