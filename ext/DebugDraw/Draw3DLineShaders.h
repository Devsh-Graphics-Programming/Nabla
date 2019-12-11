#ifndef _IRR_EXT_DRAW_3D_LINE_SHADERS_INCLUDED_
#define _IRR_EXT_DRAW_3D_LINE_SHADERS_INCLUDED_

namespace irr
{
namespace ext
{
namespace DebugDraw
{

static const char* Draw3DLineVertexShader = R"===(
#version 430 core

layout(location = 0) in vec4 vPos;
layout(location = 1) in vec4 vCol;

layout( push_constant, row_major ) uniform Block {
	mat4 modelViewProj;
} PushConstants;
layout(location = 0) out vec4 Color;

void main()
{
    gl_Position = PushConstants.modelViewProj * vPos;
    Color = vCol;
}
)===";

static const char* Draw3DLineFragmentShader = R"===(
#version 430 core

layout(location = 0) in vec4 Color;

layout(location = 0) out vec4 pixelColor;

void main()
{
    pixelColor = Color;
}
)===";

} // namespace DebugDraw
} // namespace ext
} // namespace irr

#endif // _IRR_EXT_DRAW_3D_LINE_SHADERS_INCLUDED_
