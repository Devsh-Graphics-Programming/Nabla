#ifndef _IRR_EXT_DRAW_3D_LINE_SHADERS_INCLUDED_
#define _IRR_EXT_DRAW_3D_LINE_SHADERS_INCLUDED_

namespace irr
{
namespace ext
{
namespace DebugDraw
{

static const char* Draw3DLineVertexShader =
"#version 330 core\n"

"layout(location = 0) in vec4 vPos;\n"
"layout(location = 1) in vec4 vCol;\n"

"uniform mat4 MVP;\n"
"out vec4 Color;\n"

"void main()\n"
"{\n"
"    gl_Position = MVP * vPos;\n"
"    Color = vCol;\n"
"}\n"
;

static const char* Draw3DLineFragmentShader =
"#version 330 core\n"

"in vec4 Color;\n"

"layout(location = 0) out vec4 pixelColor;\n"

"void main()\n"
"{\n"
"    pixelColor = Color;\n"
"}\n"
;

} // namespace DebugDraw
} // namespace ext
} // namespace irr

#endif // _IRR_EXT_DRAW_3D_LINE_SHADERS_INCLUDED_
