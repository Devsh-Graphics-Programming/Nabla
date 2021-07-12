
#include <nbl/builtin/glsl/utils/indirect_commands.glsl>

#ifdef ENABLE_CUBE_COMMAND_BUFFER

#ifndef CUBE_COMMAND_BUFF_SET
#define CUBE_COMMAND_BUFF_SET 0
#endif

#ifndef CUBE_COMMAND_BUFF_BINDING
#define CUBE_COMMAND_BUFF_BINDING 0
#endif

layout(set=CUBE_COMMAND_BUFF_SET, binding=CUBE_COMMAND_BUFF_BINDING, std430) restrict coherent buffer CubeIndirectDraw
{
    nbl_glsl_DrawElementsIndirectCommand_t draw;
} cubeIndirectDraw;

#endif // ENABLE_CUBE_COMMAND_BUFFER

#ifdef ENABLE_COMMAND_BUFFER
#ifndef COMMAND_BUFF_SET
#define COMMAND_BUFF_SET 0
#endif

#ifndef COMMAND_BUFF_BINDING
#define COMMAND_BUFF_BINDING 1
#endif

layout(set=COMMAND_BUFF_SET, binding=COMMAND_BUFF_BINDING, std430) restrict coherent buffer IndirectDraws
{
    nbl_glsl_DrawElementsIndirectCommand_t draws[];
}commandBuff;

#endif // ENABLE_COMMAND_BUFFER