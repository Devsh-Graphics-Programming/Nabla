
#include <nbl/builtin/glsl/utils/indirect_commands.glsl>

layout(set = 3, binding = 2, std430) restrict coherent buffer IndirectDraws
{
    nbl_glsl_DrawElementsIndirectCommand_t draws[];
}commandBuff[2];