//TODO: rename

#include <nbl/builtin/glsl/utils/indirect_commands.glsl>

#ifdef ENABLE_CUBE_COMMAND_BUFFER

layout(set=CUBE_COMMAND_BUFF_SET, binding=CUBE_COMMAND_BUFF_BINDING, std430) restrict coherent buffer CubeIndirectDraw
{
    nbl_glsl_DrawElementsIndirectCommand_t draw;
} cubeIndirectDraw;

#endif

#ifdef ENABLE_FRUSTUM_CULLED_COMMAND_BUFFER

layout(set=FRUSTUM_CULLED_COMMAND_BUFF_SET, binding=FRUSTUM_CULLED_COMMAND_BUFF_BINDING, std430) restrict coherent buffer FrustumIndirectDraws
{
    nbl_glsl_DrawElementsIndirectCommand_t draws[];
} frustumCommandBuff;

#endif

#ifdef ENABLE_OCCLUSION_CULLED_COMMAND_BUFFER

layout(set=OCCLUSION_CULLED_COMMAND_BUFF_SET, binding=OCCLUSION_CULLED_COMMAND_BUFF_BINDING, std430) restrict coherent buffer OcclusionIndirectDraws
{
    nbl_glsl_DrawElementsIndirectCommand_t draws[];
} occlusionCommandBuff;

#endif

#ifdef ENABLE_ID_TO_MDI_ELEMENT_OFFSET_BUFFER

layout(set=ID_TO_MDI_ELEMENT_OFFSET_BUFF_SET, binding=ID_TO_MDI_ELEMENT_OFFSET_BUFF_BINDING, std430) restrict readonly buffer IdToMdiElementOffsetMap
{
    uint offsets[];
} idToMdiElementOffsetMap;

#endif

#ifdef ENABLE_VISIBLE_BUFFER
#extension GL_EXT_shader_16bit_storage : require

layout(set = VISIBLE_BUFF_SET, binding = VISIBLE_BUFF_BINDING, std430) restrict coherent buffer VisibleBuff
{
    uint16_t visible[];
} visibleBuff;

#endif

#ifdef ENABLE_CUBE_DRAW_GUID_BUFFER

layout(set=CUBE_DRAW_GUID_BUFF_SET, binding=CUBE_DRAW_GUID_BUFF_BINDING, std430) restrict coherent buffer CubeDrawGUID
{
    uint drawGUID[];
} cubeDrawGUIDBuffer;

#endif