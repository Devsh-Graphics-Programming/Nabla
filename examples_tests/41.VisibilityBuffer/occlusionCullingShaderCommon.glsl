//TODO: rename
//TODO: move buffers, that are not used by multiple shaders

#include <nbl/builtin/glsl/utils/indirect_commands.glsl>

#ifdef ENABLE_CUBE_COMMAND_BUFFER

layout(set=CUBE_COMMAND_BUFF_SET, binding=CUBE_COMMAND_BUFF_BINDING, std430) restrict coherent buffer CubeIndirectDraw
{
    nbl_glsl_DrawElementsIndirectCommand_t draw;
} cubeIndirectDraw;

#endif

#ifdef ENABLE_FRUSTUM_CULLED_COMMAND_BUFFER

layout(set=FRUSTUM_CULLED_COMMAND_BUFF_SET, binding=FRUSTUM_CULLED_COMMAND_BUFF_BINDING, std430) restrict buffer FrustumIndirectDraws
{
    nbl_glsl_DrawElementsIndirectCommand_t draws[];
} frustumCommandBuff;

#endif

#ifdef ENABLE_OCCLUSION_CULLED_COMMAND_BUFFER

layout(set=OCCLUSION_CULLED_COMMAND_BUFF_SET, binding=OCCLUSION_CULLED_COMMAND_BUFF_BINDING, std430) restrict buffer OcclusionIndirectDraws
{
    nbl_glsl_DrawElementsIndirectCommand_t draws[];
} occlusionCommandBuff;

#endif

#ifdef ENABLE_VISIBLE_BUFFER

layout(set = VISIBLE_BUFF_SET, binding = VISIBLE_BUFF_BINDING, std430) restrict buffer VisibleBuff
{
    uint16_t visible[];
} visibleBuff;

#endif

#ifdef ENABLE_CUBE_DRAW_GUID_BUFFER

layout(set=CUBE_DRAW_GUID_BUFF_SET, binding=CUBE_DRAW_GUID_BUFF_BINDING, std430) restrict buffer CubeDrawGUID
{
    uint drawGUID[];
} cubeDrawGUIDBuffer;

#endif

#ifdef ENABLE_OCCLUSION_DISPATCH_INDIRECT_BUFFER

layout(set=OCCLUSION_DISPATCH_INDIRECT_BUFF_SET, binding=OCCLUSION_DISPATCH_INDIRECT_BUFF_BINDING, std430) restrict coherent buffer OcclusionDispatchIndirectBuffer
{
    nbl_glsl_DispatchIndirectCommand_t di;
} occlusionDispatchIndirect;

#endif