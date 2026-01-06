#ifndef _NBL_GLSL_CULLING_LOD_SELECTION_POTENTIALLY_VISIBLE_INSTANCE_DRAW_STRUCT_GLSL_INCLUDED_
#define _NBL_GLSL_CULLING_LOD_SELECTION_POTENTIALLY_VISIBLE_INSTANCE_DRAW_STRUCT_GLSL_INCLUDED_


#ifdef __cplusplus
#define uint uint32_t
#endif
struct nbl_glsl_culling_lod_selection_PotentiallyVisibleInstanceDraw_t
{
    uint perViewPerInstanceID;
    uint drawBaseInstanceDWORDOffset;
    uint instanceID;
}; // TODO: move
#ifdef __cplusplus
#undef uint
#endif


#endif