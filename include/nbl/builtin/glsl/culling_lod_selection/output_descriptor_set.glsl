#ifndef _NBL_GLSL_CULLING_LOD_SELECTION_OUTPUT_DESCRIPTOR_SET_GLSL_INCLUDED_
#define _NBL_GLSL_CULLING_LOD_SELECTION_OUTPUT_DESCRIPTOR_SET_GLSL_INCLUDED_


#ifndef NBL_GLSL_CULLING_LOD_SELECTION_OUTPUT_DESCRIPTOR_SET
#define NBL_GLSL_CULLING_LOD_SELECTION_OUTPUT_DESCRIPTOR_SET 2
#endif

#ifndef NBL_GLSL_CULLING_LOD_SELECTION_DRAW_CALLS_DESCRIPTOR_BINDING
#define NBL_GLSL_CULLING_LOD_SELECTION_DRAW_CALLS_DESCRIPTOR_BINDING 0
layout(
    set = NBL_GLSL_CULLING_LOD_SELECTION_OUTPUT_DESCRIPTOR_SET,
    binding = NBL_GLSL_CULLING_LOD_SELECTION_DRAW_CALLS_DESCRIPTOR_BINDING
) NBL_GLSL_CULLING_LOD_SELECTION_DRAW_CALLS_DESCRIPTOR_QUALIFIERS buffer DrawCalls
{
    uint data[];
} drawCalls;

uint nbl_glsl_culling_lod_selection_drawCallInstanceCountIncr(in uint drawCallDWORDOffset)
{
    return atomicAdd(drawCalls.data[drawCallDWORDOffset+1u],1u);
}

void nbl_glsl_culling_lod_selection_drawCallSetDWORD(in uint DWORDOffset, in uint value)
{
    drawCalls.data[DWORDOffset] = value;
}
void nbl_glsl_culling_lod_selection_drawArraysSetBaseInstance(in uint drawCallDWORDOffset, in uint baseInstance)
{
    drawCalls.data[drawCallDWORDOffset+3u] = baseInstance;
}
void nbl_glsl_culling_lod_selection_drawElementsSetBaseInstance(in uint drawCallDWORDOffset, in uint baseInstance)
{
    drawCalls.data[drawCallDWORDOffset+4u] = baseInstance;
}

uint nbl_glsl_culling_lod_selection_drawCallGetDWORD(in uint DWORDOffset)
{
    return drawCalls.data[DWORDOffset];
}
uint nbl_glsl_culling_lod_selection_drawArraysGetBaseInstance(in uint drawCallDWORDOffset)
{
    return nbl_glsl_culling_lod_selection_drawCallGetDWORD(drawCallDWORDOffset+3u);
}
uint nbl_glsl_culling_lod_selection_drawElementsGetBaseInstance(in uint drawCallDWORDOffset)
{
    return nbl_glsl_culling_lod_selection_drawCallGetDWORD(drawCallDWORDOffset+4u);
}
#endif

#ifndef NBL_GLSL_CULLING_LOD_SELECTION_PER_VIEW_PER_INSTANCE_DESCRIPTOR_BINDING
#define NBL_GLSL_CULLING_LOD_SELECTION_PER_VIEW_PER_INSTANCE_DESCRIPTOR_BINDING 1
#ifndef nbl_glsl_PerViewPerInstance_t
#error "nbl_glsl_PerViewPerInstance_t must be defined!"
#endif
layout(
    set = NBL_GLSL_CULLING_LOD_SELECTION_OUTPUT_DESCRIPTOR_SET,
    binding = NBL_GLSL_CULLING_LOD_SELECTION_PER_VIEW_PER_INSTANCE_DESCRIPTOR_BINDING
) NBL_GLSL_CULLING_LOD_SELECTION_PER_VIEW_PER_INSTANCE_DESCRIPTOR_QUALIFIERS buffer PerViewPerInstance
{
    nbl_glsl_PerViewPerInstance_t data[];
} perViewPerInstance;
#endif

#ifndef NBL_GLSL_CULLING_LOD_SELECTION_PER_INSTANCE_REDIRECT_ATTRS_DESCRIPTOR_BINDING
#define NBL_GLSL_CULLING_LOD_SELECTION_PER_INSTANCE_REDIRECT_ATTRS_DESCRIPTOR_BINDING 2
layout(
    set = NBL_GLSL_CULLING_LOD_SELECTION_OUTPUT_DESCRIPTOR_SET,
    binding = NBL_GLSL_CULLING_LOD_SELECTION_PER_INSTANCE_REDIRECT_ATTRS_DESCRIPTOR_BINDING
) restrict writeonly buffer PerInstanceRedirectAttrs
{
    uvec2 data[]; // <instanceGUID,perViewPerInstanceID>
} perInstanceRedirectAttrs;
#endif

#ifndef NBL_GLSL_CULLING_LOD_SELECTION_DRAW_CALL_COUNTS_DESCRIPTOR_BINDING
#define NBL_GLSL_CULLING_LOD_SELECTION_DRAW_CALL_COUNTS_DESCRIPTOR_BINDING 3
layout(
    set = NBL_GLSL_CULLING_LOD_SELECTION_OUTPUT_DESCRIPTOR_SET,
    binding = NBL_GLSL_CULLING_LOD_SELECTION_DRAW_CALL_COUNTS_DESCRIPTOR_BINDING
) NBL_GLSL_CULLING_LOD_SELECTION_DRAW_CALL_COUNTS_DESCRIPTOR_QUALIFIERS buffer DrawCallCounts
{
    uint data[];
} drawCallCounts;
#endif


#endif