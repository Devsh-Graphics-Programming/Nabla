// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

layout(push_constant) uniform PushConstants
{
    mat4 viewProjMat;
    uint cullableInstanceCount;
} pc;

uint nbl_glsl_culling_lod_selection_getInstanceCount()
{
    return pc.cullableInstanceCount;
}

void nbl_glsl_culling_lod_selection_initializePerViewPerInstanceData(out nbl_glsl_PerViewPerInstance_t pvpi, in uint instanceGUID)
{
    mat4 world; // TODO: Use TT
    world[0] = vec4(1.f, 0.f, 0.f, 0.f);
    world[1] = vec4(0.f, 1.f, 0.f, 0.f);
    world[2] = vec4(0.f, 0.f, 1.f, 0.f);
    world[3] = vec4(0.f, float(instanceGUID) * 6.f, 0.f, 1.f);
    pvpi.mvp = pc.viewProjMat*world;
}

uint nbl_glsl_lod_library_Table_getLoDUvec4Offset(in uint lodTableUvec4Offset, in uint lodID);
uint nbl_glsl_culling_lod_selection_chooseLoD(in uint lodTableUvec4Offset, in uint lodCount)
{
    for (uint lodID=0u; lodID<lodCount; lodID++)
    {
        const uint lodInfoUvec4Offset = nbl_glsl_lod_library_Table_getLoDUvec4Offset(lodTableUvec4Offset,lodID);
        if (lodID==gl_LocalInvocationIndex) // TODO: choose LoD properly
            return lodInfoUvec4Offset;
    }
    return lodCount;
}

void nbl_glsl_culling_lod_selection_finalizePerViewPerInstanceData(inout nbl_glsl_PerViewPerInstance_t pvpi, in uint instanceGUID)
{
    // we could do stuff here like computing normal matrices, etc.
}