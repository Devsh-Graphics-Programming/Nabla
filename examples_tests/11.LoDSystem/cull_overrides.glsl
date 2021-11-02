// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "common.glsl"
layout(push_constant, row_major) uniform PushConstants
{
    CullPushConstants_t data;
} pc;

uint nbl_glsl_culling_lod_selection_getInstanceCount()
{
    return pc.data.instanceCount;
}

// some globals to pass state around
float distanceSq;
uint lodID;

void nbl_glsl_culling_lod_selection_initializePerViewPerInstanceData(out nbl_glsl_PerViewPerInstance_t pvpi, in uint instanceGUID)
{
    mat4 world; // TODO: Use TT
    world[0] = vec4(1.f, 0.f, 0.f, 0.f);
    world[1] = vec4(0.f, 1.f, 0.f, 0.f);
    world[2] = vec4(0.f, 0.f, 1.f, 0.f);
    world[3] = vec4(0.f, float(instanceGUID) * 6.f, 0.f, 1.f);

    const vec3 toCam = pc.data.camPos-world[3].xyz;
    distanceSq = dot(toCam,toCam);

    pvpi.mvp = pc.data.viewProjMat*world;
}

uint nbl_glsl_lod_library_Table_getLoDUvec4Offset(in uint lodTableUvec4Offset, in uint lodID);
#include <nbl/builtin/glsl/lod_library/structs.glsl>
nbl_glsl_lod_library_DefaultLoDChoiceParams nbl_glsl_lod_library_DefaultInfo_getLoDChoiceParams(in uint lodInfoUvec4Offset);
uint nbl_glsl_culling_lod_selection_chooseLoD(in uint lodTableUvec4Offset, in uint lodCount)
{
    uint lodInfoUvec4Offset = 0xffffffffu;
    for (lodID=0u; lodID<lodCount; lodID++)
    {
        const uint nextLoD = nbl_glsl_lod_library_Table_getLoDUvec4Offset(lodTableUvec4Offset,lodID);
        const float threshold = nbl_glsl_lod_library_DefaultInfo_getLoDChoiceParams(nextLoD).distanceSqAtReferenceFoV;
        if (distanceSq>threshold*pc.data.fovDilationFactor)
            break;
        lodInfoUvec4Offset = nextLoD;
    }
    return lodInfoUvec4Offset;
}

void nbl_glsl_culling_lod_selection_finalizePerViewPerInstanceData(inout nbl_glsl_PerViewPerInstance_t pvpi, in uint instanceGUID)
{
    pvpi.lod = lodID-1u;
    // we could compute and store more stuff, like normal matrix, etc.
}