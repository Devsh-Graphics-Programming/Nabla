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

#define NBL_GLSL_TRANSFORM_TREE_POOL_DESCRIPTOR_SET 3
#define NBL_GLSL_TRANSFORM_TREE_POOL_NODE_GLOBAL_TRANSFORM_DESCRIPTOR_BINDING 0
// disable what we dont use
#define NBL_GLSL_TRANSFORM_TREE_POOL_NODE_PARENT_DESCRIPTOR_DECLARED
#define NBL_GLSL_TRANSFORM_TREE_POOL_NODE_RELATIVE_TRANSFORM_DESCRIPTOR_DECLARED
#define NBL_GLSL_TRANSFORM_TREE_POOL_NODE_MODIFIED_TIMESTAMP_DESCRIPTOR_DECLARED
#define NBL_GLSL_TRANSFORM_TREE_POOL_NODE_RECOMPUTED_TIMESTAMP_DESCRIPTOR_DECLARED
#define NBL_GLSL_TRANSFORM_TREE_POOL_NODE_NORMAL_MATRIX_DESCRIPTOR_DECLARED
#include <nbl/builtin/glsl/transform_tree/pool_descriptor_set.glsl>

#include <nbl/builtin/glsl/utils/transform.glsl>
void nbl_glsl_culling_lod_selection_initializePerViewPerInstanceData(out nbl_glsl_PerViewPerInstance_t pvpi, in uint instanceGUID)
{
    const mat4x3 world = nodeGlobalTransforms.data[instanceGUID];

    const vec3 toCam = pc.data.camPos-world[3];
    distanceSq = dot(toCam,toCam);

    pvpi.mvp = nbl_glsl_pseudoMul4x4with4x3(pc.data.viewProjMat,world);
}

uint nbl_glsl_lod_library_Table_getLoDUvec2Offset(in uint lodTableUvec4Offset, in uint lodID);
#include <nbl/builtin/glsl/lod_library/structs.glsl>
nbl_glsl_lod_library_DefaultLoDChoiceParams nbl_glsl_lod_library_DefaultInfo_getLoDChoiceParams(in uint lodInfoUvec2Offset);
uint nbl_glsl_culling_lod_selection_chooseLoD(in uint lodTableUvec4Offset, in uint lodCount)
{
    uint lodInfoUvec2Offset = 0xffffffffu;
    for (lodID=0u; lodID<lodCount; lodID++)
    {
        const uint nextLoD = nbl_glsl_lod_library_Table_getLoDUvec2Offset(lodTableUvec4Offset,lodID);
        const float threshold = nbl_glsl_lod_library_DefaultInfo_getLoDChoiceParams(nextLoD).distanceSqAtReferenceFoV;
        if (distanceSq>threshold*pc.data.fovDilationFactor)
            break;
        lodInfoUvec2Offset = nextLoD;
    }
    return lodInfoUvec2Offset;
}

void nbl_glsl_culling_lod_selection_finalizePerViewPerInstanceData(inout nbl_glsl_PerViewPerInstance_t pvpi, in uint instanceGUID)
{
    pvpi.lod = lodID-1u;
    // we could compute and store more stuff, like normal matrix, etc.
}