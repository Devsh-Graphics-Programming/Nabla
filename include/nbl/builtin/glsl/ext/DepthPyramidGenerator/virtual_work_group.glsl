// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_GLSL_EXT_DEPTH_PYRAMID_GENERATOR_VIRTUAL_WORK_GROUP_INCLUDED_
#define _NBL_GLSL_EXT_DEPTH_PYRAMID_GENERATOR_VIRTUAL_WORK_GROUP_INCLUDED_

#include <nbl/builtin/glsl/ext/DepthPyramidGenerator/common.glsl>

shared uvec3 unmappedVirtualWorkGroupIDShared;
shared uint virtualWorkGroupIndexShared;
shared bool shouldTerminateShared;

layout(binding = 0, set = 0, std430) restrict coherent buffer VirtualWorkGroupBuffer
{
    uint workGroupsDispatched;
    uint workGroupsFinished;
}virtualWorkGroup;

layout(binding = 1, set = 0, std430) restrict readonly buffer VirtualWorkGroupData
{
    uvec2 zLayerWorkGroupDim[];
}virtualWorkGroupData;

// NOTE: it is writen solely for 8 image binding limit
uvec3 nbl_glsl_depthPyramid_scheduler_getWork(in uint metaZLayer)
{
    //TODO: in fact `metaZLayer` is just a bool, which indicates if work group belongs to the main dispatch (0) or virtual dispatch (1), probably I should rename it to make it less confusing
    if(metaZLayer == 0u)
    {
        if(gl_LocalInvocationIndex == 0u)
            virtualWorkGroupIndexShared = atomicAdd(virtualWorkGroup.workGroupsDispatched, 1);

        return uvec3(gl_WorkGroupID.xy, metaZLayer); // TODO: why do I return metaZLayer again?
    }
    else
    {
        if(gl_LocalInvocationIndex == 0u)
        {
            virtualWorkGroupIndexShared = atomicAdd(virtualWorkGroup.workGroupsDispatched, 1);

            const uint mainDispatchWGCnt = virtualWorkGroupData.zLayerWorkGroupDim[pc.data.virtualDispatchIndex].x * virtualWorkGroupData.zLayerWorkGroupDim[pc.data.virtualDispatchIndex].y;
            const uint virtualWorkGroupGlobalInvocationIndex = virtualWorkGroupIndexShared - mainDispatchWGCnt;

            unmappedVirtualWorkGroupIDShared.y = virtualWorkGroupGlobalInvocationIndex / virtualWorkGroupData.zLayerWorkGroupDim[pc.data.virtualDispatchIndex + metaZLayer].x;
            unmappedVirtualWorkGroupIDShared.x = virtualWorkGroupGlobalInvocationIndex - unmappedVirtualWorkGroupIDShared.y * virtualWorkGroupData.zLayerWorkGroupDim[pc.data.virtualDispatchIndex + metaZLayer].x;
            unmappedVirtualWorkGroupIDShared.z = metaZLayer;
        }
        barrier();

        if(metaZLayer > 0u)
        {
            uint prevLvlWGCnt = virtualWorkGroupData.zLayerWorkGroupDim[pc.data.virtualDispatchIndex].x * virtualWorkGroupData.zLayerWorkGroupDim[pc.data.virtualDispatchIndex].y;
            while(virtualWorkGroup.workGroupsFinished < prevLvlWGCnt); //spin lock
        }

        return unmappedVirtualWorkGroupIDShared;
    }
}

bool nbl_glsl_depthPyramid_finalizeVirtualWorkgGroup(in uint metaZLayer)
{
    if(gl_LocalInvocationIndex == 0u)
    {
        if(metaZLayer >= pc.data.maxMetaZLayerCnt - 1u)
        {
            //atomicAdd(virtualWorkGroup.workGroupsFinished, 1u);
            shouldTerminateShared = true;
        }
        else
        {
            atomicAdd(virtualWorkGroup.workGroupsFinished, 1u);

            uint thisLvlWGCnt = virtualWorkGroupData.zLayerWorkGroupDim[pc.data.virtualDispatchIndex + metaZLayer].x * virtualWorkGroupData.zLayerWorkGroupDim[pc.data.virtualDispatchIndex + metaZLayer].y;
            uint nextLvlWGCnt = virtualWorkGroupData.zLayerWorkGroupDim[pc.data.virtualDispatchIndex + metaZLayer + 1u].x * virtualWorkGroupData.zLayerWorkGroupDim[pc.data.virtualDispatchIndex + metaZLayer + 1u].y;
            if(virtualWorkGroup.workGroupsFinished + nextLvlWGCnt > nextLvlWGCnt)
                shouldTerminateShared = false;
            else
                shouldTerminateShared = true;
        }
    }
    barrier();

    return shouldTerminateShared;
}

#endif