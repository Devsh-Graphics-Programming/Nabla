// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

//TODO: consistent tabs

#ifndef _NBL_GLSL_EXT_DEPTH_PYRAMID_GENERATOR_INCLUDED_
#define _NBL_GLSL_EXT_DEPTH_PYRAMID_GENERATOR_INCLUDED_

#ifdef REDUCION_OP_MIN
#define REDUCTION_OPERATOR min
#define REDUCTION_OPERATOR_2 min
#define REDUCED_VAL_T float
#elif defined(REDUCION_OP_MAX)
#define REDUCTION_OPERATOR max
#define REDUCTION_OPERATOR_2 max
#define REDUCED_VAL_T float
#elif defined(REDUCION_OP_BOTH)
#define REDUCTION_OPERATOR(a, b) vec2(min(a, b), max(a, b))
// TODO: rename
#define REDUCTION_OPERATOR_2(a, b) vec2(min(a.x, b.x), max(a.y, b.y))
#define REDUCED_VAL_T vec2
#endif

#include <nbl/builtin/glsl/ext/DepthPyramidGenerator/common.glsl>
#include <nbl/builtin/glsl/ext/DepthPyramidGenerator/virtual_work_group.glsl>

layout(binding = 2, set = 0) uniform sampler2D sourceTexture;

// for now every image2d will be read/write and coherent
layout(binding = 3, set = 0, MIP_IMAGE_FORMAT) uniform restrict coherent image2D outMips[MIPMAP_LEVELS_PER_PASS]; // MAX_MIP_LEVELS_PER_PASS - 1u

#define WORKGROUP_SIZE (WORKGROUP_X_AND_Y_SIZE * WORKGROUP_X_AND_Y_SIZE)

shared float sharedMemR[WORKGROUP_SIZE * 2u];
#ifdef REDUCION_OP_BOTH
shared float sharedMemG[WORKGROUP_SIZE * 2u];
#endif

void storeReducedValToImage(in uint mipIdx, in uvec2 coords, REDUCED_VAL_T reducedVal)
{
#ifndef REDUCION_OP_BOTH
  imageStore(outMips[mipIdx], ivec2(coords), vec4(reducedVal, 0.f, 0.f, 0.f));
#else
  imageStore(outMips[mipIdx], ivec2(coords), vec4(reducedVal, 0.f, 0.f));
#endif
}

void storeReducedValToSharedMemory(in uint idx, in REDUCED_VAL_T val)
{
#ifndef REDUCION_OP_BOTH
  sharedMemR[idx] = val;
#else
  sharedMemR[idx] = val.x;
  sharedMemG[idx] = val.y;
#endif
}

REDUCED_VAL_T reduceValFromSharedMemory(in uint val0Idx, in uint val1Idx)
{
#ifndef REDUCION_OP_BOTH
  return REDUCTION_OPERATOR(sharedMemR[val0Idx], sharedMemR[val1Idx]);
#else
  return REDUCTION_OPERATOR_2(REDUCTION_OPERATOR(sharedMemR[val0Idx], sharedMemR[val1Idx]),REDUCTION_OPERATOR(sharedMemG[val0Idx], sharedMemG[val1Idx]));
#endif
}

REDUCED_VAL_T getValFromSharedMemory(in uint idx)
{
#ifndef REDUCION_OP_BOTH
  return sharedMemR[idx];
#else
  return vec2(sharedMemR[idx], sharedMemG[idx]);
#endif
}

void copySharedMemValue(in uint dstIdx, in uint srcIdx)
{
  sharedMemR[dstIdx] = sharedMemR[srcIdx];
#ifdef REDUCION_OP_BOTH
  sharedMemG[dstIdx] = sharedMemG[srcIdx];
#endif
}

#include "nbl/builtin/glsl/utils/morton.glsl"

#if (WORKGROUP_X_AND_Y_SIZE == 32)
    #define DECODE_MORTON nbl_glsl_morton_decode2d8b 
#else
    #define DECODE_MORTON nbl_glsl_morton_decode2d4b
#endif

void main()
{
    for (uint metaZLayer = 0u; ; metaZLayer++)
    {
        const uvec3 virtualWorkGroupID = nbl_glsl_depthPyramid_scheduler_getWork(metaZLayer);
        
        if(metaZLayer == 0u)
        {
            const uvec2 base = virtualWorkGroupID.xy * gl_WorkGroupSize.xy;
            const uvec2 morton = DECODE_MORTON(gl_LocalInvocationIndex);

            const uvec2 naturalOrder = base + morton;
            #ifdef STRETCH_MIN
            const vec2 uv = (vec2(naturalOrder) + vec2(0.5)) / vec2(gl_NumWorkGroups.xy * gl_WorkGroupSize.xy); 
            #else // PAD MAX
            const vec2 uv = (vec2(naturalOrder) + vec2(0.5)) / vec2(textureSize(sourceTexture, 0));
            #endif
            const vec4 samples = textureGather(sourceTexture, uv); // border color set to far value (or far,near if doing two channel reduction)
            const REDUCED_VAL_T reducedVal = REDUCTION_OPERATOR_2(REDUCTION_OPERATOR(samples[0], samples[1]),REDUCTION_OPERATOR(samples[2], samples[3]));
            storeReducedValToImage(0, naturalOrder, reducedVal);

            storeReducedValToSharedMemory(WORKGROUP_SIZE + gl_LocalInvocationIndex, reducedVal);

            barrier();
            copySharedMemValue(gl_LocalInvocationIndex, WORKGROUP_SIZE+(bitfieldReverse(gl_LocalInvocationIndex) >> (32 - findMSB(WORKGROUP_SIZE))));
            barrier();

            uint limit = WORKGROUP_SIZE >> 1u;
            for (int i = 1; i < pc.data.mainDispatchMipCnt; i++)
            {
                if (gl_LocalInvocationIndex < limit)
                {
                  const REDUCED_VAL_T reducedVal = reduceValFromSharedMemory(gl_LocalInvocationIndex, gl_LocalInvocationIndex + limit);
                  storeReducedValToSharedMemory(gl_LocalInvocationIndex, reducedVal);
                }

                barrier();
                limit >>= 1u;
                if (gl_LocalInvocationIndex < limit)
                {
                  const REDUCED_VAL_T reducedVal = reduceValFromSharedMemory(gl_LocalInvocationIndex, gl_LocalInvocationIndex + limit);
                  storeReducedValToSharedMemory(gl_LocalInvocationIndex, reducedVal);
                  storeReducedValToImage(i, (base >> i) + morton, getValFromSharedMemory(bitfieldReverse(gl_LocalInvocationIndex) >> uint(32 - findMSB(1024) + i + i)));
                }
                barrier();
                limit >>= 1u;
            }
        }
        else
        {
            //if(gl_localInvocationIndex);
        }
        
        const bool shouldTerminate = nbl_glsl_depthPyramid_finalizeVirtualWorkgGroup(metaZLayer);
        if(shouldTerminate)
            return;
    }
}

#endif