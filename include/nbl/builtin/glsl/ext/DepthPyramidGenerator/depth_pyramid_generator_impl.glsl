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

REDUCED_VAL_T loadFromImage(in uint mipIdx, in uvec2 coords)
{
    vec4 pix = imageLoad(outMips[mipIdx], ivec2(coords));

#ifndef REDUCION_OP_BOTH
  return pix.x;
#else
  return pix.xy;
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

void calcMipsFromSharedMemoryData(in uint mipToCalcCnt, in uint firstOutputMipIdx, in uvec2 base, in uvec2 morton)
{
    copySharedMemValue(gl_LocalInvocationIndex, WORKGROUP_SIZE + (bitfieldReverse(gl_LocalInvocationIndex) >> (32 - findMSB(WORKGROUP_SIZE))));
    barrier();
            
    uint limit = WORKGROUP_SIZE >> 1u;
    for (int i = 1; i < mipToCalcCnt; i++)
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
          storeReducedValToImage(firstOutputMipIdx + i, (base >> i) + morton, getValFromSharedMemory(bitfieldReverse(gl_LocalInvocationIndex) >> uint(32 - findMSB(1024) + i + i)));
        }
        barrier();
        limit >>= 1u;
    }
}

//TODO: rename
void calcMipsFromSharedMemoryData2(in uint mipToCalcCnt, in uint firstOutputMipIdx, in uint passFirstMipPixelCnt, in uvec2 currImgExtent)
{
    //TODO: optimize
    uint limit = passFirstMipPixelCnt >> 1u;
    currImgExtent >>= 1u;
    for (int i = 1; i < mipToCalcCnt; i++)
    {
        if (gl_LocalInvocationIndex < limit)
        {
            ivec2 coords;
            coords.x = int(gl_LocalInvocationIndex) % int(currImgExtent.x);
            coords.y = int(gl_LocalInvocationIndex) / int(currImgExtent.x);
            coords <<= 1;

            currImgExtent <<= 1u;

            const uint p0Index = coords.y * currImgExtent.x + coords.x;
            const uint p1Index = p0Index + currImgExtent.x;
            const REDUCED_VAL_T reducedVal0 = reduceValFromSharedMemory(p0Index, p0Index + 1u);
            const REDUCED_VAL_T reducedVal1 = reduceValFromSharedMemory(p1Index, p1Index + 1u);
            const REDUCED_VAL_T reducedVal = REDUCTION_OPERATOR_2(reducedVal0, reducedVal1);
            storeReducedValToSharedMemory(gl_LocalInvocationIndex, reducedVal);
            storeReducedValToImage(firstOutputMipIdx + i, uvec2(coords) >> 1u, reducedVal);

            currImgExtent >>= 1u;
        }
        barrier();
        limit >>= 1u;
        currImgExtent >>= 1u;
    }
}

void main()
{
    for (uint metaZLayer = 0u; ; metaZLayer++)
    {
        const uvec3 virtualWorkGroupID = nbl_glsl_depthPyramid_scheduler_getWork(metaZLayer);
        
        // main dispatch
        if(metaZLayer == 0u)
        {
            const uint mainPassFirstMipPixelCnt = pc.data.mainDispatchFirstMipExtent.x * pc.data.mainDispatchFirstMipExtent.y;

            if(mainPassFirstMipPixelCnt >= WORKGROUP_SIZE)
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
                const REDUCED_VAL_T reducedVal = REDUCTION_OPERATOR_2(REDUCTION_OPERATOR(samples[0], samples[1]), REDUCTION_OPERATOR(samples[2], samples[3]));
                storeReducedValToImage(0, naturalOrder, reducedVal);

                storeReducedValToSharedMemory(WORKGROUP_SIZE + gl_LocalInvocationIndex, reducedVal);
                barrier();
                calcMipsFromSharedMemoryData(pc.data.mainDispatchMipCnt, 0u, base, morton);
            }
            else
            {
                if(gl_LocalInvocationIndex < mainPassFirstMipPixelCnt)
                {
                    ivec2 coords;
                    coords.x = int(gl_LocalInvocationIndex) % int(pc.data.mainDispatchFirstMipExtent.x);
                    coords.y = int(gl_LocalInvocationIndex) / int(pc.data.mainDispatchFirstMipExtent.x);

                    const vec2 uv = (vec2(coords) + vec2(0.5)) / vec2(textureSize(sourceTexture, 0));

                    const vec4 samples = textureGather(sourceTexture, uv); // border color set to far value (or far,near if doing two channel reduction)
                    //TODO: it will work only for first dispatch if `EMGO_BOTH` is set (`textureGather` argument `comp` is implicitly set to 0)
                    //possible solution send `sourceImageIsDepthOriginalDepthBuffer` flag, if this flag is set to 1 then act accordingly

                    //TODO: are samples fetched incorrectly?
                    const REDUCED_VAL_T reducedVal = REDUCTION_OPERATOR_2(REDUCTION_OPERATOR(samples[0], samples[1]), REDUCTION_OPERATOR(samples[2], samples[3]));
                    storeReducedValToImage(0, coords, reducedVal);

                    storeReducedValToSharedMemory(gl_LocalInvocationIndex, reducedVal);
                    barrier();
                    calcMipsFromSharedMemoryData2(pc.data.mainDispatchMipCnt, 0u, mainPassFirstMipPixelCnt, pc.data.mainDispatchFirstMipExtent);
                }
            }
            
        }
        // virtual dispatch
        else
        {
            const uint virutalPassFirstMipPixelCnt = pc.data.virtualDispatchFirstMipExtent.x * pc.data.virtualDispatchFirstMipExtent.y;

            if(virutalPassFirstMipPixelCnt >= WORKGROUP_SIZE)
            {
                const uvec2 base = virtualWorkGroupID.xy * gl_WorkGroupSize.xy;
                const uvec2 morton = DECODE_MORTON(gl_LocalInvocationIndex);

                uvec2 naturalOrder = base + morton;
                naturalOrder <<= 1u;

                const uint srcImgIdx = pc.data.mainDispatchMipCnt - 1u;
                REDUCED_VAL_T p[4];
                p[0] = loadFromImage(srcImgIdx, naturalOrder);
                p[1] = loadFromImage(srcImgIdx, naturalOrder + uvec2(1, 0));
                p[2] = loadFromImage(srcImgIdx, naturalOrder + uvec2(0, 1));
                p[3] = loadFromImage(srcImgIdx, naturalOrder + uvec2(1, 1));

                const REDUCED_VAL_T reducedVal = REDUCTION_OPERATOR_2(REDUCTION_OPERATOR_2(p[0], p[1]), REDUCTION_OPERATOR_2(p[2], p[3]));

                naturalOrder >>= 1u;

                const uint dstImgIdx = pc.data.mainDispatchMipCnt;
                storeReducedValToImage(dstImgIdx, uvec2(naturalOrder), reducedVal);

                storeReducedValToSharedMemory(WORKGROUP_SIZE + gl_LocalInvocationIndex, reducedVal);
                barrier();
                calcMipsFromSharedMemoryData(pc.data.virtualDispatchMipCnt, pc.data.mainDispatchMipCnt, base, morton);
            }
            else
            {
                if(gl_LocalInvocationIndex < virutalPassFirstMipPixelCnt)
                {
                    ivec2 coords;
                    coords.x = int(gl_LocalInvocationIndex) % int(pc.data.virtualDispatchFirstMipExtent.x);
                    coords.y = int(gl_LocalInvocationIndex) / int(pc.data.virtualDispatchFirstMipExtent.x);
                    coords <<= 1;

                    const uint srcImgIdx = pc.data.mainDispatchMipCnt - 1u;
                    REDUCED_VAL_T p[4];
                    p[0] = loadFromImage(srcImgIdx, uvec2(coords));
                    p[1] = loadFromImage(srcImgIdx, uvec2(coords) + uvec2(1, 0));
                    p[2] = loadFromImage(srcImgIdx, uvec2(coords) + uvec2(0, 1));
                    p[3] = loadFromImage(srcImgIdx, uvec2(coords) + uvec2(1, 1));

                    const REDUCED_VAL_T reducedVal = REDUCTION_OPERATOR_2(REDUCTION_OPERATOR_2(p[0], p[1]), REDUCTION_OPERATOR_2(p[2], p[3]));

                    const uint dstImgIdx = pc.data.mainDispatchMipCnt;
                    storeReducedValToImage(dstImgIdx, uvec2(coords) / 2u, reducedVal);

                    storeReducedValToSharedMemory(gl_LocalInvocationIndex, reducedVal);
                    barrier();
                    calcMipsFromSharedMemoryData2(pc.data.virtualDispatchMipCnt, pc.data.mainDispatchMipCnt, virutalPassFirstMipPixelCnt, pc.data.virtualDispatchFirstMipExtent);
                }
            }
        }
        
        const bool shouldTerminate = nbl_glsl_depthPyramid_finalizeVirtualWorkgGroup(metaZLayer);
        if(shouldTerminate)
        {
          nbl_glsl_depthPyramid_resetAtomicCounters();
          return;
        }
    }
}

#endif