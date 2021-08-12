
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

layout(binding = 0, set = 0) uniform sampler2D sourceTexture;
layout(binding = 1, set = 0, MIP_IMAGE_FORMAT) uniform image2D outMips[MIPMAP_LEVELS_PER_PASS];

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

void copySharedMemValue(in uint dstIdx, in uint srcIdx)
{
  sharedMemR[dstIdx] = sharedMemR[srcIdx];
#ifdef REDUCION_OP_BOTH
  sharedMemG[dstIdx] = sharedMemG[srcIdx];
#endif
}

#include "nbl/builtin/glsl/utils/morton.glsl"

#if (WORKGROUP_X_AND_Y_SIZE == 32)
    #define DECODE_MORTON decodeMorton2d8b 
#else
    #define DECODE_MORTON decodeMorton2d4b
#endif

layout(push_constant) uniform PushConstants
{
    uint thisPassMipCnt;
} pc;

void main()
{
    const uvec2 base = gl_WorkGroupID.xy * gl_WorkGroupSize.xy;
    const uvec2 morton = DECODE_MORTON(gl_LocalInvocationIndex);

    {
        const uvec2 naturalOrder = base + morton;
        #ifdef STRETCH_MIN
        const vec2 uv = (vec2(naturalOrder) + vec2(0.5)) / vec2(gl_NumWorkGroups.xy*gl_WorkGroupSize.xy); 
        #else // PAD MAX
        const vec2 uv = (vec2(naturalOrder) + vec2(0.5)) / vec2(textureSize(sourceTexture, 0));
        #endif
        const vec4 samples = textureGather(sourceTexture, uv); // border color set to far value (or far,near if doing two channel reduction)
        const REDUCED_VAL_T reducedVal = REDUCTION_OPERATOR_2(REDUCTION_OPERATOR(samples[0], samples[1]),REDUCTION_OPERATOR(samples[2], samples[3]));
        storeReducedValToImage(0, naturalOrder, reducedVal);

        #ifndef REDUCION_OP_BOTH
        sharedMemR[WORKGROUP_SIZE + gl_LocalInvocationIndex] = reducedVal;
        #else
        sharedMemR[WORKGROUP_SIZE + gl_LocalInvocationIndex] = reducedVal.x;
        sharedMemG[WORKGROUP_SIZE + gl_LocalInvocationIndex] = reducedVal.y;
        #endif

        barrier();
        copySharedMemValue(gl_LocalInvocationIndex, WORKGROUP_SIZE+(bitfieldReverse(gl_LocalInvocationIndex) >> (32 - findMSB(WORKGROUP_SIZE))));
        barrier();

        uint limit = WORKGROUP_SIZE >> 1u;
        for (int i = 1; i < pc.thisPassMipCnt; i++)
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
            storeReducedValToImage(i, (base + morton), reducedVal);
          }
          barrier();
          limit >>= 1u;
        }

    }
}