
#ifdef STRETCH_MIN
#define REDUCTION_OPERATOR min
#else // PAD_MAX
#define REDUCTION_OPERATOR max
#endif

layout(binding = 0, set = 0) uniform sampler2D sourceTexture;
layout(binding = 1, set = 0, MIP_IMAGE_FORMAT) uniform image2D outMips[MIPMAP_LEVELS_PER_PASS];

#include "nbl/builtin/glsl/utils/morton.glsl"

#define WORKGROUP_SIZE (WORKGROUP_X_AND_Y_SIZE * WORKGROUP_X_AND_Y_SIZE)

#if (WORKGROUP_X_AND_Y_SIZE == 32)
    #define DECODE_MORTON decodeMorton2d8b 
#else
    #define DECODE_MORTON decodeMorton2d4b
#endif

shared float sharedMem[WORKGROUP_SIZE * 2u];

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
        const float reducedVal = REDUCTION_OPERATOR(REDUCTION_OPERATOR(samples[0], samples[1]),REDUCTION_OPERATOR(samples[2], samples[3]));
        imageStore(outMips[0], ivec2(naturalOrder), vec4(reducedVal, 0.f, 0.f, 0.f));
        sharedMem[WORKGROUP_SIZE + gl_LocalInvocationIndex] = reducedVal;

        barrier();
        sharedMem[gl_LocalInvocationIndex] = sharedMem[WORKGROUP_SIZE+(bitfieldReverse(gl_LocalInvocationIndex) >> (32 - findMSB(WORKGROUP_SIZE)))];
        barrier();

        uint limit = WORKGROUP_SIZE >> 1u;
        for (int i=1; i < pc.thisPassMipCnt; i++)
        {
          if (gl_LocalInvocationIndex < limit)
            sharedMem[gl_LocalInvocationIndex] = REDUCTION_OPERATOR(sharedMem[gl_LocalInvocationIndex], sharedMem[gl_LocalInvocationIndex + limit]);
          barrier();
          limit >>= 1u;
          if (gl_LocalInvocationIndex < limit)
          {
            const float reducedVal = REDUCTION_OPERATOR(sharedMem[gl_LocalInvocationIndex], sharedMem[gl_LocalInvocationIndex + limit]);
            sharedMem[gl_LocalInvocationIndex] = reducedVal;
            imageStore(outMips[i], ivec2(base + morton) >> (i + i), vec4(reducedVal, 0.0, 0.0, 0.0));
          }
          barrier();
          limit >>= 1u;
        }

    }
}