
#ifdef STRETCH_MIN
#define REDUCTION_OPERATOR min
#else // PAD_MAX
#define REDUCTION_OPERATOR max
#endif

layout(binding = 0, set = 0) uniform sampler2D sourceTexture;
layout(binding = 1, set = 0, MIP_IMAGE_FORMAT) uniform image2D outMips;

#include "nbl/builtin/glsl/utils/morton.glsl"

#if (WORKGROUP_X_AND_Y_SIZE == 32)
    #define DECODE_MORTON decodeMorton2d8b 
#else
    #define DECODE_MORTON decodeMorton2d4b
#endif

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
        imageStore(outMips, ivec2(naturalOrder), vec4(reducedVal, 0.f, 0.f, 0.f));
        //sharedMem[WORKGROUP_SIZE+gl_LocalInvocationIndex] = reducedVal;
    }
}