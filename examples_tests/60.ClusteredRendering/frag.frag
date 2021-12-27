
#version 430 core

// Todo(achal): Get these from host, some of these are required
// in the culling compute shader as well, need to make the code DRY
#define LOD_COUNT 10
#define VOXEL_COUNT_PER_DIM 4
#define VOXEL_COUNT_PER_LEVEL 64
#define CLIPMAP_EXTENT 11977.0674f
// Somewhat arbitrary
#define LIGHT_CONTRIBUTION_THRESHOLD 2.f
#define LIGHT_RADIUS 25.f

layout (location = 4) flat in vec3 EyePos;
layout (location = 5) in vec3 Normal_WorldSpace;
layout (location = 6) flat in mat4 MVP;

// Todo(achal): This needs to go into a separate header
struct nbl_glsl_ext_ClusteredLighting_SpotLight
{
	vec3 position;
	float outerCosineOverCosineRange;
	uvec2 intensity;
	uvec2 direction;
};

layout (set = 2, binding = 0, std430) restrict buffer readonly LIGHT_DATA
{
	nbl_glsl_ext_ClusteredLighting_SpotLight lights[LIGHT_COUNT];
} ssbo;

layout (set = 2, binding = 1) uniform usampler3D lightGrid;
layout (set = 2, binding = 2) uniform usamplerBuffer lightIndexList;

#define _NBL_COMPUTE_LIGHTING_DEFINED_
#include <nbl/builtin/glsl/bxdf/common.glsl>
vec3 nbl_computeLighting(out nbl_glsl_IsotropicViewSurfaceInteraction out_interaction, in mat2 dUV);

#include <nbl/builtin/glsl/format/decode.glsl>
#include <nbl/builtin/shader/loader/mtl/fragment_impl.glsl>
#include <nbl/builtin/glsl/algorithm.glsl>

// NBL_GLSL_DECLARE_UPPER_BOUND(arr, uint)
// NBL_GLSL_DEFINE_UPPER_BOUND(arr, uint)
uint upper_bound_arr_NBL_GLSL_LESS(uint begin, in uint end, in float value, in float arr[10])
{
    uint len = end - begin;
    if (NBL_GLSL_IS_NOT_POT(len))
    {
        const uint newLen = 0x1u << findMSB(len);
        const uint diff = len - newLen;

        begin = NBL_GLSL_LESS(value, NBL_GLSL_EVAL(arr)[newLen]) ? 0u : diff;
        len = newLen;
    }

    while (len != 0u)
    {
        begin += NBL_GLSL_LESS(value, NBL_GLSL_EVAL(arr)[begin + (len >>= 1u)]) ? 0u : len;
        begin += NBL_GLSL_LESS(value, NBL_GLSL_EVAL(arr)[begin + (len >>= 1u)]) ? 0u : len;
    }

    return begin + (NBL_GLSL_LESS(value, NBL_GLSL_EVAL(arr)[begin]) ? 0u : 1u);
}

// Todo(achal): Probably its better to use a linear search for such small number of
// elements (<10)
#if 0
uint getClipmapLevel(in float clipmapHalfExtent, in float dist)
{
    uint level = 0u;
    // Todo(achal): LOD_COUNT
    const uint LOD_COUNT = 10u;
    while (level < LOD_COUNT)
    {
        const float currHalfExtent = (clipmapHalfExtent / (1 << (LOD_COUNT - 1 - level)));
        if (dist < currHalfExtent)
            return level;

        ++level;
    }

    return level;
}
#endif

vec3 getWorldPosFromFramebufferCoords(in vec4 fragCoord)
{
    // Todo(achal): Need screen dims here
    const float x = (fragCoord.x / 1280.f) * 2.f - 1.f;
    const float y = (fragCoord.y / 720.f) * 2.f - 1.f;
    vec4 ndc = vec4(x, y, 1.f-fragCoord.z, 1.f);
     // Todo(achal): I should probably input VP (or even inverseViewProj) with a UBO
    vec4 intermediate = inverse(MVP) * ndc;
    vec4 worldPos = intermediate / intermediate.w;
    return worldPos.xyz;
}

uint getClipmapLevel(in vec3 worldPos, in vec3 camPos)
{
    const vec3 distFromCamera = abs(worldPos - camPos);
    const float chebyshevDist = max(distFromCamera.x, max(distFromCamera.y, distFromCamera.z));

    float halfExtents[LOD_COUNT];

    float halfExtent = CLIPMAP_EXTENT/2.f;
    for (int i = LOD_COUNT-1; i >= 0; --i)
    {
        halfExtents[i] = halfExtent;
        halfExtent /= 2.f;
    }

    return upper_bound_arr_NBL_GLSL_LESS(0, LOD_COUNT, chebyshevDist, halfExtents);
}

uint getClipmapClusterAtLevel(in uint level, in vec3 worldPos, in vec3 eyePos)
{
    const float levelExtent = CLIPMAP_EXTENT / (1 << (LOD_COUNT - 1 - level));
    const vec3 levelMinVertex = vec3(-levelExtent/2.f);
    const float voxelSideLength = levelExtent / VOXEL_COUNT_PER_DIM;
    const vec3 fromClipmapCenter = worldPos - eyePos;
    uvec3 localClusterCoord = uvec3(floor((fromClipmapCenter - levelMinVertex) / voxelSideLength));

    const uint globalClusterID = (LOD_COUNT - 1 - level) * VOXEL_COUNT_PER_LEVEL
        + (VOXEL_COUNT_PER_DIM * VOXEL_COUNT_PER_DIM * localClusterCoord.z)
        + (VOXEL_COUNT_PER_DIM * localClusterCoord.y)
        + localClusterCoord.x;

    return globalClusterID;
}

ivec3 getLightGridCoords(in uint globalClusterID)
{
    // Todo(achal): Probably should get this from the host???
    const uvec3 voxelCount = uvec3(4u, 4u, 40u);

    ivec3 texCoords;
    texCoords.z = int(globalClusterID/(voxelCount.x * voxelCount.y));
    const int xy = int(globalClusterID%(voxelCount.x * voxelCount.y));
    texCoords.y = int(xy/voxelCount.x);
    texCoords.x = int(xy%voxelCount.x);

    return texCoords;
}

void getLightIndexListOffsetAndCount(in ivec3 gridCoords, out uint offset, out uint count)
{
    uint packedCountOffset = texelFetch(lightGrid, gridCoords, 0).x;
    count = packedCountOffset & 0xFFFF;
    offset = (packedCountOffset >> 16) & 0xFFFF;
}

vec3 computeLightContribution(in vec3 worldPos, in nbl_glsl_ext_ClusteredLighting_SpotLight light, inout nbl_glsl_IsotropicViewSurfaceInteraction interaction, in mat2 dUV)
{
    vec3 result = vec3(0.f);

    const vec3 fromLight = worldPos - light.position.xyz;
    const vec2 dir0 = unpackSnorm2x16(light.direction[0]);
    const vec2 dir1 = unpackSnorm2x16(light.direction[1]);
    const vec3 spotDir = vec3(dir0.xy,dir1.x);
    const float cosineRange = dir1.y;
    const vec3 intensity = nbl_glsl_decodeRGB19E7(light.intensity);

    const float cosTestAngle = dot(normalize(fromLight), spotDir);
    const float cosOuterHalfAngle = light.outerCosineOverCosineRange * cosineRange;

    if (cosTestAngle < cosOuterHalfAngle)
        return result;

    const float lenSq = dot(fromLight,fromLight);
    const float radiusSq = LIGHT_RADIUS*LIGHT_RADIUS;

    float attenuation = 0.5f*radiusSq*(1.f-inversesqrt(1.f+radiusSq/lenSq));
    const float spotHalfAngle = dot(spotDir,fromLight)*inversesqrt(dot(spotDir,spotDir)*dot(fromLight,fromLight));
    const float spotPenumbra = clamp(((spotHalfAngle/cosineRange) - light.outerCosineOverCosineRange), 0.f, 1.f);

    attenuation *= spotPenumbra;

    const vec3 L = light.position.xyz-worldPos;
    const float lenL2 = dot(L,L);
    const float invLenL = inversesqrt(lenL2);
    nbl_glsl_LightSample _sample = nbl_glsl_createLightSample(L*invLenL, interaction);

    if (any(greaterThanEqual(intensity*attenuation,vec3(LIGHT_CONTRIBUTION_THRESHOLD))))
        result += attenuation*intensity*nbl_bsdf_cos_eval(_sample,interaction, dUV);

    return result;
}

vec3 nbl_computeLighting(out nbl_glsl_IsotropicViewSurfaceInteraction out_interaction, in mat2 dUV)
{
    const vec3 WorldPos = getWorldPosFromFramebufferCoords(gl_FragCoord);
    const uint level = getClipmapLevel(WorldPos, EyePos);
    const uint globalClusterID = getClipmapClusterAtLevel(level, WorldPos, EyePos);

    nbl_glsl_IsotropicViewSurfaceInteraction interaction = nbl_glsl_calcSurfaceInteraction(
        EyePos,
        WorldPos,
        Normal_WorldSpace,
        mat2x3(dFdx(ViewPos),dFdy(ViewPos)));

    ivec3 lightGridCoords = getLightGridCoords(globalClusterID);
    
    uint lightOffset, lightCount;
    getLightIndexListOffsetAndCount(lightGridCoords, lightOffset, lightCount);

    vec3 lightAccum = vec3(0.f);
    // for (uint lightID = 0u; lightID < LIGHT_COUNT; ++lightID)
    for (uint i = 0u; i < lightCount; ++i)
    {
        const uint lightID = texelFetch(lightIndexList, int(lightOffset + i)).x;
        const nbl_glsl_ext_ClusteredLighting_SpotLight light = ssbo.lights[lightID];

        lightAccum += computeLightContribution(WorldPos, light, interaction, dUV);
    }

    nbl_glsl_MTLMaterialParameters mtParams = nbl_glsl_getMaterialParameters();
#ifndef _NO_UV
    if ((mtParams.extra&map_bump_MASK) == map_bump_MASK)
    {
        interaction.N = normalize(interaction.N);

        vec2 dh = nbl_sample_bump(UV, dUV).xy;

        interaction.N = nbl_glsl_perturbNormal_derivativeMap(interaction.N, dh, interaction.V.dPosdScreen, dUV);
    }
#endif

    vec3 Ka;
    switch ((mtParams.extra&ILLUM_MODEL_MASK))
    {
    case 0:
    {
#ifndef _NO_UV
    if ((mtParams.extra&(map_Kd_MASK)) == (map_Kd_MASK))
        Ka = nbl_sample_bump(UV, dUV).rgb;
    else
#endif
        Ka = mtParams.Kd;
    }
    break;
    default:
#define Ia 0.1
    {
#ifndef _NO_UV
    if ((mtParams.extra&(map_Ka_MASK)) == (map_Ka_MASK))
        Ka = nbl_sample_Ka(UV, dUV).rgb;
    else
#endif
        Ka = mtParams.Ka;
    Ka *= Ia;
    }
#undef Ia
    break;
    }

    out_interaction = interaction;

    return lightAccum + Ka;
}
