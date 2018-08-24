#include "CGLSLFunctionGenerator.h"

using namespace irr;
using namespace core;

std::string CGLSLFunctionGenerator::getLinearSkinningFunction(const uint32_t& maxBoneInfluences)
{
    const char src_begin[] = 
R"(void linearSkin(in samplerBuffer boneTBO, out vec3 skinnedPos, out vec3 skinnedNormal, in vec3 vxPos, in vec3 vxNormal, in ivec4 boneIDs, in vec4 boneWeightsXYZBoneCountNormalized, in int baseBoneTBOOffset, in int boneStrideIn128BitUnits)
{
    vec4 boneData[5];
    float lastBoneData;

    int boneOffset;
)";
    const char src_infl1[] = 
R"(    boneOffset = boneIDs.x * 7;
    //global matrix
    boneData[0] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*boneOffset);
    boneData[1] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*(boneOffset + 1));
    boneData[2] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*(boneOffset + 2));
    //normal matrix
    boneData[3] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*(boneOffset + 3));
    boneData[4] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*(boneOffset + 4));
    lastBoneData = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*(boneOffset + 5)).x;

    skinnedPos = mat4x3(boneData[0], boneData[1], boneData[2])*vec4(vxPos*boneWeightsXYZBoneCountNormalized.x, boneWeightsXYZBoneCountNormalized.x);
    skinnedNormal = mat3(boneData[3], boneData[4], lastBoneData)*(vxNormal*boneWeightsXYZBoneCountNormalized.x);
)";
    const char src_infl2[] = 

R"(    if (boneWeightsXYZBoneCountNormalized.w>0.25)
    {
        boneOffset = boneIDs.y * 7;
        //global matrix
        boneData[0] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*boneOffset);
        boneData[1] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*(boneOffset + 1));
        boneData[2] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*(boneOffset + 2));
        //normal matrix
        boneData[3] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*(boneOffset + 3));
        boneData[4] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*(boneOffset + 4));
        lastBoneData = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*(boneOffset + 5)).x;

        skinnedPos += mat4x3(boneData[0], boneData[1], boneData[2])*vec4(vxPos*boneWeightsXYZBoneCountNormalized.y, boneWeightsXYZBoneCountNormalized.y);
        skinnedNormal += mat3(boneData[3], boneData[4], lastBoneData)*(vxNormal*boneWeightsXYZBoneCountNormalized.y);
    }
)";
    const char src_infl3[] =
R"(    if (boneWeightsXYZBoneCountNormalized.w>0.5)
    {
        boneOffset = boneIDs.z * 7;
        //global matrix
        boneData[0] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*boneOffset);
        boneData[1] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*(boneOffset + 1));
        boneData[2] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*(boneOffset + 2));
        //normal matrix
        boneData[3] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*(boneOffset + 3));
        boneData[4] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*(boneOffset + 4));
        lastBoneData = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*(boneOffset + 5)).x;

        skinnedPos += mat4x3(boneData[0], boneData[1], boneData[2])*vec4(vxPos*boneWeightsXYZBoneCountNormalized.z, boneWeightsXYZBoneCountNormalized.z);
        skinnedNormal += mat3(boneData[3], boneData[4], lastBoneData)*(vxNormal*boneWeightsXYZBoneCountNormalized.z);
    }
)";
    const char src_infl4[] =
R"(    if (boneWeightsXYZBoneCountNormalized.w>0.75)
    {
        boneOffset = boneIDs.w * 7;
        //global matrix
        boneData[0] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*boneOffset);
        boneData[1] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*(boneOffset + 1));
        boneData[2] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*(boneOffset + 2));
        //normal matrix
        boneData[3] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*(boneOffset + 3));
        boneData[4] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*(boneOffset + 4));
        lastBoneData = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*(boneOffset + 5)).x;

        float lastWeight = 1.0 - boneWeightsXYZBoneCountNormalized.x - boneWeightsXYZBoneCountNormalized.y - boneWeightsXYZBoneCountNormalized.z;
        skinnedPos += mat4x3(boneData[0], boneData[1], boneData[2])*vec4(vxPos*lastWeight, lastWeight);
        skinnedNormal += mat3(boneData[3], boneData[4], lastBoneData)*(vxNormal*lastWeight);
    }
)";
    const char src_infl0[] = 
R"(    skinnedPos = vxPos;
    skinnedNormal = vxNormal;
)";
    const char src_end[] = "}";

    std::string sourceStr = src_begin;
    if (maxBoneInfluences == 0u)
    {
        sourceStr += src_infl0;
        sourceStr += src_end;

        return sourceStr;
    }
    sourceStr += src_infl1;
    if (maxBoneInfluences > 1u)
        sourceStr += src_infl2;
    if (maxBoneInfluences > 2u)
        sourceStr += src_infl3;
    if (maxBoneInfluences > 3u)
        sourceStr += src_infl4;

    sourceStr += src_end;

    return sourceStr;
}