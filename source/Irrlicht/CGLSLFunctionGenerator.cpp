#include "CGLSLFunctionGenerator.h"

using namespace irr;
using namespace video;

std::string CGLSLFunctionGenerator::getLinearSkinningFunction(const uint32_t& maxBoneInfluences)
{
    const char src_begin[] =
R"(
#ifndef _IRR_GENERATED_SKINNING_FUNC_INCLUDED_
#define _IRR_GENERATED_SKINNING_FUNC_INCLUDED_
void linearSkin(in samplerBuffer boneTBO, out vec3 skinnedPos, out vec3 skinnedNormal, in vec3 vxPos, in vec3 vxNormal, in ivec4 boneIDs, in vec4 boneWeightsXYZBoneCountNormalized, in int baseBoneTBOOffset, in int boneStrideIn128BitUnits)
{
    vec4 boneData[5];
    float lastBoneData;

    int boneOffset;
)";
    const char src_infl1[] =
R"(    boneOffset = boneIDs.x * 7;
    //global matrix
    boneData[0] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*boneOffset);
    boneData[1] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*boneOffset + 1);
    boneData[2] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*boneOffset + 2);
    //normal matrix
    boneData[3] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*boneOffset + 3);
    boneData[4] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*boneOffset + 4);
    lastBoneData = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*boneOffset + 5).x;

    skinnedPos = mat4x3(boneData[0], boneData[1], boneData[2])*vec4(vxPos*boneWeightsXYZBoneCountNormalized.x, boneWeightsXYZBoneCountNormalized.x);
    skinnedNormal = mat3(boneData[3], boneData[4], lastBoneData)*(vxNormal*boneWeightsXYZBoneCountNormalized.x);
)";
    const char src_infl2[] =

R"(    if (boneWeightsXYZBoneCountNormalized.w>0.25)
    {
        boneOffset = boneIDs.y * 7;
        //global matrix
        boneData[0] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*boneOffset);
        boneData[1] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*boneOffset + 1);
        boneData[2] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*boneOffset + 2);
        //normal matrix
        boneData[3] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*boneOffset + 3);
        boneData[4] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*boneOffset + 4);
        lastBoneData = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*boneOffset + 5).x;

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
        boneData[1] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*boneOffset + 1);
        boneData[2] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*boneOffset + 2);
        //normal matrix
        boneData[3] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*boneOffset + 3);
        boneData[4] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*boneOffset + 4);
        lastBoneData = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*boneOffset + 5).x;

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
        boneData[1] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*boneOffset + 1);
        boneData[2] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*boneOffset + 2);
        //normal matrix
        boneData[3] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*boneOffset + 3);
        boneData[4] = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*boneOffset + 4);
        lastBoneData = texelFetch(boneTBO, baseBoneTBOOffset + boneStrideIn128BitUnits*boneOffset + 5).x;

        float lastWeight = 1.0 - boneWeightsXYZBoneCountNormalized.x - boneWeightsXYZBoneCountNormalized.y - boneWeightsXYZBoneCountNormalized.z;
        skinnedPos += mat4x3(boneData[0], boneData[1], boneData[2])*vec4(vxPos*lastWeight, lastWeight);
        skinnedNormal += mat3(boneData[3], boneData[4], lastBoneData)*(vxNormal*lastWeight);
    }
)";
    const char src_infl0[] =
R"(    skinnedPos = vxPos;
    skinnedNormal = vxNormal;
)";
    const char src_end[] = "}\n#endif//_IRR_GENERATED_SKINNING_FUNC_INCLUDED_\n";

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

std::string CGLSLFunctionGenerator::getWarpScanPaddingFunctions()
{
    return R"===(
#ifndef _IRR_GENERATED_WARP_SCAN_PAD_FUNCS_INCLUDED_
#define _IRR_GENERATED_WARP_SCAN_PAD_FUNCS_INCLUDED_
uint scan_warpPadAddress(in uint addr, in uint width) {return addr+((addr>>1u)&(~(width/2u-1u)));}

uint scan_warpPadAddress4(in uint addr) {return addr+((addr>>1u)&0xfffffffeu);}
uint scan_warpPadAddress8(in uint addr) {return addr+((addr>>1u)&0xfffffffcu);}
uint scan_warpPadAddress16(in uint addr) {return addr+((addr>>1u)&0xfffffff8u);}
uint scan_warpPadAddress32(in uint addr) {return addr+((addr>>1u)&0xfffffff0u);}
uint scan_warpPadAddress64(in uint addr) {return addr+((addr>>1u)&0xffffffe0u);}
uint scan_warpPadAddress128(in uint addr) {return addr+((addr>>1u)&0xffffffc0u);}
uint scan_warpPadAddress256(in uint addr) {return addr+((addr>>1u)&0xffffff80u);}
uint scan_warpPadAddress512(in uint addr) {return addr+((addr>>1u)&0xffffff00u);}
uint scan_warpPadAddress1024(in uint addr) {return addr+((addr>>1u)&0xfffffe00u);}
uint scan_warpPadAddress2048(in uint addr) {return addr+((addr>>1u)&0xfffffc00u);}
uint scan_warpPadAddress4096(in uint addr) {return addr+((addr>>1u)&0xfffff800u);}
#endif //_IRR_GENERATED_WARP_SCAN_PAD_FUNCS_INCLUDED_
            )===";
}

const char* glslTypeNames[CGLSLFunctionGenerator::EGT_COUNT] = {
    "float",
    "vec2",
    "vec3",
    "vec4"
};

std::string getCommOpDefine(const CGLSLFunctionGenerator::E_GLSL_COMMUTATIVE_OP& oper)
{
    switch (oper)
    {
        case CGLSLFunctionGenerator::EGCO_ADD:
            return "\n#define SCAN_OP(X,Y) (X+Y)\n";
            break;
        case CGLSLFunctionGenerator::EGCO_AND:
            return "\n#define SCAN_OP(X,Y) (X&Y)\n";
            break;
        case CGLSLFunctionGenerator::EGCO_MAX:
            return "\n#define SCAN_OP(X,Y) max(X,Y)\n";
            break;
        case CGLSLFunctionGenerator::EGCO_MIN:
            return "\n#define SCAN_OP(X,Y) min(X,Y)\n";
            break;
        case CGLSLFunctionGenerator::EGCO_MUL:
            return "\n#define SCAN_OP(X,Y) (X*Y)\n";
            break;
        case CGLSLFunctionGenerator::EGCO_OR:
            return "\n#define SCAN_OP(X,Y) (X|Y)\n";
            break;
        case CGLSLFunctionGenerator::EGCO_XOR:
            return "\n#define SCAN_OP(X,Y) (X^Y)\n";
            break;
        default:
            return "";
            break;
    }

    return "";
}

std::string CGLSLFunctionGenerator::getWarpInclusiveScanFunctionsPadded(const E_GLSL_COMMUTATIVE_OP& oper, const E_GLSL_TYPE& dataType,
                                                                        const std::string& namePostfix, const std::string& getterFunc, const std::string& setterFunc)
{
    const std::string includeGuardMacroName = "_IRR_GENERATED_WARP_SCAN_PADDED_"+namePostfix+"_FUNCS_INCLUDED_";

    std::string sourceStr = "\n#ifndef "+includeGuardMacroName+"\n#define "+includeGuardMacroName+"\n";

    sourceStr += getWarpScanPaddingFunctions();

    //sourceStr += "\n#undef SCAN_OP\n";

    sourceStr += "\n#endif //"+includeGuardMacroName+"\n";
    return sourceStr;
}

std::string CGLSLFunctionGenerator::getWarpReduceFunctionsPadded(   const E_GLSL_COMMUTATIVE_OP& oper, const E_GLSL_TYPE& dataType,
                                                                    const std::string& namePostfix, const std::string& getterFunc, const std::string& setterFunc)
{
    const std::string includeGuardMacroName = "_IRR_GENERATED_WARP_REDUCE_PADDED_"+namePostfix+"_FUNCS_INCLUDED_";

    std::string sourceStr = "\n#ifndef "+includeGuardMacroName+"\n#define "+includeGuardMacroName+"\n";

    sourceStr += getWarpScanPaddingFunctions();

    //sourceStr += "\n#undef SCAN_OP\n";

    sourceStr += "\n#endif //"+includeGuardMacroName+"\n";
    return sourceStr;
}
