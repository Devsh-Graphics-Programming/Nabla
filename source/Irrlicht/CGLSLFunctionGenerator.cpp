#include "CGLSLFunctionGenerator.h"
#include "COpenGLExtensionHandler.h"

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

std::string CGLSLFunctionGenerator::getReduceAndScanExtensionEnables(IVideoCapabilityReporter* reporter)
{
    std::string retval = "\n#define GL_WARP_SIZE_NV ";

#ifdef _IRR_COMPILE_WITH_OPENGL_
    if (reporter->getDriverType()==EDT_OPENGL&&COpenGLExtensionHandler::FeatureAvailable[COpenGLExtensionHandler::IRR_NV_shader_thread_group])
    {
        int32_t tmp;
        glGetIntegerv(GL_WARP_SIZE_NV,&tmp);
        retval += std::to_string(tmp)+"u\n";
    }
#endif // _IRR_COMPILE_WITH_OPENGL_
#ifdef _IRR_COMPILE_WITH_VULKAN_
    else if (reporter->getDriverType()==EDT_VULKAN)
    {
        //! something
    }
#endif // _IRR_COMPILE_WITH_VULKAN_
#if defined(_IRR_COMPILE_WITH_OPENGL_)||defined(_IRR_COMPILE_WITH_VULKAN_)
    else
#endif
        retval += "gl_WarpSizeNV\n";

    retval += R"===(
#extension GL_KHR_shader_subgroup_arithmetic: enable

#ifdef GL_KHR_shader_subgroup_arithmetic
    #define SUBGROUP_SIZE gl_SubgroupSize
#else
    #extension GL_NV_shader_thread_shuffle: enable

    #ifdef GL_NV_shader_thread_shuffle
        #extension GL_NV_shader_thread_group: enable

        #ifdef GL_NV_shader_thread_group
            #define SUBGROUP_SIZE GL_WARP_SIZE_NV
        #else
            #define SUBGROUP_SIZE 32u
        #endif
        #define CONSTANT_SUBGROUP_SIZE

        #define subgroupShuffleUpEMUL(VAL,DELTA) shuffleUpNV((VAL),(DELTA),SUBGROUP_SIZE)
    #else
        #extension GL_AMD_shader_ballot: enable

        #ifdef GL_AMD_shader_ballot
            #define SUBGROUP_SIZE gl_SubGroupSizeARB
        #else
            #extension GL_KHR_shader_subgroup_shuffle_relative: enable
            #extension GL_KHR_shader_subgroup_shuffle: enable

            #ifdef GL_KHR_shader_subgroup_shuffle||GL_KHR_shader_subgroup_shuffle_relative
                #define SUBGROUP_SIZE gl_SubgroupSize

                #ifdef GL_KHR_shader_subgroup_shuffle_relative
                    #define subgroupShuffleUpEMUL(VAL,DELTA) subgroupShuffleUp((VAL),(DELTA))
                #else
                    #define subgroupShuffleUpEMUL(VAL,DELTA) subgroupShuffle((VAL),gl_SubgroupInvocationID-(DELTA))
                #endif
            #else
                #extension GL_NV_shader_thread_group: enable

                #ifdef GL_NV_shader_thread_group
                    #define SUBGROUP_SIZE GL_WARP_SIZE_NV
                    #define CONSTANT_SUBGROUP_SIZE
                #else
                    #extension GL_ARB_shader_ballot: enable

                    #ifdef GL_ARB_shader_ballot
                        #define SUBGROUP_SIZE gl_SubGroupSizeARB
                    #else
                        #extension GL_KHR_shader_subgroup_basic: enable

                        #ifdef GL_KHR_shader_subgroup_basic
                            #define SUBGROUP_SIZE gl_SubgroupSize
                        #endif // KHR_shader_subgroup_basic
                    #endif // ARB_shader_ballot
                #endif // NV_shader_thread_group
            #endif // KHR_shader_subgroup_shuffle
        #endif // AMD_shader_ballot
    #endif // NV_shader_thread_shuffle
#endif // KHR_shader_subgroup_arithmetic

#ifdef SUBGROUP_SIZE
    #define PROBABLE_SUBGROUP_SIZE SUBGROUP_SIZE
    #ifdef CONSTANT_SUBGROUP_SIZE
        #define CONSTANT_PROBABLE_SUBGROUP_SIZE
    #endif
#else
    #define PROBABLE_SUBGROUP_SIZE 4u //only size we can guarantee on Intel, unless someone can prove to me that Intel uses SIMD16 mode always for Compute
    #define CONSTANT_PROBABLE_SUBGROUP_SIZE
#endif // SUBGROUP_SIZE
                    )===";

    return retval;
}

std::string CGLSLFunctionGenerator::getWarpPaddingFunctions()
{
    return R"===(
#ifndef _IRR_GENERATED_WARP_PAD_FUNCS_INCLUDED_
#define _IRR_GENERATED_WARP_PAD_FUNCS_INCLUDED_
uint warpPadAddress(in uint addr, in uint width) {return addr+((addr>>1u)&(~(width/2u-1u)));}

uint warpPadAddress4(in uint addr) {return addr+((addr>>1u)&0xfffffffeu);}
uint warpPadAddress8(in uint addr) {return addr+((addr>>1u)&0xfffffffcu);}
uint warpPadAddress16(in uint addr) {return addr+((addr>>1u)&0xfffffff8u);}
uint warpPadAddress32(in uint addr) {return addr+((addr>>1u)&0xfffffff0u);}
uint warpPadAddress64(in uint addr) {return addr+((addr>>1u)&0xffffffe0u);}
uint warpPadAddress128(in uint addr) {return addr+((addr>>1u)&0xffffffc0u);}
uint warpPadAddress256(in uint addr) {return addr+((addr>>1u)&0xffffff80u);}
uint warpPadAddress512(in uint addr) {return addr+((addr>>1u)&0xffffff00u);}
uint warpPadAddress1024(in uint addr) {return addr+((addr>>1u)&0xfffffe00u);}
uint warpPadAddress2048(in uint addr) {return addr+((addr>>1u)&0xfffffc00u);}
uint warpPadAddress4096(in uint addr) {return addr+((addr>>1u)&0xfffff800u);}
#endif //_IRR_GENERATED_WARP_PAD_FUNCS_INCLUDED_
            )===";
}

const char* glslTypeNames[CGLSLFunctionGenerator::EGT_COUNT] = {
    "float",
    "vec2",
    "vec3",
    "vec4"
};

std::string getTypeDef(const CGLSLFunctionGenerator::E_GLSL_TYPE& type)
{
    return std::string("\n#undef TYPE\n#define TYPE ")+glslTypeNames[type]+"\n";
}

std::string getCommOpDefine(const CGLSLFunctionGenerator::E_GLSL_COMMUTATIVE_OP& oper)
{
    switch (oper)
    {
        case CGLSLFunctionGenerator::EGCO_ADD:
            return "\n#undef COMM_OPInvocationsInclusiveScanAMD\n#undef subgroupInclusiveCOMM_OP\n#undef COMM_OP\n#define COMM_OP(X,Y) ((X)+(Y))\n#define subgroupInclusiveCOMM_OP(val) subgroupInclusiveAdd(val)\n#define COMM_OPInvocationsInclusiveScanAMD(val) addInvocationsInclusiveScanAMD(val)\n";
            break;
        case CGLSLFunctionGenerator::EGCO_AND:
            return "\n#undef COMM_OPInvocationsInclusiveScanAMD\n#undef subgroupInclusiveCOMM_OP\n#undef COMM_OP\n#define COMM_OP(X,Y) ((X)&(Y))\n#define subgroupInclusiveCOMM_OP(val) subgroupInclusiveAnd(val)\n#define COMM_OPInvocationsInclusiveScanAMD(val) andInvocationsInclusiveScanAMD(val)\n";
            break;
        case CGLSLFunctionGenerator::EGCO_MAX:
            return "\n#undef COMM_OPInvocationsInclusiveScanAMD\n#undef subgroupInclusiveCOMM_OP\n#undef COMM_OP\n#define COMM_OP(X,Y) max(X,Y)\n#define subgroupInclusiveCOMM_OP(val) subgroupInclusiveMax(val)\n#define COMM_OPInvocationsInclusiveScanAMD(val) maxInvocationsInclusiveScanAMD(val)\n";
            break;
        case CGLSLFunctionGenerator::EGCO_MIN:
            return "\n#undef COMM_OPInvocationsInclusiveScanAMD\n#undef subgroupInclusiveCOMM_OP\n#undef COMM_OP\n#define COMM_OP(X,Y) min(X,Y)\n#define subgroupInclusiveCOMM_OP(val) subgroupInclusiveMin(val)\n#define COMM_OPInvocationsInclusiveScanAMD(val) minInvocationsInclusiveScanAMD(val)\n";
            break;/*
        case CGLSLFunctionGenerator::EGCO_MUL:
            return "\n#undef COMM_OPInvocationsInclusiveScanAMD\n#undef subgroupInclusiveCOMM_OP\n#undef COMM_OP\n#define COMM_OP(X,Y) ((X)*(Y))\n#define subgroupInclusiveCOMM_OP(val) subgroupInclusiveMul(val)\n#define COMM_OPInvocationsInclusiveScanAMD(val) !?!?!?!??!(val)\n";
            break;*/
        case CGLSLFunctionGenerator::EGCO_OR:
            return "\n#undef COMM_OPInvocationsInclusiveScanAMD\n#undef subgroupInclusiveCOMM_OP\n#undef COMM_OP\n#define COMM_OP(X,Y) ((X)|(Y))\n#define subgroupInclusiveCOMM_OP(val) subgroupInclusiveOr(val)\n#define COMM_OPInvocationsInclusiveScanAMD(val) orInvocationsInclusiveScanAMD(val)\n";
            break;
        case CGLSLFunctionGenerator::EGCO_XOR:
            return "\n#undef COMM_OPInvocationsInclusiveScanAMD\n#undef subgroupInclusiveCOMM_OP\n#undef COMM_OP\n#define COMM_OP(X,Y) ((X)^(Y))\n#define subgroupInclusiveCOMM_OP(val) subgroupInclusiveXor(val)\n#define COMM_OPInvocationsInclusiveScanAMD(val) xorInvocationsInclusiveScanAMD(val)\n";
            break;
        default:
            return "";
            break;
    }

    return "";
}

std::string CGLSLFunctionGenerator::getWarpInclusiveScanFunctionsPadded(const E_GLSL_COMMUTATIVE_OP& oper, const E_GLSL_TYPE& dataType,
                                                                        const std::string& namePostfix, const std::string& tmpSharedMemName, const size_t& smemOffset)
{
    const std::string includeGuardMacroName = "_IRR_GENERATED_WARP_SCAN_PADDED_"+namePostfix+"_FUNCS_INCLUDED_";

    std::string sourceStr = "\n#ifndef "+includeGuardMacroName+"\n#define "+includeGuardMacroName+"\n";

    sourceStr += getWarpPaddingFunctions()+getCommOpDefine(oper)+getTypeDef(dataType);

    sourceStr += "\n#undef WARP_SCAN_SIZE\n#define WARP_SCAN_SIZE PROBABLE_SUBGROUP_SIZE\n";

    sourceStr += "\n#undef WARP_SCAN_FUNC_NAME\n#define WARP_SCAN_FUNC_NAME warp_incl_scan_padded"+namePostfix+"\n";

    sourceStr += "\n#undef WARP_SCAN_FUNC_NAME_SAFE\n#define WARP_SCAN_FUNC_NAME_SAFE warp_incl_scan_padded_safe"+namePostfix+"\n";

    sourceStr += "\n#undef WARP_SCAN_FUNC_NAME_SIZED(W)\n#define WARP_SCAN_FUNC_NAME_SIZED warp_incl_scan_padded ## W ## "+namePostfix+"\n";

    sourceStr += "\n#undef TMP_SMEM\n#define TMP_SMEM(IDX) "+tmpSharedMemName+"[(IDX)+"+std::to_string(smemOffset)+"]\n";

    sourceStr += R"==(
//! Warning these functions will reduce across the entire warp under GL_KHR_shader_subgroup_arithmetic,GL_AMD_shader_ballot
TYPE WARP_SCAN_FUNC_NAME (in TYPE val, in uint idx);

// base case
TYPE WARP_SCAN_FUNC_NAME_SIZED(4u) (in TYPE val, in uint idx)
{
#if defined(GL_KHR_shader_subgroup_arithmetic)||defined(GL_AMD_shader_ballot)
    return WARP_SCAN_FUNC_NAME (val,idx);
#elif defined(GL_NV_shader_thread_shuffle)||defined(GL_KHR_shader_subgroup_shuffle)||defined(GL_KHR_shader_subgroup_shuffle_relative)
    TYPE tmpA = val;
    TYPE tmpB = subgroupShuffleUpEMUL(tmpA,1);
    if (gl_ThreadInWarpNV>1)
        tmpA = COMM_OP(tmpA,tmpB);
    tmpB = subgroupShuffleUpEMUL(tmpA,2);
    if (gl_ThreadInWarpNV>2)
        tmpA = COMM_OP(tmpA,tmpB);
    return tmpA;
#elif defined(GL_KHR_shader_subgroup_basic)
    subgroupBarrier();
    TMP_SMEM(idx+1u) = COMM_OP(TMP_SMEM(idx),TMP_SMEM(idx+1u));
    subgroupBarrier();
    TMP_SMEM(idx+2u) = COMM_OP(TMP_SMEM(idx),TMP_SMEM(idx+2u));
    subgroupBarrier();
    return TMP_SMEM(idx);
#else
    TMP_SMEM(idx+1u) = COMM_OP(TMP_SMEM(idx),TMP_SMEM(idx+1u));
    TMP_SMEM(idx+2u) = COMM_OP(TMP_SMEM(idx),TMP_SMEM(idx+2u));
    return TMP_SMEM(idx);
#endif
}

// recursive
#undef WARP_INCL_SCAN_PADDED_DECL
#define WARP_INCL_SCAN_PADDED_DECL(W)    TYPE WARP_SCAN_FUNC_NAME_SIZED(W) (in TYPE val, in uint idx)

#undef WARP_INCL_SCAN_PADDED_DEF
#if defined(GL_KHR_shader_subgroup_arithmetic)||defined(GL_AMD_shader_ballot)
    #define WARP_INCL_SCAN_PADDED_DEF(W) WARP_INCL_SCAN_PADDED_DECL(W) \
    {\
        return WARP_SCAN_FUNC_NAME (val,idx); \
    }
#elif defined(GL_NV_shader_thread_shuffle)||defined(GL_KHR_shader_subgroup_shuffle)||defined(GL_KHR_shader_subgroup_shuffle_relative)
    #define WARP_INCL_SCAN_PADDED_DEF(W) WARP_INCL_SCAN_PADDED_DECL(W) \
    {\
        TYPE tmpA = WARP_SCAN_FUNC_NAME_SIZED(W/2u) (idx); \
        TYPE tmpB = subgroupShuffleUpEMUL(tmpA,W/2u); \
        if (gl_ThreadInWarpNV>(W/2u)) \
            tmpA = COMM_OP(tmpA,tmpB); \
        return tmpA;
    }
#elif defined(GL_KHR_shader_subgroup_basic)
    #define WARP_INCL_SCAN_PADDED_DEF(W) WARP_INCL_SCAN_PADDED_DECL(W) \
    {\
        TMP_SMEM(idx+(W/2u)) = COMM_OP(WARP_SCAN_FUNC_NAME_SIZED(W/2u) (idx),TMP_SMEM(idx+(W/2u))); \
        subgroupBarrier(); \
        return TMP_SMEM(idx); \
    }
#else
    #define WARP_INCL_SCAN_PADDED_DEF(W) WARP_INCL_SCAN_PADDED_DECL(W) \
    {\
        TMP_SMEM(idx+(W/2u)) = COMM_OP(WARP_SCAN_FUNC_NAME_SIZED(W/2u) (idx),TMP_SMEM(idx+(W/2u))); \
        return TMP_SMEM(idx); \
    }
#endif


WARP_INCL_SCAN_PADDED_DEF(8u)

WARP_INCL_SCAN_PADDED_DEF(16u)

WARP_INCL_SCAN_PADDED_DEF(32u)

WARP_INCL_SCAN_PADDED_DEF(64u)


TYPE WARP_SCAN_FUNC_NAME (in TYPE val, in uint idx)
{
#ifdef GL_KHR_shader_subgroup_arithmetic
    return subgroupInclusiveCOMM_OP(val);
#elif defined(GL_NV_shader_thread_shuffle)
    #if SUBGROUP_SIZE==32u
        return WARP_SCAN_FUNC_NAME_SIZED(32u) (val,idx);
    #elif SUBGROUP_SIZE==16u
        return WARP_SCAN_FUNC_NAME_SIZED(16u) (val,idx);
    #else
        #error "What size are your NV warps!?"
    #endif
#elif defined(GL_AMD_shader_ballot)
    return COMM_OPInvocationsInclusiveScanAMD(val);
#else
	switch(WARP_SCAN_SIZE)
	{
		case 8u:
			return WARP_SCAN_FUNC_NAME_SIZED(8u) (val,idx);
            break;
		case 16u:
			return WARP_SCAN_FUNC_NAME_SIZED(16u) (val,idx);
            break;
		case 32u:
			return WARP_SCAN_FUNC_NAME_SIZED(32u) (val,idx);
            break;
		case 64u:
			return WARP_SCAN_FUNC_NAME_SIZED(64u) (val,idx);
            break;
	}

    return WARP_SCAN_FUNC_NAME_SIZED(4u) (val,idx);
#endif
}

// COMM_OP_IDENTITY will be user provided
#ifdef COMM_OP_IDENTITY
    //! safe function which will not reduce more of the warp than VARIABLE_SCAN_SZ
    // optimized for a constant/dynamically uniform VARIABLE_SCAN_SZ
    TYPE WARP_SCAN_FUNC_NAME_SAFE (TYPE val, in uint idx,in uint VARIABLE_SCAN_SZ)
    {
        val = idx<VARIABLE_SCAN_SZ ? val:COMM_OP_IDENTITY;
        return WARP_SCAN_FUNC_NAME (val,idx);
    }
#endif
                      )==";

    sourceStr += "\n#ifdef CONSTANT_PROBABLE_SUBGROUP_SIZE\n    #define WARP_INCL_SCAN_PADDED_DEFAULT"+namePostfix+
                    "(VAL,_IDX) WARP_SCAN_FUNC_NAME ## WARP_SCAN_SIZE (VAL,_IDX)\n#else\n    #define WARP_INCL_SCAN_PADDED_DEFAULT"+namePostfix+
                    "(VAL,_IDX) WARP_SCAN_FUNC_NAME (VAL,_IDX)\n#endif // CONSTANT_PROBABLE_SUBGROUP_SIZE\n";

    // undefine all that stuff
    sourceStr += "\n#undef WARP_INCL_SCAN_PADDED_DEF\n#undef WARP_INCL_SCAN_PADDED_DECL\n#undef WARP_SCAN_FUNC_NAME_SIZED\n#undef WARP_SCAN_FUNC_NAME_SAFE\n#undef WARP_SCAN_FUNC_NAME\n#undef TMP_SMEM\n";
    sourceStr += "\n#undef TYPE\n#undef COMM_OPInvocationsInclusiveScanAMD\n#undef subgroupInclusiveCOMM_OP\n#undef COMM_OP\n";

    sourceStr += "\n#endif //"+includeGuardMacroName+"\n";
    return sourceStr;
}

/** TODO: Much later
std::string CGLSLFunctionGenerator::getWarpReduceFunctionsPadded(   const E_GLSL_COMMUTATIVE_OP& oper, const E_GLSL_TYPE& dataType,
                                                                    const std::string& namePostfix, const std::string& tmpSharedMemName, const size_t& smemOffset)
{
    const std::string includeGuardMacroName = "_IRR_GENERATED_WARP_REDUCE_PADDED_"+namePostfix+"_FUNCS_INCLUDED_";

    std::string sourceStr = "\n#ifndef "+includeGuardMacroName+"\n#define "+includeGuardMacroName+"\n";

    sourceStr += getWarpPaddingFunctions()+getCommOpDefine(oper)+getTypeDef(dataType);

    sourceStr += R"==(
                      )==";

    // undefine all that stuff
    sourceStr += "\n#undef TYPE\n#undef COMM_OPInvocationsInclusiveScanAMD\n#undef subgroupInclusiveCOMM_OP\n#undef COMM_OP\n";

    sourceStr += "\n#endif //"+includeGuardMacroName+"\n";
    return sourceStr;
}
*/

/** TODO: NOW
Implement a getBlockReduceFunctionsPadded and getBlockScanFunctionsPadded
Get the correct amount of shared memory to declare
.*/
