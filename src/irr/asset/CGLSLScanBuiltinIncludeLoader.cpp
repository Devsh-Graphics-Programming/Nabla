#include "irr/asset/CGLSLScanBuiltinIncludeLoader.h"

#include <cctype>
#include <regex>

#include "COpenGLExtensionHandler.h"

using namespace irr;
using namespace asset;

static core::vector<std::string> parseArgumentsFromPath(const std::string& _path)
{
    core::vector<std::string> args;

    std::stringstream ss{_path};
    std::string arg;
    while (std::getline(ss, arg, '/'))
        args.push_back(std::move(arg));

    return args;
}

auto CGLSLScanBuiltinIncludeLoader::getBuiltinNamesToFunctionMapping() const -> core::vector<std::pair<std::regex, HandleFunc_t>>
{
    auto handle_incl_scan_common = [](const std::string& _op_s, const std::string& _type_s) {
        E_GLSL_COMMUTATIVE_OP op = EGCO_COUNT;
        if (_op_s == "add") op = EGCO_ADD;
        else if (_op_s == "and") op = EGCO_AND;
        else if (_op_s == "max") op = EGCO_MAX;
        else if (_op_s == "min") op = EGCO_MIN;
        else if (_op_s == "or") op = EGCO_OR;
        else if (_op_s == "xor") op = EGCO_XOR;

        E_GLSL_TYPE type = EGT_COUNT;
        if (_type_s == "float") type = EGT_FLOAT;
        else if (_type_s == "vec2") type = EGT_VEC2;
        else if (_type_s == "vec3") type = EGT_VEC3;
        else if (_type_s == "vec4") type = EGT_VEC4;

        return std::make_tuple(op, type);
    };
    auto handle_warp_incl_scan = [this,&handle_incl_scan_common](const std::string& _path) -> std::string {
        auto args = parseArgumentsFromPath(_path.substr(_path.find_first_of('/')+1, _path.npos));
        if (args.size() < 5u)
            return {};

        const std::string& op_s = args[0];
        const std::string& type_s = args[1];
        E_GLSL_COMMUTATIVE_OP op = EGCO_COUNT;
        E_GLSL_TYPE type = EGT_COUNT;
        std::tie(op, type) = handle_incl_scan_common(op_s, type_s);

        const std::string& postfix = args[2], &getter = args[3], &setter = args[4];

        return getWarpInclusiveScanFunctionsPadded(op, type, postfix, getter, setter);
    };
    auto handle_block_incl_scan = [this,&handle_incl_scan_common](const std::string& _path) -> std::string {
        auto args = parseArgumentsFromPath(_path.substr(_path.find_first_of('/')+1, _path.npos));
        if (args.size() < 7u)
            return {};

        const std::string& op_s = args[0];
        const std::string& type_s = args[1];
        E_GLSL_COMMUTATIVE_OP op = EGCO_COUNT;
        E_GLSL_TYPE type = EGT_COUNT;
        std::tie(op, type) = handle_incl_scan_common(op_s, type_s);

        const std::string& elementsToReduce_s = args[2];
        const std::string& wgSize_s = args[3];
        const uint32_t elementsToReduce = std::atoi(elementsToReduce_s.c_str());
        const uint32_t wgSize = std::atoi(wgSize_s.c_str());

        // TODO: what is this `postfix` actually needed for?? (same in handle_warp_incl_scan)
        const std::string& postfix = args[4], &getter = args[5], &setter = args[6];

        return getBlockInclusiveScanFunctionsPadded(elementsToReduce, wgSize, op, type, postfix, getter, setter);
    };

    return {
        {std::regex{"reduce_and_scan_enables\\.glsl"}, [this](const std::string&) { return getReduceAndScanExtensionEnables(); }},
        {std::regex{"warp_padding\\.glsl"}, [](const std::string&) { return getWarpPaddingFunctions(); }},
        {std::regex{"warp_inclusive_scan\\.glsl/(add|and|max|min|or|xor)/(float|vec2|vec3|vec4)/[a-zA-Z][a-zA-Z0-9]*/[a-zA-Z][a-zA-Z0-9]*/[a-zA-Z][a-zA-Z0-9]*"}, handle_warp_incl_scan },
        {std::regex{"block_inclusive_scan\\.glsl/(add|and|max|min|or|xor)/(float|vec2|vec3|vec4)/[0-9]+/[0-9]+/[a-zA-Z][a-zA-Z0-9]*/[a-zA-Z][a-zA-Z0-9]*/[a-zA-Z][a-zA-Z0-9]*"}, handle_block_incl_scan }
    };
}

std::string CGLSLScanBuiltinIncludeLoader::getReduceAndScanExtensionEnables() const
{


    std::string retval = R"===(\n#ifndef _IRR_GENERATED_REDUCE_AND_SCAN_EXTS_ENABLES_INCLUDED_
#define _IRR_GENERATED_REDUCE_AND_SCAN_EXTS_ENABLES_INCLUDED_
#define GL_WARP_SIZE_NV 32";
)===";
/*
#ifdef _IRR_COMPILE_WITH_OPENGL_
    if (m_capabilityReporter->getDriverType()==EDT_OPENGL&&COpenGLExtensionHandler::FeatureAvailable[COpenGLExtensionHandler::IRR_NV_shader_thread_group])
    {
        int32_t tmp;
        glGetIntegerv(GL_WARP_SIZE_NV,&tmp);
        retval += std::to_string(tmp)+"u\n";
    }
#endif // _IRR_COMPILE_WITH_OPENGL_
#ifdef _IRR_COMPILE_WITH_VULKAN_
    
    //else if (m_capabilityReporter->getDriverType()==EDT_VULKAN)
    //{
    //    //! something
   // }
#endif // _IRR_COMPILE_WITH_VULKAN_
#if defined(_IRR_COMPILE_WITH_OPENGL_)||defined(_IRR_COMPILE_WITH_VULKAN_)
    else
#endif
        retval += "gl_WarpSizeNV\n";
*/
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

            #if defined(GL_KHR_shader_subgroup_shuffle)||defined(GL_KHR_shader_subgroup_shuffle_relative)
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

#endif //_IRR_GENERATED_REDUCE_AND_SCAN_EXTS_ENABLES_INCLUDED_
                    )===";

    return retval;
}

std::string CGLSLScanBuiltinIncludeLoader::getWarpPaddingFunctions()
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

#endif
            )===";
}

std::string CGLSLScanBuiltinIncludeLoader::getTypeDef(const E_GLSL_TYPE& type)
{
    const char* glslTypeNames[CGLSLScanBuiltinIncludeLoader::EGT_COUNT] = {
    "float",
    "vec2",
    "vec3",
    "vec4"
    };
    return std::string("\n#undef TYPE\n#define TYPE ")+glslTypeNames[type]+"\n";
}

std::string CGLSLScanBuiltinIncludeLoader::getCommOpDefine(const E_GLSL_COMMUTATIVE_OP& oper)
{
    switch (oper)
    {
        case CGLSLScanBuiltinIncludeLoader::EGCO_ADD:
            return "\n#undef COMM_OPInvocationsInclusiveScanAMD\n#undef subgroupInclusiveCOMM_OP\n#undef COMM_OP\n#define COMM_OP(X,Y) ((X)+(Y))\n#define subgroupInclusiveCOMM_OP(val) subgroupInclusiveAdd(val)\n#define COMM_OPInvocationsInclusiveScanAMD(val) addInvocationsInclusiveScanAMD(val)\n";
            break;
        case CGLSLScanBuiltinIncludeLoader::EGCO_AND:
            return "\n#undef COMM_OPInvocationsInclusiveScanAMD\n#undef subgroupInclusiveCOMM_OP\n#undef COMM_OP\n#define COMM_OP(X,Y) ((X)&(Y))\n#define subgroupInclusiveCOMM_OP(val) subgroupInclusiveAnd(val)\n#define COMM_OPInvocationsInclusiveScanAMD(val) andInvocationsInclusiveScanAMD(val)\n";
            break;
        case CGLSLScanBuiltinIncludeLoader::EGCO_MAX:
            return "\n#undef COMM_OPInvocationsInclusiveScanAMD\n#undef subgroupInclusiveCOMM_OP\n#undef COMM_OP\n#define COMM_OP(X,Y) max(X,Y)\n#define subgroupInclusiveCOMM_OP(val) subgroupInclusiveMax(val)\n#define COMM_OPInvocationsInclusiveScanAMD(val) maxInvocationsInclusiveScanAMD(val)\n";
            break;
        case CGLSLScanBuiltinIncludeLoader::EGCO_MIN:
            return "\n#undef COMM_OPInvocationsInclusiveScanAMD\n#undef subgroupInclusiveCOMM_OP\n#undef COMM_OP\n#define COMM_OP(X,Y) min(X,Y)\n#define subgroupInclusiveCOMM_OP(val) subgroupInclusiveMin(val)\n#define COMM_OPInvocationsInclusiveScanAMD(val) minInvocationsInclusiveScanAMD(val)\n";
            break;/*
        case CGLSLScanBuiltinIncludeLoader::EGCO_MUL:
            return "\n#undef COMM_OPInvocationsInclusiveScanAMD\n#undef subgroupInclusiveCOMM_OP\n#undef COMM_OP\n#define COMM_OP(X,Y) ((X)*(Y))\n#define subgroupInclusiveCOMM_OP(val) subgroupInclusiveMul(val)\n#define COMM_OPInvocationsInclusiveScanAMD(val) !?!?!?!??!(val)\n";
            break;*/
        case CGLSLScanBuiltinIncludeLoader::EGCO_OR:
            return "\n#undef COMM_OPInvocationsInclusiveScanAMD\n#undef subgroupInclusiveCOMM_OP\n#undef COMM_OP\n#define COMM_OP(X,Y) ((X)|(Y))\n#define subgroupInclusiveCOMM_OP(val) subgroupInclusiveOr(val)\n#define COMM_OPInvocationsInclusiveScanAMD(val) orInvocationsInclusiveScanAMD(val)\n";
            break;
        case CGLSLScanBuiltinIncludeLoader::EGCO_XOR:
            return "\n#undef COMM_OPInvocationsInclusiveScanAMD\n#undef subgroupInclusiveCOMM_OP\n#undef COMM_OP\n#define COMM_OP(X,Y) ((X)^(Y))\n#define subgroupInclusiveCOMM_OP(val) subgroupInclusiveXor(val)\n#define COMM_OPInvocationsInclusiveScanAMD(val) xorInvocationsInclusiveScanAMD(val)\n";
            break;
        default:
            return "";
            break;
    }

    return "";
}

std::string CGLSLScanBuiltinIncludeLoader::getWarpInclusiveScanFunctionsPadded(const E_GLSL_COMMUTATIVE_OP& oper, const E_GLSL_TYPE& dataType, const std::string& namePostfix,
                                                                        const std::string& getterFuncName, const std::string& setterFuncName)
{
    const std::string includeGuardMacroName = "_IRR_GENERATED_WARP_SCAN_PADDED_" + namePostfix + "_FUNCS_INCLUDED_";

    //TODO name include's name (warp_padding.glsl now) could be returned by some func so it's easily changable in code
    std::string sourceStr = //"#include \"irr/builtin/warp_padding.glsl\"\n" + // TODO UNCOMMENT LATER
        "\n#ifndef " + includeGuardMacroName + "\n#define " + includeGuardMacroName + "\n" +
        getWarpPaddingFunctions() +
        getCommOpDefine(oper) + getTypeDef(dataType);

    sourceStr += R"==(
#undef WARP_SCAN_SIZE
#define WARP_SCAN_SIZE  PROBABLE_SUBGROUP_SIZE

#undef PASTER2
#undef PASTER3
#define PASTER2(a,b)    a ## b
#define PASTER3(a,b,c)  a ## b ## c
                      )==";

    sourceStr += "\n#undef WARP_SCAN_FUNC_NAME\n#define WARP_SCAN_FUNC_NAME warp_incl_scan_padded"+namePostfix+"\n";

    sourceStr += "\n#undef WARP_SCAN_FUNC_NAME_SAFE\n#define WARP_SCAN_FUNC_NAME_SAFE warp_incl_scan_padded_safe"+namePostfix+"\n";

    sourceStr += "\n#undef WARP_SCAN_FUNC_NAME_SIZED\n#define WARP_SCAN_FUNC_NAME_SIZED(W) PASTER3(warp_incl_scan_padded,W,"+namePostfix+")\n";

    sourceStr += "\n#undef GETTER\n#define GETTER(IDX) "+getterFuncName+"(IDX)\n";
    sourceStr += "\n#undef SETTER\n#define SETTER(IDX,VAL) "+setterFuncName+"(IDX,VAL)\n";

    sourceStr += R"==(
//! Warning these functions will reduce across the entire warp under GL_KHR_shader_subgroup_arithmetic and GL_AMD_shader_ballot
TYPE WARP_SCAN_FUNC_NAME (in TYPE val, in uint idx);

// base case
TYPE WARP_SCAN_FUNC_NAME_SIZED(4u) (in TYPE val, in uint idx)
{
#if defined(GL_KHR_shader_subgroup_arithmetic)||defined(GL_AMD_shader_ballot)
    return WARP_SCAN_FUNC_NAME (val,idx);
#elif defined(GL_NV_shader_thread_shuffle)||defined(GL_KHR_shader_subgroup_shuffle)||defined(GL_KHR_shader_subgroup_shuffle_relative)
    TYPE tmpA = val;
    TYPE tmpB = subgroupShuffleUpEMUL(tmpA,1u);
    if (gl_ThreadInWarpNV>=1u)
        tmpA = COMM_OP(tmpA,tmpB);
    tmpB = subgroupShuffleUpEMUL(tmpA,2u);
    if (gl_ThreadInWarpNV>=2u)
        tmpA = COMM_OP(tmpA,tmpB);
    return tmpA;
#elif defined(GL_KHR_shader_subgroup_basic)
    subgroupBarrier();
    SETTER(idx+1u,COMM_OP(GETTER(idx),GETTER(idx+1u)));
    subgroupBarrier();
    SETTER(idx+2u,COMM_OP(GETTER(idx),GETTER(idx+2u)));
    subgroupBarrier();
    return GETTER(idx);
#else
    SETTER(idx+1u,COMM_OP(GETTER(idx),GETTER(idx+1u)));
    SETTER(idx+2u,COMM_OP(GETTER(idx),GETTER(idx+2u)));
    return GETTER(idx);
#endif
}

// recursive
#undef WARP_INCL_SCAN_PADDED_DECL
#define WARP_INCL_SCAN_PADDED_DECL(W)    TYPE WARP_SCAN_FUNC_NAME_SIZED(W) (in TYPE val, in uint idx)

#define GET_LOWER_WARP_SCAN_FUNC_NAME(currentSize) PASTER2(GET_LOWER_WARP_SCAN_FUNC_NAME_,currentSize)
#define GET_LOWER_WARP_SCAN_FUNC_NAME_8u WARP_SCAN_FUNC_NAME_SIZED(4u)
#define GET_LOWER_WARP_SCAN_FUNC_NAME_16u WARP_SCAN_FUNC_NAME_SIZED(8u)
#define GET_LOWER_WARP_SCAN_FUNC_NAME_32u WARP_SCAN_FUNC_NAME_SIZED(16u)
#define GET_LOWER_WARP_SCAN_FUNC_NAME_64u WARP_SCAN_FUNC_NAME_SIZED(32u)

#undef WARP_INCL_SCAN_PADDED_DEF
#if defined(GL_KHR_shader_subgroup_arithmetic)||defined(GL_AMD_shader_ballot)
    #define WARP_INCL_SCAN_PADDED_DEF(W) WARP_INCL_SCAN_PADDED_DECL(W) \
    {\
        return WARP_SCAN_FUNC_NAME (val,idx); \
    }
#elif defined(GL_NV_shader_thread_shuffle)||defined(GL_KHR_shader_subgroup_shuffle)||defined(GL_KHR_shader_subgroup_shuffle_relative)
    #define WARP_INCL_SCAN_PADDED_DEF(W) WARP_INCL_SCAN_PADDED_DECL(W) \
    {\
        TYPE tmpA = GET_LOWER_WARP_SCAN_FUNC_NAME(W) (val,idx); \
        TYPE tmpB = subgroupShuffleUpEMUL(tmpA,W/2u); \
        if (gl_ThreadInWarpNV>=(W/2u)) \
            tmpA = COMM_OP(tmpA,tmpB); \
        return tmpA; \
    }
#elif defined(GL_KHR_shader_subgroup_basic)
    #define WARP_INCL_SCAN_PADDED_DEF(W) WARP_INCL_SCAN_PADDED_DECL(W) \
    {\
        SETTER(idx+(W/2u),COMM_OP(GET_LOWER_WARP_SCAN_FUNC_NAME(W) (val,idx),GETTER(idx+(W/2u)))); \
        subgroupBarrier(); \
        return GETTER(idx); \
    }
#else
    #define WARP_INCL_SCAN_PADDED_DEF(W) WARP_INCL_SCAN_PADDED_DECL(W) \
    {\
        SETTER(idx+(W/2u),COMM_OP(GET_LOWER_WARP_SCAN_FUNC_NAME(W) (val,idx),GETTER(idx+(W/2u)))); \
        return GETTER(idx); \
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
#elif defined(CONSTANT_PROBABLE_SUBGROUP_SIZE)
    return WARP_SCAN_FUNC_NAME_SIZED(WARP_SCAN_SIZE) (val,idx);
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

    // undefine all that stuff
    sourceStr += R"==(
#undef WARP_INCL_SCAN_PADDED_DEF

#undef GET_LOWER_WARP_SCAN_FUNC_NAME
#undef GET_LOWER_WARP_SCAN_FUNC_NAME_8u
#undef GET_LOWER_WARP_SCAN_FUNC_NAME_16u
#undef GET_LOWER_WARP_SCAN_FUNC_NAME_32u
#undef GET_LOWER_WARP_SCAN_FUNC_NAME_64u

#undef WARP_INCL_SCAN_PADDED_DECL
#undef SETTER
#undef GETTER
#undef WARP_SCAN_FUNC_NAME_SIZED
#undef WARP_SCAN_FUNC_NAME_SAFE
#undef WARP_SCAN_FUNC_NAME

#undef PASTER2
#undef PASTER3

#undef TYPE
#undef COMM_OPInvocationsInclusiveScanAMD
#undef subgroupInclusiveCOMM_OP
#undef COMM_OP

#endif //include guard
                      )==";

    return sourceStr;
}

/** TODO: Much later
std::string CGLSLScanBuiltinIncludeLoader::getWarpReduceFunctionsPadded(   const E_GLSL_COMMUTATIVE_OP& oper, const E_GLSL_TYPE& dataType, const std::string& namePostfix,
                                                                    const std::string& getterFuncName, const std::string& setterFuncName)
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
