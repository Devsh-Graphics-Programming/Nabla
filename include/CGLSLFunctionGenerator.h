#ifndef __IRR_C_GLSL_FUNCTION_GENERATOR_H_INCLUDED__
#define __IRR_C_GLSL_FUNCTION_GENERATOR_H_INCLUDED__

#include "IrrCompileConfig.h"
#include "IVideoCapabilityReporter.h"

#include <string>
#include <cstdint>

namespace irr { namespace video
{

class CGLSLFunctionGenerator
{
        CGLSLFunctionGenerator() = delete;
    public:
        static std::string getLinearSkinningFunction(const uint32_t& maxBoneInfluences = 4u);

        static std::string getReduceAndScanExtensionEnables(IVideoCapabilityReporter* reporter);

        static std::string getWarpScanPaddingFunctions();
        enum E_GLSL_COMMUTATIVE_OP
        {
            EGCO_ADD=0, // type supported natively by AMD
            EGCO_AND,
            EGCO_MAX, // type supported natively by AMD
            EGCO_MIN, // type supported natively by AMD
            EGCO_MUL,
            EGCO_OR,
            EGCO_XOR,
            EGCO_COUNT
        };
        enum E_GLSL_TYPE //could get some enum from shaderc files instead to not run multiple definitions
        {
            EGT_FLOAT=0,
            EGT_VEC2,
            EGT_VEC3,
            EGT_VEC4,
            EGT_COUNT
        };
        static std::string getWarpInclusiveScanFunctionsPadded(const E_GLSL_COMMUTATIVE_OP& oper, const E_GLSL_TYPE& dataType, const std::string& namePostfix, const std::string& getterFunc, const std::string& setterFunc);

        static std::string getWarpReduceFunctionsPadded(const E_GLSL_COMMUTATIVE_OP& oper, const E_GLSL_TYPE& dataType, const std::string& namePostfix, const std::string& getterFunc, const std::string& setterFunc);
};

}} // irr::video

#endif // __IRR_C_GLSL_FUNCTION_GENERATOR_H_INCLUDED__
