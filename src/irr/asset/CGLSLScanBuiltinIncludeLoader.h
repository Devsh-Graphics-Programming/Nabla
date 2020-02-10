#ifndef __IRR_C_GLSL_SCAN_BUILTIN_LOADER_H_INCLUDED__
#define __IRR_C_GLSL_SCAN_BUILTIN_LOADER_H_INCLUDED__

#include "IrrCompileConfig.h"
#include "IVideoCapabilityReporter.h"
#include "irr/asset/IBuiltinIncludeLoader.h"

#include <string>
#include <cstdint>
#include <cassert>

namespace irr { namespace asset
{

class CGLSLScanBuiltinIncludeLoader : public asset::IBuiltinIncludeLoader
{
public:
    const char* getVirtualDirectoryName() const override { return "glsl/scan/"; }

protected:
    core::vector<std::pair<std::regex, HandleFunc_t>> getBuiltinNamesToFunctionMapping() const override;

private:
    std::string getReduceAndScanExtensionEnables() const;

    static std::string getWarpPaddingFunctions();

    enum E_GLSL_COMMUTATIVE_OP
    {
        EGCO_ADD=0, // type supported natively by AMD
        EGCO_AND,
        EGCO_MAX, // type supported natively by AMD
        EGCO_MIN, // type supported natively by AMD
        //EGCO_MUL, // hard to implement in our framework
        EGCO_OR,
        EGCO_XOR,
        EGCO_COUNT
    };
    static std::string getCommOpDefine(const E_GLSL_COMMUTATIVE_OP& oper);

    enum E_GLSL_TYPE //could get some enum from shaderc files instead to not run multiple definitions
    {
        EGT_FLOAT=0,
        EGT_VEC2,
        EGT_VEC3,
        EGT_VEC4,
        EGT_COUNT
    };
    static std::string getTypeDef(const E_GLSL_TYPE& type);

    static std::string getWarpInclusiveScanFunctionsPadded(const E_GLSL_COMMUTATIVE_OP& oper, const E_GLSL_TYPE& dataType, const std::string& namePostfix, const std::string& getterFuncName, const std::string& setterFuncName);
    /*
    static uint32_t getNeededSharedElementsForBlockScan(const uint32_t& elementCount, const uint32_t& fixedSubgroupSize) {assert(false); return 0u;} //! TO IMPLEMENT THIS
    static inline uint32_t getNeededSharedElementsForBlockScan(const uint32_t& blockSize)
    {
    uint32_t maxMemSize = blockSize;
    for (uint32_t subgroupSize=4u; subgroupSize<=64u; subgroupSize*=2)
    {
        auto tmp = getNeededSharedElementsForBlockScan(blockSize,subgroupSize);
        if (tmp>maxMemSize)
            maxMemSize = tmp;
    }
    return maxMemSize;
    }
    */
    static std::string getBlockInclusiveScanFunctionsPadded(const uint32_t& elementsToReduce, const uint32_t& workgroupSize, const E_GLSL_COMMUTATIVE_OP& oper, const E_GLSL_TYPE& dataType,
                                                        const std::string& namePostfix, const std::string& getterFuncName, const std::string& setterFuncName)
    {
    return "\n#error \"UNIMPLEMENTED\"\n";
    }

    //! TODO: Later
    //static std::string getWarpReduceFunctionsPadded(const E_GLSL_COMMUTATIVE_OP& oper, const E_GLSL_TYPE& dataType, const std::string& namePostfix, const std::string& getterFuncName, const std::string& setterFuncName);
};

}}

#endif //__IRR_C_GLSL_SCAN_BUILTIN_LOADER_H_INCLUDED__