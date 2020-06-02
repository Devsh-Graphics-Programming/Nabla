#ifndef __IRR_C_GLSL_LIMITS_BUILTIN_INCLUDE_LOADER_H_INCLUDED__
#define __IRR_C_GLSL_LIMITS_BUILTIN_INCLUDE_LOADER_H_INCLUDED__

#include <irr/asset/IBuiltinIncludeLoader.h>

namespace irr {
namespace asset
{    

class CGLSLLimitsBuiltinIncludeLoader : public IBuiltinIncludeLoader
{
public:
    const char* getVirtualDirectoryName() const override { return "glsl/limits/"; }

private:
    static std::string getNumericLimits(const std::string&)
    {
        return 
R"(#ifndef _IRR_LIMITS_NUMERIC_INCLUDED_
#define _IRR_LIMITS_NUMERIC_INCLUDED_

#ifndef UINT_MAX
#define UINT_MAX 4294967295u
#endif

#ifndef FLT_MIN
#define FLT_MIN 1.175494351e-38
#endif

#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38
#endif

#endif
)";
    }

protected:
    core::vector<std::pair<std::regex, HandleFunc_t>> getBuiltinNamesToFunctionMapping() const override
    {
        return {
            { std::regex{"numeric\\.glsl"}, &getNumericLimits }
        };
    }
};

}}

#endif