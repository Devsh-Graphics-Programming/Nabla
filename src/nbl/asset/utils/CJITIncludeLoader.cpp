
#include "CJITIncludeLoader.h"


CJITIncludeLoader::CJITIncludeLoader()
{
    m_includes["device_capabilities.hlsl"] = collectDeviceCaps();
}

std::string CJITIncludeLoader::loadInclude(const std::string path)
{
    const std::string prefix = "nbl/builtin/hlsl/jit/";
    if (path.compare(0, prefix.size(), prefix) == 0)
    {
        std::string includeFileName = path.substr(prefix.size());

        // Look up the content in m_includes map
        auto it = m_includes.find(includeFileName);
        if (it != m_includes.end())
        {
            // Return the content of the specified include file
            return it->second;
        }
        else
        {
            // Handle error: include file not found
            std::cerr << "Error: Include file '" << path << "' not found!" << std::endl;
            return "";
        }
    }
    else
    {
        // Handle error: invalid include path
        std::cerr << "Error: Invalid include path '" << path << "'!" << std::endl;
        return "";
    }
}


std::string CJITIncludeLoader::collectDeviceCaps()
{
    std::ostringstream content;
    content << "#ifndef _NBL_BUILTIN_HLSL_JIT_DEVICE_CAPABILITIES_INCLUDED_" << std::endl;
    content << "#define _NBL_BUILTIN_HLSL_JIT_DEVICE_CAPABILITIES_INCLUDED_" << std::endl;
    content << "namespace nbl {" << std::endl;
    content << "namespace hlsl {" << std::endl;
    content << "namespace jit {" << std::endl;
    content << "struct device_capabilities {" << std::endl;
    content << "  NBL_CONSTEXPR_STATIC_INLINE bool shaderFloat64 = false" << std::endl;
    content << "  NBL_CONSTEXPR_STATIC_INLINE bool shaderDrawParameters = false" << std::endl;
    content << "  NBL_CONSTEXPR_STATIC_INLINE bool subgroupArithmetic = false" << std::endl;
    content << "  NBL_CONSTEXPR_STATIC_INLINE bool fragmentShaderPixelInterlock = false" << std::endl;
    content << std::endl;
    content << "  NBL_CONSTEXPR_STATIC_INLINE uint16_t maxOptimallyResidentWorkgroupInvocations = 0" << std::endl;
    content << "};" << std::endl;
    content << "}" << std::endl;
    content << "}" << std::endl;
    content << "#endif" << std::endl;

    return content.str();
}