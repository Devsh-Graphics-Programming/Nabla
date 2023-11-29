#include "nbl/asset/utils/CJITIncludeLoader.h"

namespace nbl::video
{

CJITIncludeLoader::CJITIncludeLoader(const SPhysicalDeviceLimits& limits, const SPhysicalDeviceFeatures& features)
{
    m_includes["nbl/builtin/hlsl/jit/device_capabilities.hlsl"] = collectDeviceCaps(limits, features);
}

std::optional<std::string> CJITIncludeLoader::getInclude(const system::path& searchPath, const std::string& includeName) const
{
    system::path path = searchPath / includeName;

    assert(searchPath == "nbl/builtin/hlsl/jit");
    
    // Look up the content in m_includes map
    auto it = m_includes.find(path);
    if (it != m_includes.end())
    {
        // Return the content of the specified include file
        return it->second;
    }

    return std::nullopt;
}


std::string CJITIncludeLoader::collectDeviceCaps(const SPhysicalDeviceLimits& limits, const SPhysicalDeviceFeatures& features)
{
    return R"===(
        #ifndef _NBL_BUILTIN_HLSL_JIT_DEVICE_CAPABILITIES_INCLUDED_
        #define _NBL_BUILTIN_HLSL_JIT_DEVICE_CAPABILITIES_INCLUDED_

        namespace nbl
        {
        namespace hlsl
        {
        namespace jit
        {
            struct device_capabilities
            {
                NBL_CONSTEXPR_STATIC_INLINE bool shaderFloat64 = )===" + std::to_string(features.shaderFloat64) + R"===(;
                NBL_CONSTEXPR_STATIC_INLINE bool shaderDrawParameters = )===" + std::to_string(features.shaderDrawParameters) + R"===(;
                NBL_CONSTEXPR_STATIC_INLINE bool subgroupArithmetic = )===" + std::to_string(limits.shaderSubgroupArithmetic) + R"===(;
                NBL_CONSTEXPR_STATIC_INLINE bool fragmentShaderPixelInterlock = )===" + std::to_string(features.fragmentShaderPixelInterlock) + R"===(;

                NBL_CONSTEXPR_STATIC_INLINE uint16_t maxOptimallyResidentWorkgroupInvocations = )===" + std::to_string(limits.maxOptimallyResidentWorkgroupInvocations) + R"===(;
            };
        }
        }
        }

        #endif  // _NBL_BUILTIN_HLSL_JIT_DEVICE_CAPABILITIES_INCLUDED_
    )===";
}

}