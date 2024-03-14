#include "nbl/video/CJITIncludeLoader.h"

namespace nbl::video
{
auto CJITIncludeLoader::getInclude(const system::path& searchPath, const std::string& includeName) const -> found_t
{
    assert(searchPath=="nbl/builtin/hlsl/jit");
    
    // Look up the content in m_includes map
    auto it = m_includes.find(includeName);
    if (it!=m_includes.end())
        return {includeName,it->second};

    return {};
}


std::string CJITIncludeLoader::collectDeviceCaps(const SPhysicalDeviceLimits& limits, const SPhysicalDeviceFeatures& features)
{
    return R"===(
#ifndef _NBL_BUILTIN_HLSL_JIT_DEVICE_CAPABILITIES_INCLUDED_
#define _NBL_BUILTIN_HLSL_JIT_DEVICE_CAPABILITIES_INCLUDED_

#include <nbl/builtin/hlsl/device_capabilities_traits.hlsl>

namespace nbl
{
namespace hlsl
{
namespace jit
{
    struct device_capabilities
    {
        NBL_CONSTEXPR_STATIC_INLINE bool shaderFloat64 = )===" + std::to_string(limits.shaderFloat64) + R"===(;
        NBL_CONSTEXPR_STATIC_INLINE bool subgroupArithmetic = )===" + std::to_string(limits.shaderSubgroupArithmetic) + R"===(;
        NBL_CONSTEXPR_STATIC_INLINE bool fragmentShaderPixelInterlock = )===" + std::to_string(features.fragmentShaderPixelInterlock) + R"===(;

        NBL_CONSTEXPR_STATIC_INLINE uint16_t maxOptimallyResidentWorkgroupInvocations = )===" + std::to_string(limits.maxOptimallyResidentWorkgroupInvocations) + R"===(;
        NBL_CONSTEXPR_STATIC_INLINE uint32_t maxResidentInvocations = )===" + std::to_string(limits.maxResidentInvocations) + R"===(;
    };

//TODO: when `device_capabilities_traits` is ready
//typedef nbl::hlsl::device_capabilities_traits<device_capabilities> device_capabilities_traits;
//for now just alias them
typedef device_capabilities device_capabilities_traits;
}
}
}

#endif  // _NBL_BUILTIN_HLSL_JIT_DEVICE_CAPABILITIES_INCLUDED_
    )===";
}
}