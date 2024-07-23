#include <bit>

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
    #include "nbl/video/device_capabilities_traits_jit.h"

    std::string start = R"===(
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
                )===";

    std::string end = R"===(    };
        
        using device_capabilities_traits = nbl::hlsl::device_capabilities_traits<device_capabilities>;
        }
        }
        }
        
        #endif  // _NBL_BUILTIN_HLSL_JIT_DEVICE_CAPABILITIES_INCLUDED_
            )===";

    return start + jit_traits + end;
}
}