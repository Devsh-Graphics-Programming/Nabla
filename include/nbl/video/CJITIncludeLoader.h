#ifndef _NBL_VIDEO_C_JIT_INCLUDE_LOADER_H_INCLUDED_
#define _NBL_VIDEO_C_JIT_INCLUDE_LOADER_H_INCLUDED_

#include "nbl/asset/utils/IShaderCompiler.h"

#include "nbl/video/SPhysicalDeviceFeatures.h"
#include "nbl/video/SPhysicalDeviceLimits.h"

#include <string>


namespace nbl::video
{

namespace impl
{
    template<typename T>
    struct to_string
    {
        inline std::string operator()(const T& object) { return std::to_string(object); }
    };

    template<core::Bitflag T>
    struct to_string<T>
    {
        inline std::string operator()(const T& object) {
            return "Test";
            //return std::to_string(static_cast<T::UNDERLYING_TYPE>(object.value)); 
        }
    };
}

class NBL_API2 CJITIncludeLoader : public asset::IShaderCompiler::IIncludeLoader
{
    public:
        inline CJITIncludeLoader(const SPhysicalDeviceLimits& limits, const SPhysicalDeviceFeatures& features)
        {
            m_includes["nbl/builtin/hlsl/jit/device_capabilities.hlsl"] = collectDeviceCaps(limits,features);
        }

        found_t getInclude(const system::path& searchPath, const std::string& includeName) const override;

    private:
        core::unordered_map<system::path,std::string> m_includes;
        std::string collectDeviceCaps(const SPhysicalDeviceLimits& limits, const SPhysicalDeviceFeatures& features);

        template<typename T>
        static inline std::string to_string(T&& object)
        {
            return impl::to_string<T>()(std::forward<T>(object));
        }

};

} //nbl::video

#endif // CJITINCLUDELOADER_H
