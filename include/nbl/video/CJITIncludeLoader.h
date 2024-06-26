#ifndef _NBL_VIDEO_C_JIT_INCLUDE_LOADER_H_INCLUDED_
#define _NBL_VIDEO_C_JIT_INCLUDE_LOADER_H_INCLUDED_

#include "nbl/asset/utils/IShaderCompiler.h"

#include "nbl/video/SPhysicalDeviceFeatures.h"
#include "nbl/video/SPhysicalDeviceLimits.h"

#include <string>

#include "nbl/builtin/hlsl/type_traits.hlsl"

#include "glm/gtx/string_cast.hpp"

namespace nbl::video
{
class NBL_API2 CJITIncludeLoader : public asset::IShaderCompiler::IIncludeLoader
{
    public:
        inline CJITIncludeLoader(const SPhysicalDeviceLimits& limits, const SPhysicalDeviceFeatures& features)
        {
            m_includes["nbl/builtin/hlsl/jit/device_capabilities.hlsl"] = collectDeviceCaps(limits,features);
        }

        found_t getInclude(const system::path& searchPath, const std::string& includeName) const override;

    protected:
        template<typename T>
        struct to_string_impl
        {
            inline std::string operator()(const T& object) { return std::to_string(object); }
        };

        template<typename T> requires core::Bitflag<std::remove_cvref_t<T>>
        struct to_string_impl<T>
        {
            inline std::string operator()(const T& object) {
                return std::to_string(static_cast<int>(object.value));
            }
        };

        template<typename T> requires is_scoped_enum<std::remove_cvref_t<T>>
            struct to_string_impl<T>
        {
            inline std::string operator()(const T& object) {
                return std::to_string(static_cast<int>(object));
            }
        };

        template<typename T> requires nbl::hlsl::Vector<std::remove_cvref_t<T>>
        struct to_string_impl<T>
        {
            inline std::string operator()(const T& object) {
                return glm::to_string(object);
            }
        };

    private:
        core::unordered_map<system::path,std::string> m_includes;
        std::string collectDeviceCaps(const SPhysicalDeviceLimits& limits, const SPhysicalDeviceFeatures& features);

        template<typename T>
        static inline std::string to_string(T&& object)
        {
            return to_string_impl<T>()(std::forward<T>(object));
        }

};

} //nbl::video

#endif // CJITINCLUDELOADER_H
