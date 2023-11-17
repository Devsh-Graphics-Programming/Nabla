#ifndef _NBL_ASSET_C_JIT_INCLUDE_LOADER_H_INCLUDED_
#define _NBL_ASSET_C_JIT_INCLUDE_LOADER_H_INCLUDED_

#include "nbl/asset/utils/IShaderCompiler.h"
#include "nbl/video/SPhysicalDeviceFeatures.h"
#include "nbl/video/SPhysicalDeviceLimits.h"

#include <string>

using namespace nbl::asset;


namespace nbl::video
{

class NBL_API2 CJITIncludeLoader : public IShaderCompiler::IIncludeLoader
{
public:
    CJITIncludeLoader();
    std::optional<std::string> getInclude(const system::path& searchPath, const std::string& includeName) const override;

private:
    core::unordered_map<system::path, std::string> m_includes;
    std::string collectDeviceCaps(const SPhysicalDeviceLimits& limits, const SPhysicalDeviceFeatures& features);
};

} //nbl::asset

#endif // CJITINCLUDELOADER_H
