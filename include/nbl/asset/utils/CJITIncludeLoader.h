#ifndef _NBL_ASSET_C_JIT_INCLUDE_LOADER_H_INCLUDED_
#define _NBL_ASSET_C_JIT_INCLUDE_LOADER_H_INCLUDED_

#include <string>
#include <unordered_map>

class CJITIncludeLoader
{
public:
    CJITIncludeLoader();
    std::string loadInclude(const std::string path);

private:
    std::unordered_map<std::string, std::string> m_includes;
    std::string collectDeviceCaps();
};

#endif // CJITINCLUDELOADER_H
