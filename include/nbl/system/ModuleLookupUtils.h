#ifndef _NBL_SYSTEM_MODULE_LOOKUP_UTILS_H_INCLUDED_
#define _NBL_SYSTEM_MODULE_LOOKUP_UTILS_H_INCLUDED_

#include "nbl/system/path.h"

#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>

#if defined(_NBL_PLATFORM_WINDOWS_)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace nbl::system
{
inline bool moduleExistsInDirectory(const system::path& dir, std::string_view moduleName)
{
    if (dir.empty() || moduleName.empty() || !std::filesystem::exists(dir) || !std::filesystem::is_directory(dir))
        return false;

    const std::string baseName(moduleName);
    const auto hasRegularFile = [&dir](const std::string& fileName)
    {
        const auto filePath = dir / fileName;
        return std::filesystem::exists(filePath) && std::filesystem::is_regular_file(filePath);
    };

    if (hasRegularFile(baseName))
        return true;

    #if defined(_NBL_PLATFORM_WINDOWS_)
    if (hasRegularFile(baseName + ".dll"))
        return true;
    #elif defined(_NBL_PLATFORM_LINUX_) || defined(_NBL_PLATFORM_ANDROID_)
    if (hasRegularFile(baseName + ".so"))
        return true;

    const bool hasLibPrefix = (baseName.rfind("lib", 0) == 0);
    const std::string libBaseName = hasLibPrefix ? baseName : ("lib" + baseName);
    if (hasRegularFile(libBaseName + ".so"))
        return true;

    const std::string versionedPrefix = libBaseName + ".so.";
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec))
    {
        if (ec)
            break;
        if (!entry.is_regular_file(ec))
            continue;

        const auto fileName = entry.path().filename().string();
        if (fileName.rfind(versionedPrefix, 0) == 0)
            return true;
    }
    #elif defined(__APPLE__)
    if (hasRegularFile(baseName + ".dylib"))
        return true;

    const bool hasLibPrefix = (baseName.rfind("lib", 0) == 0);
    if (!hasLibPrefix && hasRegularFile("lib" + baseName + ".dylib"))
        return true;
    #endif

    return false;
}

inline system::path executableDirectory()
{
    #if defined(_NBL_PLATFORM_WINDOWS_)
    wchar_t modulePath[MAX_PATH] = {};
    const auto length = GetModuleFileNameW(nullptr, modulePath, MAX_PATH);
    if ((length == 0) || (length >= MAX_PATH))
        return system::path("");
    return std::filesystem::path(modulePath).parent_path();
    #elif defined(_NBL_PLATFORM_LINUX_) || defined(_NBL_PLATFORM_ANDROID_)
    std::error_code ec;
    const auto executablePath = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (ec)
        return system::path("");
    return executablePath.parent_path();
    #else
    return system::path("");
    #endif
}

inline system::path loadedModuleDirectory(std::string_view moduleName)
{
    #if defined(_NBL_PLATFORM_WINDOWS_)
    if (moduleName.empty())
        return system::path("");

    const auto moduleHandle = GetModuleHandleA(moduleName.data());
    if (moduleHandle == nullptr)
        return system::path("");

    wchar_t modulePath[MAX_PATH] = {};
    const auto length = GetModuleFileNameW(moduleHandle, modulePath, MAX_PATH);
    if ((length == 0) || (length >= MAX_PATH))
        return system::path("");

    return std::filesystem::path(modulePath).parent_path();
    #else
    // TODO: implement loaded module directory lookup for non-Windows platforms.
    return system::path("");
    #endif
}
}

#endif
