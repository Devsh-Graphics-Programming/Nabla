// Internal src-only header. Do not include from public headers.
#ifndef _NBL_SYSTEM_S_WIN32_PATH_UTILITIES_H_INCLUDED_
#define _NBL_SYSTEM_S_WIN32_PATH_UTILITIES_H_INCLUDED_

#ifdef _NBL_PLATFORM_WINDOWS_

#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>

namespace nbl::system::impl
{

inline std::wstring makeLongPathAwareWindowsPath(std::filesystem::path path)
{
    path = path.lexically_normal();
    if (!path.is_absolute())
    {
        std::error_code ec;
        const auto absolutePath = std::filesystem::absolute(path, ec);
        if (!ec)
            path = absolutePath.lexically_normal();
    }
    path.make_preferred();

    std::wstring native = path.native();
    constexpr std::wstring_view ExtendedPrefix = LR"(\\?\)";
    constexpr std::wstring_view UncPrefix = LR"(\\)";
    constexpr std::wstring_view ExtendedUncPrefix = LR"(\\?\UNC\)";

    if (native.rfind(ExtendedPrefix.data(), 0u) == 0u)
        return native;
    if (native.rfind(UncPrefix.data(), 0u) == 0u)
        return std::wstring(ExtendedUncPrefix) + native.substr(2u);
    return std::wstring(ExtendedPrefix) + native;
}

}

#endif

#endif
