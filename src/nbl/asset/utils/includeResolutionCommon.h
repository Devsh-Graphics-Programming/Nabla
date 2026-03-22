// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_INCLUDE_RESOLUTION_COMMON_H_INCLUDED_
#define _NBL_ASSET_INCLUDE_RESOLUTION_COMMON_H_INCLUDED_

#include <string_view>

namespace nbl::asset::detail
{
inline bool isGloballyResolvedIncludeName(std::string_view includeName)
{
    constexpr std::string_view globalPrefixes[] = {
        "nbl/",
        "nbl\\",
        "boost/",
        "boost\\",
        "glm/",
        "glm\\",
        "spirv/",
        "spirv\\",
        "Imath/",
        "Imath\\"
    };

    for (const auto prefix : globalPrefixes)
    {
        if (includeName.rfind(prefix, 0ull) == 0ull)
            return true;
    }

    return false;
}
}

#endif
