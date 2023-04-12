// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h



#ifndef _NBL_BUILTIN_COMMON_H_INCLUDED_
#define _NBL_BUILTIN_COMMON_H_INCLUDED_

#include "BuildConfigOptions.h"

#include <string>
#include <string_view>

#include "nbl/builtin/builtinResources.h"

namespace nbl::builtin
{

// if you attempt to use this without `NBL_EMBED_BUILTIN_RESOURCES_` CMake option, this will always return `{nullptr,0ull}`
std::pair<const uint8_t*,size_t> get_resource_runtime(const std::string&);

}
#endif
