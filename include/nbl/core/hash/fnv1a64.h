// Copyright (C) 2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_CORE_FNV1A64_H_INCLUDED_
#define _NBL_CORE_FNV1A64_H_INCLUDED_

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace nbl::core
{

// FNV-1a 64-bit hash.
constexpr uint64_t FNV1a_64(std::string_view sv)
{
    uint64_t h = 14695981039346656037ull;
    for (unsigned char c : sv)
    {
        h ^= c;
        h *= 1099511628211ull;
    }
    return h;
}

}

#endif // _NBL_CORE_FNV1A64_H_INCLUDED_
