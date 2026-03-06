// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/asset/interchange/SGeometryWriterCommon.h"

#include <array>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <limits>
#include <system_error>
#include <type_traits>


namespace nbl::asset
{

namespace
{

template<typename T>
inline constexpr size_t FloatingPointScratchSize = std::numeric_limits<T>::max_digits10 + 9ull;

template<typename T>
char* appendFloatingPointToBuffer(char* dst, char* const end, const T value)
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

    if (!dst || dst >= end)
        return end;

    const auto result = std::to_chars(dst, end, value);
    if (result.ec == std::errc())
        return result.ptr;

    std::array<char, FloatingPointScratchSize<T>> scratch = {};
    constexpr int Precision = std::numeric_limits<T>::max_digits10;
    const int written = std::snprintf(scratch.data(), scratch.size(), "%.*g", Precision, static_cast<double>(value));
    if (written <= 0)
        return dst;

    const size_t writeLen = static_cast<size_t>(written);
    if (writeLen > static_cast<size_t>(end - dst))
        return end;

    std::memcpy(dst, scratch.data(), writeLen);
    return dst + writeLen;
}

}

char* SGeometryWriterCommon::appendFloatToBuffer(char* dst, char* end, float value)
{
    return appendFloatingPointToBuffer(dst, end, value);
}

char* SGeometryWriterCommon::appendFloatToBuffer(char* dst, char* end, double value)
{
    return appendFloatingPointToBuffer(dst, end, value);
}

}
