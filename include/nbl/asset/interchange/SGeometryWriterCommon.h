// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_S_GEOMETRY_WRITER_COMMON_H_INCLUDED_
#define _NBL_ASSET_S_GEOMETRY_WRITER_COMMON_H_INCLUDED_


#include "nbl/asset/ICPUPolygonGeometry.h"

#include <charconv>
#include <cstdio>
#include <system_error>


namespace nbl::asset
{

inline const hlsl::float32_t3* getTightFloat3View(const ICPUPolygonGeometry::SDataView& view)
{
    if (!view)
        return nullptr;
    if (view.composed.format != EF_R32G32B32_SFLOAT)
        return nullptr;
    if (view.composed.getStride() != sizeof(hlsl::float32_t3))
        return nullptr;
    return reinterpret_cast<const hlsl::float32_t3*>(view.getPointer());
}

inline const hlsl::float32_t2* getTightFloat2View(const ICPUPolygonGeometry::SDataView& view)
{
    if (!view)
        return nullptr;
    if (view.composed.format != EF_R32G32_SFLOAT)
        return nullptr;
    if (view.composed.getStride() != sizeof(hlsl::float32_t2))
        return nullptr;
    return reinterpret_cast<const hlsl::float32_t2*>(view.getPointer());
}

inline char* appendFloatFixed6ToBuffer(char* dst, char* const end, const float value)
{
    if (!dst || dst >= end)
        return end;

    const auto result = std::to_chars(dst, end, value, std::chars_format::fixed, 6);
    if (result.ec == std::errc())
        return result.ptr;

    const int written = std::snprintf(dst, static_cast<size_t>(end - dst), "%.6f", static_cast<double>(value));
    if (written <= 0)
        return dst;
    const size_t writeLen = static_cast<size_t>(written);
    return (writeLen < static_cast<size_t>(end - dst)) ? (dst + writeLen) : end;
}

}


#endif
