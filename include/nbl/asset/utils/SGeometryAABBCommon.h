// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_S_GEOMETRY_AABB_COMMON_H_INCLUDED_
#define _NBL_ASSET_S_GEOMETRY_AABB_COMMON_H_INCLUDED_


#include "nbl/asset/ICPUPolygonGeometry.h"

#include <array>
#include <type_traits>


namespace nbl::asset
{

template<typename Scalar>
struct SAABBAccumulator3
{
    bool has = false;
    std::array<Scalar, 3> min = {};
    std::array<Scalar, 3> max = {};
};

template<typename Scalar>
inline void extendAABBAccumulator(SAABBAccumulator3<Scalar>& aabb, const Scalar x, const Scalar y, const Scalar z)
{
    if (!aabb.has)
    {
        aabb.has = true;
        aabb.min[0] = x;
        aabb.min[1] = y;
        aabb.min[2] = z;
        aabb.max[0] = x;
        aabb.max[1] = y;
        aabb.max[2] = z;
        return;
    }

    if (x < aabb.min[0]) aabb.min[0] = x;
    if (y < aabb.min[1]) aabb.min[1] = y;
    if (z < aabb.min[2]) aabb.min[2] = z;
    if (x > aabb.max[0]) aabb.max[0] = x;
    if (y > aabb.max[1]) aabb.max[1] = y;
    if (z > aabb.max[2]) aabb.max[2] = z;
}

template<typename Scalar, typename Point>
inline void extendAABBAccumulator(SAABBAccumulator3<Scalar>& aabb, const Point& point)
{
    if constexpr (requires { point.x; point.y; point.z; })
        extendAABBAccumulator(aabb, static_cast<Scalar>(point.x), static_cast<Scalar>(point.y), static_cast<Scalar>(point.z));
    else
        extendAABBAccumulator(aabb, static_cast<Scalar>(point[0]), static_cast<Scalar>(point[1]), static_cast<Scalar>(point[2]));
}

template<typename Scalar, typename AABB>
inline void assignAABBFromAccumulator(AABB& dst, const SAABBAccumulator3<Scalar>& aabb)
{
    if (!aabb.has)
        return;

    dst = std::remove_reference_t<AABB>::create();
    if constexpr (requires { dst.minVx.x; dst.minVx.y; dst.minVx.z; dst.maxVx.x; dst.maxVx.y; dst.maxVx.z; })
    {
        dst.minVx.x = static_cast<decltype(dst.minVx.x)>(aabb.min[0]);
        dst.minVx.y = static_cast<decltype(dst.minVx.y)>(aabb.min[1]);
        dst.minVx.z = static_cast<decltype(dst.minVx.z)>(aabb.min[2]);
        dst.maxVx.x = static_cast<decltype(dst.maxVx.x)>(aabb.max[0]);
        dst.maxVx.y = static_cast<decltype(dst.maxVx.y)>(aabb.max[1]);
        dst.maxVx.z = static_cast<decltype(dst.maxVx.z)>(aabb.max[2]);
        if constexpr (requires { dst.minVx.w; dst.maxVx.w; })
        {
            dst.minVx.w = 0;
            dst.maxVx.w = 0;
        }
    }
    else
    {
        dst.minVx[0] = static_cast<decltype(dst.minVx[0])>(aabb.min[0]);
        dst.minVx[1] = static_cast<decltype(dst.minVx[1])>(aabb.min[1]);
        dst.minVx[2] = static_cast<decltype(dst.minVx[2])>(aabb.min[2]);
        dst.maxVx[0] = static_cast<decltype(dst.maxVx[0])>(aabb.max[0]);
        dst.maxVx[1] = static_cast<decltype(dst.maxVx[1])>(aabb.max[1]);
        dst.maxVx[2] = static_cast<decltype(dst.maxVx[2])>(aabb.max[2]);
    }
}

template<typename Scalar>
inline void applyAABBToGeometry(ICPUPolygonGeometry* geometry, const SAABBAccumulator3<Scalar>& aabb)
{
    if (!geometry || !aabb.has)
        return;

    geometry->visitAABB([&aabb](auto& ref)->void
    {
        assignAABBFromAccumulator(ref, aabb);
    });
}

}


#endif
