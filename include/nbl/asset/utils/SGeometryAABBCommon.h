// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_S_GEOMETRY_AABB_COMMON_H_INCLUDED_
#define _NBL_ASSET_S_GEOMETRY_AABB_COMMON_H_INCLUDED_


#include "nbl/asset/ICPUPolygonGeometry.h"
#include "nbl/builtin/hlsl/shapes/AABBAccumulator.hlsl"

#include <type_traits>


namespace nbl::asset
{

template<typename Scalar>
using SAABBAccumulator3 = hlsl::shapes::util::AABBAccumulator3<Scalar>;

template<typename Scalar>
inline SAABBAccumulator3<Scalar> createAABBAccumulator()
{
    return SAABBAccumulator3<Scalar>::create();
}

template<typename Scalar>
inline void extendAABBAccumulator(SAABBAccumulator3<Scalar>& aabb, const Scalar x, const Scalar y, const Scalar z)
{
    aabb.addXYZ(x, y, z);
}

template<typename Scalar, typename Point>
inline void extendAABBAccumulator(SAABBAccumulator3<Scalar>& aabb, const Point& point)
{
    typename SAABBAccumulator3<Scalar>::point_t converted;
    if constexpr (requires { point.x; point.y; point.z; })
    {
        converted.x = static_cast<Scalar>(point.x);
        converted.y = static_cast<Scalar>(point.y);
        converted.z = static_cast<Scalar>(point.z);
    }
    else
    {
        converted.x = static_cast<Scalar>(point[0]);
        converted.y = static_cast<Scalar>(point[1]);
        converted.z = static_cast<Scalar>(point[2]);
    }
    aabb.addPoint(converted);
}

template<typename Scalar, typename AABB>
inline void assignAABBFromAccumulator(AABB& dst, const SAABBAccumulator3<Scalar>& aabb)
{
    if (aabb.empty())
        return;

    dst = std::remove_reference_t<AABB>::create();
    if constexpr (requires { dst.minVx.x; dst.minVx.y; dst.minVx.z; dst.maxVx.x; dst.maxVx.y; dst.maxVx.z; })
    {
        dst.minVx.x = static_cast<decltype(dst.minVx.x)>(aabb.value.minVx.x);
        dst.minVx.y = static_cast<decltype(dst.minVx.y)>(aabb.value.minVx.y);
        dst.minVx.z = static_cast<decltype(dst.minVx.z)>(aabb.value.minVx.z);
        dst.maxVx.x = static_cast<decltype(dst.maxVx.x)>(aabb.value.maxVx.x);
        dst.maxVx.y = static_cast<decltype(dst.maxVx.y)>(aabb.value.maxVx.y);
        dst.maxVx.z = static_cast<decltype(dst.maxVx.z)>(aabb.value.maxVx.z);
        if constexpr (requires { dst.minVx.w; dst.maxVx.w; })
        {
            dst.minVx.w = 0;
            dst.maxVx.w = 0;
        }
    }
    else
    {
        dst.minVx[0] = static_cast<decltype(dst.minVx[0])>(aabb.value.minVx[0]);
        dst.minVx[1] = static_cast<decltype(dst.minVx[1])>(aabb.value.minVx[1]);
        dst.minVx[2] = static_cast<decltype(dst.minVx[2])>(aabb.value.minVx[2]);
        dst.maxVx[0] = static_cast<decltype(dst.maxVx[0])>(aabb.value.maxVx[0]);
        dst.maxVx[1] = static_cast<decltype(dst.maxVx[1])>(aabb.value.maxVx[1]);
        dst.maxVx[2] = static_cast<decltype(dst.maxVx[2])>(aabb.value.maxVx[2]);
    }
}

}


#endif
