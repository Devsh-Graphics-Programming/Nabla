// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SHAPES_AABB_INCLUDED_
#define _NBL_BUILTIN_HLSL_SHAPES_AABB_INCLUDED_

#include "nbl/builtin/hlsl/concepts.hlsl"
#include "nbl/builtin/hlsl/limits.hlsl"
#include "nbl/builtin/hlsl/shapes/util.hlsl"

namespace nbl
{
namespace hlsl
{
namespace shapes
{

template<int16_t D=3, typename Scalar=float32_t>
struct AABB
{
    using point_t = vector<Scalar,D>;

    static AABB create()
    {
        AABB retval;
        retval.minVx = promote<point_t>(numeric_limits<Scalar>::max);
        retval.maxVx = promote<point_t>(numeric_limits<Scalar>::lowest);
        return retval;
    }

    //
    void addPoint(const point_t pt)
    {
        minVx = min<point_t>(pt, minVx);
        maxVx = max<point_t>(pt, maxVx);
    }
    //
    point_t getExtent()
    {
        return maxVx - minVx;
    }

    //
    Scalar getVolume()
    {
        const point_t extent = getExtent();
        return extent.x * extent.y * extent.z;
    }

    // returns the corner of the AABB which has the most positive dot product
    point_t getFarthestPointInFront(const point_t planeNormal)
    {
        return hlsl::mix(maxVx, minVx, hlsl::lessThan(planeNormal,promote<point_t>(0.f)));
    }

    point_t minVx;
    point_t maxVx;
};

namespace util
{
namespace impl
{
template<int16_t D, typename Scalar>
struct intersect_helper<AABB<D,Scalar>>
{
    using type = AABB<D,Scalar>;

    static inline type __call(NBL_CONST_REF_ARG(type) lhs, NBL_CONST_REF_ARG(type) rhs)
    {
        type retval;
        retval.minVx = max<type::point_t>(lhs.minVx,rhs.minVx);
        retval.maxVx = min<type::point_t>(lhs.maxVx,rhs.maxVx);
        return retval;
    }
};
template<int16_t D, typename Scalar>
struct union_helper<AABB<D,Scalar>>
{
    using type = AABB<D,Scalar>;

    static inline type __call(NBL_CONST_REF_ARG(type) lhs, NBL_CONST_REF_ARG(type) rhs)
    {
        type retval;
        retval.minVx = min<type::point_t>(lhs.minVx,rhs.minVx);
        retval.maxVx = max<type::point_t>(lhs.maxVx,rhs.maxVx);
        return retval;
    }
};
}
}

#if 0 // experimental
#include <nbl/builtin/hlsl/format/shared_exp.hlsl>
struct CompressedAABB_t
{
    //
    AABB_t decompress()
    {
        AABB_t retval;
        retval.minVx = decodeRGB18E7S3(minVx18E7S3);
        retval.maxVx = decodeRGB18E7S3(maxVx18E7S3);
        return retval;
    }

    uint2 minVx18E7S3;
    uint2 maxVx18E7S3;
};
#endif


}
}
}

#endif
