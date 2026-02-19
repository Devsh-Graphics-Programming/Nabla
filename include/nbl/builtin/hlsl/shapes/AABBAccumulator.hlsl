// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SHAPES_AABB_ACCUMULATOR_INCLUDED_
#define _NBL_BUILTIN_HLSL_SHAPES_AABB_ACCUMULATOR_INCLUDED_


#include "nbl/builtin/hlsl/shapes/aabb.hlsl"


namespace nbl
{
namespace hlsl
{
namespace shapes
{
namespace util
{

template<typename Scalar = float32_t>
struct AABBAccumulator3
{
    using scalar_t = Scalar;
    using aabb_t = AABB<3, Scalar>;
    using point_t = typename aabb_t::point_t;

    static AABBAccumulator3 create()
    {
        AABBAccumulator3 retval = {};
        retval.value = aabb_t::create();
        return retval;
    }

    bool empty() NBL_CONST_MEMBER_FUNC
    {
        return
            value.minVx.x > value.maxVx.x ||
            value.minVx.y > value.maxVx.y ||
            value.minVx.z > value.maxVx.z;
    }

    void addPoint(NBL_CONST_REF_ARG(point_t) point)
    {
        value.addPoint(point);
    }

    void addXYZ(const Scalar x, const Scalar y, const Scalar z)
    {
        point_t point;
        point.x = x;
        point.y = y;
        point.z = z;
        value.addPoint(point);
    }

    aabb_t value;
};

}
}
}
}

#endif
