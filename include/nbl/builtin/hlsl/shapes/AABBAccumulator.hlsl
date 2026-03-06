// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SHAPES_AABB_ACCUMULATOR_INCLUDED_
#define _NBL_BUILTIN_HLSL_SHAPES_AABB_ACCUMULATOR_INCLUDED_


#include "nbl/builtin/hlsl/shapes/aabb.hlsl"
#include "nbl/builtin/hlsl/array_accessors.hlsl"
#include "nbl/builtin/hlsl/concepts/vector.hlsl"


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

template<typename Scalar>
inline AABBAccumulator3<Scalar> createAABBAccumulator()
{
    return AABBAccumulator3<Scalar>::create();
}

template<typename Scalar>
inline void extendAABBAccumulator(NBL_REF_ARG(AABBAccumulator3<Scalar>) aabb, const Scalar x, const Scalar y, const Scalar z)
{
    aabb.addXYZ(x, y, z);
}

template<typename Scalar, typename Point NBL_FUNC_REQUIRES(concepts::Vectorial<Point> && (vector_traits<Point>::Dimension >= 3))
inline void extendAABBAccumulator(NBL_REF_ARG(AABBAccumulator3<Scalar>) aabb, NBL_CONST_REF_ARG(Point) point)
{
    array_get<Point, typename vector_traits<Point>::scalar_type> getter;
    typename AABBAccumulator3<Scalar>::point_t converted;
    converted.x = Scalar(getter(point, 0));
    converted.y = Scalar(getter(point, 1));
    converted.z = Scalar(getter(point, 2));
    aabb.addPoint(converted);
}

template<int16_t D, typename DstScalar, typename SrcScalar NBL_FUNC_REQUIRES(D >= 3)
inline void assignAABBFromAccumulator(NBL_REF_ARG(AABB<D, DstScalar>) dst, NBL_CONST_REF_ARG(AABBAccumulator3<SrcScalar>) aabb)
{
    if (aabb.empty())
        return;

    dst = AABB<D, DstScalar>::create();
    array_set<typename AABB<D, DstScalar>::point_t, DstScalar> setter;
    setter(dst.minVx, 0, DstScalar(aabb.value.minVx.x));
    setter(dst.minVx, 1, DstScalar(aabb.value.minVx.y));
    setter(dst.minVx, 2, DstScalar(aabb.value.minVx.z));
    setter(dst.maxVx, 0, DstScalar(aabb.value.maxVx.x));
    setter(dst.maxVx, 1, DstScalar(aabb.value.maxVx.y));
    setter(dst.maxVx, 2, DstScalar(aabb.value.maxVx.z));
    for (int16_t i = 3; i < D; ++i)
    {
        setter(dst.minVx, i, DstScalar(0));
        setter(dst.maxVx, i, DstScalar(0));
    }
}

}
}
}
}

#endif
